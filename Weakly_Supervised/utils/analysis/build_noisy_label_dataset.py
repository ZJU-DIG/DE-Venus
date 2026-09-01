"""Advance noisy-label parquet after probe: relabel rows when probe majority differs from current GT."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .probes import _answers_equal, analyze_probe_summary_file, load_probe_summaries, save_analysis
from .build_stage2_dataset import _is_labeled_extra_info
from .stage2_selection import (
    enrich_relabel_candidates_from_parquet,
    promotion_summary,
    save_relabel_manifest,
    select_relabel_noisy_label_samples,
)


def ensure_noisy_label_parquet_all_labeled(
    parquet_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Mark every row ``extra_info.labeled=True`` (NL-OLR v2 uses TraPO v2 labeled-only warmup on full table)."""
    from datasets import Dataset, load_dataset

    parquet_path = Path(parquet_path)
    out_path = Path(output_path) if output_path is not None else parquet_path
    ds = load_dataset("parquet", data_files=str(parquet_path))["train"]
    rows: list[dict[str, Any]] = []
    n_fixed = 0
    for row_pos in range(len(ds)):
        row = dict(ds[row_pos])
        extra: dict[str, Any] = dict(row.get("extra_info") or {})
        if not _is_labeled_extra_info(extra):
            extra["labeled"] = True
            row["extra_info"] = extra
            n_fixed += 1
        rows.append(row)
    stats = {
        "source": str(parquet_path),
        "output": str(out_path),
        "n_rows": len(rows),
        "n_set_labeled_true": n_fixed,
    }
    if n_fixed == 0 and out_path.resolve() == parquet_path.resolve():
        return stats
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(out_path))
    return stats


def _dataset_index_from_row(row: dict[str, Any], row_pos: int) -> int:
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict) and extra.get("index") is not None:
        return int(extra["index"])
    return row_pos


def _original_ground_truth(row: dict[str, Any]) -> Any:
    extra = row.get("extra_info") or {}
    rm = row.get("reward_model") or {}
    if isinstance(extra, dict) and extra.get("original_ground_truth") is not None:
        return extra["original_ground_truth"]
    if isinstance(rm, dict) and rm.get("real_gt") is not None:
        return rm["real_gt"]
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return None


def advance_noisy_label_parquet(
    source_parquet: str | Path,
    output_parquet: str | Path,
    relabel_candidates: list[dict[str, Any]],
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    relabel_iteration: int | None = None,
) -> dict[str, Any]:
    """Rewrite ``reward_model.ground_truth`` when probe majority differs from the current label."""
    from datasets import Dataset, load_dataset

    source_parquet = Path(source_parquet)
    output_parquet = Path(output_parquet)
    by_index = {int(p["sample_index"]): p for p in relabel_candidates}

    ds = load_dataset("parquet", data_files=str(source_parquet))["train"]
    out_rows: list[dict[str, Any]] = []
    n_relabeled = 0
    n_candidate_same_gt = 0
    n_not_selected = 0

    for row_pos in range(len(ds)):
        row = dict(ds[row_pos])
        idx = _dataset_index_from_row(row, row_pos)
        rm: dict[str, Any] = dict(row.get("reward_model") or {})
        extra: dict[str, Any] = dict(row.get("extra_info") or {})
        current_gt = rm.get("ground_truth")

        if idx not in by_index:
            out_rows.append(row)
            n_not_selected += 1
            continue

        cand = by_index[idx]
        pseudo_gt = cand["pseudo_gt"]
        if _answers_equal(pseudo_gt, current_gt):
            out_rows.append(row)
            n_candidate_same_gt += 1
            continue

        new_row = copy.deepcopy(row)
        if "original_ground_truth" not in extra and current_gt is not None:
            extra["original_ground_truth"] = current_gt
        if extra.get("initial_ground_truth") is None and current_gt is not None:
            extra["initial_ground_truth"] = extra.get("original_ground_truth", current_gt)
        extra["labeled"] = True
        extra["relabel_iteration"] = int(relabel_iteration) if relabel_iteration is not None else extra.get(
            "relabel_iteration"
        )
        extra["relabel_passrate_min"] = float(passrate_min)
        if slope_min is not None:
            extra["relabel_slope_min"] = float(slope_min)
        extra["pseudo_final_passrate"] = float(cand["final_passrate"])
        extra["pseudo_first_passrate"] = float(cand["first_passrate"])
        extra["pseudo_slope"] = float(cand["slope"])
        extra["initial_slope_at_relabel"] = float(cand.get("initial_slope", 0.0))
        extra["initial_final_passrate_at_relabel"] = float(cand.get("initial_final_passrate", 0.0))
        extra["initial_gt_in_rollouts"] = bool(cand.get("initial_gt_in_rollouts", False))
        extra["slope_delta_vs_initial"] = float(cand.get("slope_delta_vs_initial", 0.0))
        extra["dataset_index"] = idx

        rm["ground_truth"] = pseudo_gt
        rm["original_gt"] = current_gt
        new_row["extra_info"] = extra
        new_row["reward_model"] = rm
        out_rows.append(new_row)
        n_relabeled += 1

    out_ds = Dataset.from_list(out_rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_parquet(str(output_parquet))

    from .build_stage2_dataset import count_labeled_unlabeled_rows

    counts = count_labeled_unlabeled_rows(output_parquet)
    return {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "n_source_rows": len(ds),
        "n_output_rows": len(out_rows),
        "n_relabeled": n_relabeled,
        "n_candidate_same_gt": n_candidate_same_gt,
        "n_not_selected": n_not_selected,
        "relabel_summary": promotion_summary(relabel_candidates),
        **counts,
    }


def advance_noisy_label_parquet_from_probe_dir(
    source_parquet: str | Path,
    probe_dir: str | Path,
    output_parquet: str | Path,
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    slope_margin: float = 0.05,
    passrate_margin: float | None = None,
    hist_len: int = 8,
    relabel_iteration: int | None = None,
    manifest_path: str | Path | None = None,
    rebuild_analysis: bool = False,
) -> dict[str, Any]:
    probe_dir = Path(probe_dir)
    summary_path = probe_dir / "all_probes_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing probe summary: {summary_path}")

    analysis_path = probe_dir / "all_answer_analysis.json"
    if rebuild_analysis or not analysis_path.is_file():
        analysis = analyze_probe_summary_file(summary_path, sample_type="all")
        save_analysis(analysis, analysis_path)
    else:
        with open(analysis_path, encoding="utf-8") as f:
            analysis = json.load(f)

    candidates = select_relabel_noisy_label_samples(
        analysis,
        load_probe_summaries(summary_path),
        passrate_min=passrate_min,
        slope_min=slope_min,
        slope_margin=slope_margin,
        passrate_margin=passrate_margin,
        hist_len=hist_len,
        source_parquet=source_parquet,
    )

    manifest = manifest_path or probe_dir / "relabel_manifest.json"
    manifest_rows = enrich_relabel_candidates_from_parquet(candidates, source_parquet)
    save_relabel_manifest(
        manifest_rows,
        manifest,
        passrate_min=passrate_min,
        slope_min=slope_min,
        hist_len=hist_len,
        source_parquet=source_parquet,
        slope_margin=slope_margin,
        passrate_margin=passrate_margin,
        extra_criteria={
            "comparison_baseline": "reward_model.ground_truth",
            "initial_gt_missing_from_rollouts_slope_zero": True,
        },
    )

    stats = advance_noisy_label_parquet(
        source_parquet,
        output_parquet,
        candidates,
        passrate_min=passrate_min,
        slope_min=slope_min,
        relabel_iteration=relabel_iteration,
    )
    stats["manifest_path"] = str(manifest)
    stats["passrate_min"] = passrate_min
    stats["slope_min"] = slope_min
    stats["slope_margin"] = slope_margin
    stats["passrate_margin"] = passrate_margin
    stats["hist_len"] = hist_len
    if relabel_iteration is not None:
        stats["relabel_iteration"] = relabel_iteration

    meta_path = Path(output_parquet).with_suffix(".advance_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    stats["meta_path"] = str(meta_path)
    return stats


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Refine noisy-label parquet from probe dumps.")
    parser.add_argument("source_parquet", type=Path)
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--passrate-min", type=float, default=0.35)
    parser.add_argument("--slope-min", type=float, default=0.05)
    parser.add_argument(
        "--slope-margin",
        type=float,
        default=0.05,
        help="Require majority_slope > initial_slope + margin (initial missing in rollouts -> slope 0).",
    )
    parser.add_argument(
        "--passrate-margin",
        type=float,
        default=-1.0,
        help="If >= 0, require majority_final >= initial_final + margin.",
    )
    parser.add_argument("--hist-len", type=int, default=8)
    parser.add_argument("--relabel-iteration", type=int, default=None)
    parser.add_argument("--rebuild-analysis", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (args.probe_dir / "train_next.parquet")
    slope_min = None if args.slope_min < 0 else args.slope_min
    passrate_margin = None if args.passrate_margin < 0 else args.passrate_margin
    stats = advance_noisy_label_parquet_from_probe_dir(
        args.source_parquet,
        args.probe_dir,
        output,
        passrate_min=args.passrate_min,
        slope_min=slope_min,
        slope_margin=args.slope_margin,
        passrate_margin=passrate_margin,
        hist_len=args.hist_len,
        relabel_iteration=args.relabel_iteration,
        rebuild_analysis=args.rebuild_analysis,
    )
    summ = stats["relabel_summary"]
    print("=== Advance noisy-label parquet ===")
    print(f"rows: {stats['n_source_rows']} -> {stats['n_output_rows']}")
    print(
        f"relabeled={stats['n_relabeled']}  same_gt_skipped={stats['n_candidate_same_gt']}  "
        f"not_selected={stats['n_not_selected']}"
    )
    print(
        f"selection: passrate_min={stats['passrate_min']}  slope_min={stats['slope_min']}  "
        f"slope_margin={stats['slope_margin']}  passrate_margin={stats.get('passrate_margin')}"
    )
    print(f"candidate accuracy={summ.get('accuracy')}  n_candidates={summ.get('n_promoted')}")
    print(f"Wrote -> {stats['output_parquet']}")
    return stats


if __name__ == "__main__":
    main()
