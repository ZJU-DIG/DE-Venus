"""Build / advance training parquet for iterative TraPO (labeled + newly promoted + remaining unlabeled)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .probes import analyze_probe_summary_file, save_analysis
from .stage2_selection import (
    promotion_summary,
    save_promoted_manifest,
    select_promoted_unlabeled_samples,
)


def _is_labeled_extra_info(extra_info: Any) -> bool:
    if extra_info is None:
        return False
    if isinstance(extra_info, dict):
        labeled = extra_info.get("labeled", False)
        return labeled is True or labeled == 1 or str(labeled).lower() == "true"
    return False


def _is_labeled_row(row: dict[str, Any], row_pos: int) -> bool:
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict):
        return _is_labeled_extra_info(extra)
    return False


def _dataset_index_from_row(row: dict[str, Any], row_pos: int) -> int:
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict) and extra.get("index") is not None:
        return int(extra["index"])
    return row_pos


def count_labeled_unlabeled_rows(parquet_path: str | Path) -> dict[str, int]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(parquet_path))["train"]
    n_labeled = 0
    n_unlabeled = 0
    for row_pos in range(len(ds)):
        row = dict(ds[row_pos])
        if _is_labeled_row(row, row_pos):
            n_labeled += 1
        else:
            n_unlabeled += 1
    return {
        "n_total": len(ds),
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
    }


def advance_training_parquet(
    source_parquet: str | Path,
    output_parquet: str | Path,
    promoted: list[dict[str, Any]],
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    keep_remaining_unlabeled: bool = True,
    promotion_iteration: int | None = None,
) -> dict[str, Any]:
    """Promote selected unlabeled rows; optionally keep all other rows unchanged.

    When ``keep_remaining_unlabeled`` is True (iterative TraPO default), output has:
    gold labeled + previously promoted + newly promoted + still-unlabeled rows.
    """
    from datasets import Dataset, load_dataset

    source_parquet = Path(source_parquet)
    output_parquet = Path(output_parquet)
    promoted_by_index = {int(p["sample_index"]): p for p in promoted}

    ds = load_dataset("parquet", data_files=str(source_parquet))["train"]

    out_rows: list[dict[str, Any]] = []
    n_original_labeled = 0
    n_already_labeled = 0
    n_promoted = 0
    n_remaining_unlabeled = 0
    n_promote_skipped_not_in_pool = 0

    for row_pos in range(len(ds)):
        row = dict(ds[row_pos])
        idx = _dataset_index_from_row(row, row_pos)

        if _is_labeled_row(row, row_pos):
            out_rows.append(row)
            extra = row.get("extra_info") or {}
            if isinstance(extra, dict) and extra.get("promoted_from_unlabeled"):
                n_already_labeled += 1
            else:
                n_original_labeled += 1
            continue

        if idx in promoted_by_index:
            p = promoted_by_index[idx]
            new_row = copy.deepcopy(row)
            extra: dict[str, Any] = dict(new_row.get("extra_info") or {})
            rm: dict[str, Any] = dict(new_row.get("reward_model") or {})

            original_gt = rm.get("ground_truth")
            extra["original_gt"] = original_gt
            extra["original_labeled"] = False
            extra["labeled"] = True
            extra["promoted_from_unlabeled"] = True
            extra["promotion_passrate_min"] = float(passrate_min)
            if slope_min is not None:
                extra["promotion_slope_min"] = float(slope_min)
            if promotion_iteration is not None:
                extra["promotion_iteration"] = int(promotion_iteration)
            extra["pseudo_final_passrate"] = float(p["final_passrate"])
            extra["pseudo_first_passrate"] = float(p["first_passrate"])
            extra["pseudo_slope"] = float(p["slope"])
            extra["dataset_index"] = idx

            rm["ground_truth"] = p["pseudo_gt"]
            rm["original_gt"] = original_gt

            new_row["extra_info"] = extra
            new_row["reward_model"] = rm
            out_rows.append(new_row)
            n_promoted += 1
            continue

        if keep_remaining_unlabeled:
            out_rows.append(row)
            n_remaining_unlabeled += 1
        else:
            n_promote_skipped_not_in_pool += 1

    out_ds = Dataset.from_list(out_rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_parquet(str(output_parquet))

    counts = count_labeled_unlabeled_rows(output_parquet)
    return {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "n_source_rows": len(ds),
        "n_output_rows": len(out_rows),
        "n_original_labeled": n_original_labeled,
        "n_already_promoted_labeled": n_already_labeled,
        "n_newly_promoted": n_promoted,
        "n_remaining_unlabeled": n_remaining_unlabeled,
        "n_dropped_unlabeled": n_promote_skipped_not_in_pool,
        "keep_remaining_unlabeled": keep_remaining_unlabeled,
        "promotion_summary": promotion_summary(promoted),
        **counts,
    }


def advance_training_parquet_from_probe_dir(
    source_parquet: str | Path,
    probe_dir: str | Path,
    output_parquet: str | Path,
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    hist_len: int = 8,
    keep_remaining_unlabeled: bool = True,
    promotion_iteration: int | None = None,
    manifest_path: str | Path | None = None,
    rebuild_analysis: bool = False,
) -> dict[str, Any]:
    probe_dir = Path(probe_dir)
    summary_path = probe_dir / "all_probes_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing probe summary: {summary_path}")

    analysis_path = probe_dir / "unlabeled_answer_analysis.json"
    if rebuild_analysis or not analysis_path.is_file():
        analysis = analyze_probe_summary_file(summary_path, sample_type="unlabeled")
        save_analysis(analysis, analysis_path)
    else:
        from .probes import load_unlabeled_answer_analysis

        analysis = load_unlabeled_answer_analysis(analysis_path)

    from .probes import load_probe_summaries

    promoted = select_promoted_unlabeled_samples(
        analysis,
        load_probe_summaries(summary_path),
        passrate_min=passrate_min,
        slope_min=slope_min,
        hist_len=hist_len,
    )

    manifest = manifest_path or probe_dir / "promoted_manifest.json"
    save_promoted_manifest(
        promoted,
        manifest,
        passrate_min=passrate_min,
        slope_min=slope_min,
        hist_len=hist_len,
        source_parquet=source_parquet,
    )

    stats = advance_training_parquet(
        source_parquet,
        output_parquet,
        promoted,
        passrate_min=passrate_min,
        slope_min=slope_min,
        keep_remaining_unlabeled=keep_remaining_unlabeled,
        promotion_iteration=promotion_iteration,
    )
    stats["manifest_path"] = str(manifest)
    stats["passrate_min"] = passrate_min
    stats["slope_min"] = slope_min
    stats["hist_len"] = hist_len
    if promotion_iteration is not None:
        stats["promotion_iteration"] = promotion_iteration

    meta_path = Path(output_parquet).with_suffix(".advance_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    stats["meta_path"] = str(meta_path)
    return stats


# Backward-compatible aliases
build_stage2_parquet = advance_training_parquet
build_stage2_from_probe_dir = advance_training_parquet_from_probe_dir


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advance training parquet after a probe round (promote pr>=threshold).",
    )
    parser.add_argument("source_parquet", type=Path, help="Current training parquet")
    parser.add_argument("probe_dir", type=Path, help="Probe dump dir with all_probes_summary.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output parquet (default: <probe_dir>/train_next.parquet)",
    )
    parser.add_argument("--passrate-min", type=float, default=0.35)
    parser.add_argument(
        "--slope-min",
        type=float,
        default=0.05,
        help="Promote only if majority slope > this value (strict). Use a negative value to disable.",
    )
    parser.add_argument("--hist-len", type=int, default=8)
    parser.add_argument("--promotion-iteration", type=int, default=None)
    parser.add_argument(
        "--drop-remaining-unlabeled",
        action="store_true",
        help="Legacy stage-2: only labeled + promoted (drop other unlabeled)",
    )
    parser.add_argument("--rebuild-analysis", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)
    output = args.output or (args.probe_dir / "train_next.parquet")
    slope_min = None if args.slope_min < 0 else args.slope_min
    stats = advance_training_parquet_from_probe_dir(
        args.source_parquet,
        args.probe_dir,
        output,
        passrate_min=args.passrate_min,
        slope_min=slope_min,
        hist_len=args.hist_len,
        keep_remaining_unlabeled=not args.drop_remaining_unlabeled,
        promotion_iteration=args.promotion_iteration,
        rebuild_analysis=args.rebuild_analysis,
    )
    summ = stats["promotion_summary"]
    print("=== Advance training parquet ===")
    print(
        f"rows: {stats['n_source_rows']} -> {stats['n_output_rows']}  "
        f"labeled={stats['n_labeled']}  unlabeled={stats['n_unlabeled']}"
    )
    print(
        f"gold_labeled={stats['n_original_labeled']}  "
        f"prior_promoted={stats['n_already_promoted_labeled']}  "
        f"newly_promoted={stats['n_newly_promoted']}  "
        f"remaining_unlabeled={stats['n_remaining_unlabeled']}"
    )
    print(
        f"new_promoted accuracy={summ.get('accuracy')}  "
        f"n={summ.get('n_promoted')}  mean_final_pr={summ.get('mean_final_passrate')}"
    )
    print(f"Wrote parquet -> {stats['output_parquet']}")
    print(f"Wrote manifest -> {stats['manifest_path']}")
    return stats


if __name__ == "__main__":
    main()
