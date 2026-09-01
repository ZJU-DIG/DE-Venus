"""Select unlabeled samples to promote for stage-2 training (pseudo-labeled as labeled)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .probe_slope_analysis import load_unlabeled_answer_analysis
from .probe_slope_history_consistency_analysis import (
    build_sample_majority_meta,
    enrich_answer_records,
)
from .probes import _answers_equal, load_probe_summaries

PromotedSample = dict[str, Any]


def _lookup_answer_passrate_stats(sample_analysis: dict[str, Any], answer: Any) -> dict[str, Any]:
    """Passrate trajectory stats for ``answer``; missing from rollouts -> slope/final/first = 0."""
    default = {
        "first_passrate": 0.0,
        "final_passrate": 0.0,
        "slope": 0.0,
        "passrate_trajectory": [],
        "in_rollouts": False,
    }
    if answer is None:
        return default

    answers = sample_analysis.get("answers") or {}
    for ans_key, stats in answers.items():
        if not isinstance(stats, (list, tuple)) or len(stats) < 2:
            continue
        if not _answers_equal(ans_key, answer):
            continue
        traj = list(stats[0]) if stats[0] is not None else []
        slope = float(stats[1])
        first_pr = float(traj[0]) if traj else 0.0
        final_pr = float(traj[-1]) if traj else 0.0
        return {
            "first_passrate": first_pr,
            "final_passrate": final_pr,
            "slope": slope,
            "passrate_trajectory": traj,
            "in_rollouts": True,
        }
    return default


def _dataset_index_from_row(row: dict[str, Any], row_pos: int) -> int:
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict) and extra.get("index") is not None:
        return int(extra["index"])
    return row_pos


def select_relabel_noisy_label_samples(
    analysis: dict[str, Any],
    summary_records: list[dict],
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    slope_margin: float = 0.05,
    passrate_margin: float | None = None,
    hist_len: int = 8,
    require_current_majority: bool = True,
    labels_by_index: dict[int, dict[str, Any]] | None = None,
    source_parquet: str | Path | None = None,
) -> list[PromotedSample]:
    """Select probe majority relabels vs current parquet ``reward_model.ground_truth``.

    Gates on the **majority** answer:
    - ``final_passrate >= passrate_min``
    - if ``slope_min`` is set: ``slope > slope_min``

    Comparative gate vs **current label** (``reward_model.ground_truth`` in ``source_parquet``,
    not ``extra_info.original_ground_truth`` nor probe analysis ``ground_truth``):
    - Look up that answer in per-answer analysis; if it never appeared in any rollout,
      treat ``slope=0``, ``final_passrate=0``, ``first_passrate=0``.
    - Require ``slope_majority > slope_initial + slope_margin`` (strict).
    - If ``passrate_margin`` is set: ``final_majority >= final_initial + passrate_margin``.

    Requires ``labels_by_index`` or ``source_parquet`` so the baseline label comes from parquet.
    """
    if labels_by_index is None and source_parquet is not None:
        labels_by_index = load_parquet_labels_by_index(source_parquet)
    if labels_by_index is None:
        raise ValueError(
            "select_relabel_noisy_label_samples requires labels_by_index or source_parquet "
            "(reward_model.ground_truth baseline)"
        )
    sample_meta = build_sample_majority_meta(
        summary_records, hist_len=hist_len, sample_type="all"
    )
    rows = enrich_answer_records(analysis, sample_meta)
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_index"]), []).append(row)

    selected: list[PromotedSample] = []
    for sample_index, candidates in sorted(by_sample.items()):
        sample_block = analysis.get(str(sample_index)) or analysis.get(sample_index) or {}
        label_row = labels_by_index.get(int(sample_index), {})
        current_gt = label_row.get("current_gt_before_relabel")
        probe_analysis_gt = sample_block.get("ground_truth")
        init_stats = _lookup_answer_passrate_stats(sample_block, current_gt)

        if require_current_majority:
            maj_rows = [r for r in candidates if r.get("is_current_majority")]
        else:
            maj_rows = candidates
        if not maj_rows:
            continue
        if len(maj_rows) > 1:
            maj_rows = [max(maj_rows, key=lambda r: float(r["slope"]))]
        row = maj_rows[0]
        pseudo_gt = row["answer"]

        if current_gt is not None and _answers_equal(pseudo_gt, current_gt):
            continue

        final_pr = float(row["final_passrate"])
        slope = float(row["slope"])
        if final_pr < passrate_min:
            continue
        if slope_min is not None and slope <= slope_min:
            continue

        init_slope = float(init_stats["slope"])
        init_final = float(init_stats["final_passrate"])
        if slope <= init_slope + float(slope_margin):
            continue
        if passrate_margin is not None and final_pr < init_final + float(passrate_margin):
            continue

        traj = row.get("passrate_trajectory") or []
        first_pr = float(traj[0]) if traj else 0.0
        selected.append(
            {
                "sample_index": sample_index,
                "pseudo_gt": pseudo_gt,
                "initial_gt": current_gt,
                "current_gt_before_relabel": current_gt,
                "probe_analysis_ground_truth": probe_analysis_gt,
                "final_passrate": final_pr,
                "first_passrate": first_pr,
                "slope": slope,
                "initial_final_passrate": init_final,
                "initial_first_passrate": float(init_stats["first_passrate"]),
                "initial_slope": init_slope,
                "initial_gt_in_rollouts": bool(init_stats["in_rollouts"]),
                "slope_delta_vs_initial": slope - init_slope,
                "final_passrate_delta_vs_initial": final_pr - init_final,
                "is_correct": bool(row.get("is_correct")),
                "current_majority": row.get("current_majority"),
            }
        )
    return selected


def select_promoted_unlabeled_samples(
    analysis: dict[str, Any],
    summary_records: list[dict],
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    hist_len: int = 8,
    require_current_majority: bool = True,
) -> list[PromotedSample]:
    """Per unlabeled sample: promote current majority when passrate and slope pass.

    Criteria (on the current-majority answer row):
    - ``final_passrate >= passrate_min`` (inclusive)
    - if ``slope_min`` is not None: ``slope > slope_min`` (strict)

    Returns one dict per promoted sample with ``sample_index``, ``pseudo_gt``,
    ``final_passrate``, ``slope``, ``first_passrate``, and ``is_correct`` (eval only).
    """
    sample_meta = build_sample_majority_meta(
        summary_records, hist_len=hist_len, sample_type="unlabeled"
    )
    rows = enrich_answer_records(analysis, sample_meta)

    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_index"]), []).append(row)

    promoted: list[PromotedSample] = []
    for sample_index, candidates in sorted(by_sample.items()):
        if require_current_majority:
            maj_rows = [r for r in candidates if r.get("is_current_majority")]
        else:
            maj_rows = candidates
        if not maj_rows:
            continue
        if len(maj_rows) > 1:
            maj_rows = [max(maj_rows, key=lambda r: float(r["slope"]))]
        row = maj_rows[0]
        final_pr = float(row["final_passrate"])
        slope = float(row["slope"])
        if final_pr < passrate_min:
            continue
        if slope_min is not None and slope <= slope_min:
            continue
        traj = row.get("passrate_trajectory") or []
        first_pr = float(traj[0]) if traj else 0.0
        promoted.append(
            {
                "sample_index": sample_index,
                "pseudo_gt": row["answer"],
                "final_passrate": final_pr,
                "first_passrate": first_pr,
                "slope": slope,
                "is_correct": bool(row["is_correct"]),
                "current_majority": row.get("current_majority"),
            }
        )
    return promoted


def select_promoted_from_files(
    analysis_path: str | Path,
    summary_path: str | Path,
    *,
    passrate_min: float = 0.35,
    slope_min: float | None = 0.05,
    hist_len: int = 8,
) -> list[PromotedSample]:
    analysis = load_unlabeled_answer_analysis(analysis_path)
    summary_records = load_probe_summaries(summary_path)
    return select_promoted_unlabeled_samples(
        analysis,
        summary_records,
        passrate_min=passrate_min,
        slope_min=slope_min,
        hist_len=hist_len,
    )


def promotion_summary(promoted: list[PromotedSample]) -> dict[str, Any]:
    if not promoted:
        return {
            "n_promoted": 0,
            "n_correct": 0,
            "accuracy": None,
            "mean_final_passrate": None,
            "mean_first_passrate": None,
            "mean_slope": None,
        }
    n_correct = sum(1 for p in promoted if p["is_correct"])
    n = len(promoted)
    return {
        "n_promoted": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "mean_final_passrate": sum(p["final_passrate"] for p in promoted) / n,
        "mean_first_passrate": sum(p["first_passrate"] for p in promoted) / n,
        "mean_slope": sum(p["slope"] for p in promoted) / n,
    }


def load_parquet_labels_by_index(source_parquet: str | Path) -> dict[int, dict[str, Any]]:
    """Load per-row label fields from training parquet for relabel manifest analysis."""
    from datasets import load_dataset

    source_parquet = Path(source_parquet)
    ds = load_dataset("parquet", data_files=str(source_parquet))["train"]
    by_index: dict[int, dict[str, Any]] = {}
    for row_pos in range(len(ds)):
        row = dict(ds[row_pos])
        extra = row.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}
        rm = row.get("reward_model") or {}
        if not isinstance(rm, dict):
            rm = {}
        idx = _dataset_index_from_row(row, row_pos)
        current_gt = rm.get("ground_truth")
        initial_gt = (
            extra.get("initial_ground_truth")
            or extra.get("original_ground_truth")
            or rm.get("original_gt")
        )
        by_index[idx] = {
            "current_gt_before_relabel": current_gt,
            "initial_ground_truth": initial_gt,
            "original_ground_truth": extra.get("original_ground_truth") or rm.get("original_gt"),
            "real_gt": rm.get("real_gt"),
            "relabel_iteration_in_parquet": extra.get("relabel_iteration"),
            "promoted_from_unlabeled": extra.get("promoted_from_unlabeled"),
        }
    return by_index


def enrich_relabel_candidates_from_parquet(
    candidates: list[PromotedSample],
    source_parquet: str | Path,
) -> list[PromotedSample]:
    """Attach parquet original/current labels to each relabel candidate for manifest export."""
    labels_by_index = load_parquet_labels_by_index(source_parquet)
    enriched: list[PromotedSample] = []
    for cand in candidates:
        idx = int(cand["sample_index"])
        labels = labels_by_index.get(idx, {})
        current_gt = labels.get("current_gt_before_relabel")
        initial_gt = labels.get("initial_ground_truth")
        probe_ref_gt = cand.get("probe_analysis_ground_truth")
        pseudo_gt = cand.get("pseudo_gt")
        enriched.append(
            {
                **cand,
                "original_gt": current_gt,
                "current_gt_before_relabel": current_gt,
                "initial_ground_truth": initial_gt,
                "original_ground_truth": labels.get("original_ground_truth"),
                "real_gt": labels.get("real_gt"),
                "probe_reference_gt": probe_ref_gt,
                "relabel_to": pseudo_gt,
                "pseudo_gt": pseudo_gt,
            }
        )
    return enriched


def save_relabel_manifest(
    relabels: list[PromotedSample],
    path: str | Path,
    *,
    passrate_min: float,
    slope_min: float | None,
    hist_len: int,
    source_parquet: str | Path | None = None,
    slope_margin: float | None = None,
    passrate_margin: float | None = None,
    extra_criteria: dict[str, Any] | None = None,
) -> None:
    """Write ``relabel_manifest.json`` with original labels and probe/relabel targets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "passrate_min_inclusive": passrate_min,
        "slope_min_exclusive": slope_min,
        "hist_len": hist_len,
        "source_parquet": str(source_parquet) if source_parquet else None,
        "summary": promotion_summary(relabels),
        "relabels": relabels,
        "promoted": relabels,
    }
    if slope_margin is not None:
        payload["slope_margin_exclusive"] = slope_margin
    if passrate_margin is not None:
        payload["passrate_margin_inclusive"] = passrate_margin
    if extra_criteria:
        payload.update(extra_criteria)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_promoted_manifest(
    promoted: list[PromotedSample],
    path: str | Path,
    *,
    passrate_min: float,
    slope_min: float | None,
    hist_len: int,
    source_parquet: str | Path | None = None,
    extra_criteria: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passrate_min_inclusive": passrate_min,
        "slope_min_exclusive": slope_min,
        "hist_len": hist_len,
        "source_parquet": str(source_parquet) if source_parquet else None,
        "summary": promotion_summary(promoted),
        "promoted": promoted,
    }
    if extra_criteria:
        payload.update(extra_criteria)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
