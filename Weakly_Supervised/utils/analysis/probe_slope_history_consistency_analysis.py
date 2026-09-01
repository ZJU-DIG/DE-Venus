"""Combine passrate slope with majority history consistency for probe answer selection analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from math_verify import parse, verify
from scipy.stats import linregress

# from .probe_slope_analysis import iter_flat_answer_records, load_unlabeled_answer_analysis
# from .probes import _answers_equal, _is_unlabeled_record, _round_sort_key, load_probe_summaries

EnrichedRecord = dict[str, Any]
CombinedAnalysisResult = dict[str, Any]
FlatAnswerRecord = dict[str, Any]
SlopeBinAnalysisResult = dict[str, Any]

AnswerStats = list[Any]  # [passrate_trajectory, slope, is_correct]
SampleAnswerAnalysis = dict[str, AnswerStats]
UnlabeledAnalysisResult = dict[int, dict[str, Any]]


def load_probe_summaries(path: str | Path) -> list[dict]:
    """Load JSONL probe summary records."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return records


def _is_unlabeled_record(record: dict) -> bool:
    sample_type = record.get("sample_type")
    if sample_type is not None:
        return str(sample_type).lower() == "unlabeled"
    labeled = record.get("labeled")
    if labeled is None:
        return False
    return labeled is False or labeled == 0 or str(labeled).lower() == "false"


def _answers_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        pa, pb = parse(str(a)), parse(str(b))
        return bool(verify(pa, pb))
    except Exception:
        return str(a).strip() == str(b).strip()


def _passrate_slope(passrates: list[float]) -> float:
    if len(passrates) < 2:
        return 0.0
    y = np.asarray(passrates, dtype=np.float64)
    x = np.arange(len(y))
    slope, *_ = linregress(x, y)
    return float(slope)


def _merge_answer_into_canonical(canonical_answers: list[Any], candidate: Any) -> None:
    for ans in canonical_answers:
        if _answers_equal(ans, candidate):
            return
    canonical_answers.append(candidate)


def _answer_passrate_in_round(rollout_answers: list[Any], answer: Any) -> float:
    if not rollout_answers:
        return 0.0
    hits = sum(1 for r in rollout_answers if _answers_equal(r, answer))
    return hits / len(rollout_answers)


def _round_sort_key(record: dict) -> tuple[int, int]:
    probe_round = int(record.get("probe_round", 0))
    global_step = int(record.get("global_step", 0))
    return (probe_round, global_step)


def _build_answer_stats(
    canonical_answers: list[Any],
    round_records: list[dict],
    ground_truth: Any,
) -> SampleAnswerAnalysis:
    probe_rounds = [int(r.get("probe_round", 0)) for r in round_records]
    answer_stats: SampleAnswerAnalysis = {}

    for answer in canonical_answers:
        passrate_traj = [
            _answer_passrate_in_round(r.get("rollout_answers") or [], answer) for r in round_records
        ]
        answer_stats[str(answer)] = [
            passrate_traj,
            _passrate_slope(passrate_traj),
            _answers_equal(answer, ground_truth),
        ]

    return answer_stats


def analyze_unlabeled_answer_trajectories(
    records: list[dict],
) -> UnlabeledAnalysisResult:
    """Per unlabeled sample index, analyze each canonical answer across probe rounds.

    Returns::
        {
            sample_index: {
                "ground_truth": ...,
                "question": ...,
                "probe_rounds": [int, ...],
                "answers": {
                    answer_str: [passrate_list, slope, is_correct],
                    ...
                },
            },
            ...
        }
    """
    by_index: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if not _is_unlabeled_record(record):
            continue
        if "index" not in record:
            continue
        by_index[int(record["index"])].append(record)

    result: UnlabeledAnalysisResult = {}
    for sample_index, sample_records in sorted(by_index.items()):
        sample_records = sorted(sample_records, key=_round_sort_key)

        canonical_answers: list[Any] = []
        for rec in sample_records:
            for ans in rec.get("rollout_answers") or []:
                _merge_answer_into_canonical(canonical_answers, ans)

        ground_truth = sample_records[-1].get("ground_truth")
        result[sample_index] = {
            "ground_truth": ground_truth,
            "question": sample_records[-1].get("question"),
            "probe_rounds": [int(r.get("probe_round", 0)) for r in sample_records],
            "answers": _build_answer_stats(canonical_answers, sample_records, ground_truth),
        }

    return result


def analyze_probe_summary_file(path: str | Path) -> UnlabeledAnalysisResult:
    """Load ``all_probes_summary.jsonl`` and run unlabeled answer-trajectory analysis."""
    return analyze_unlabeled_answer_trajectories(load_probe_summaries(path))


def save_analysis(result: UnlabeledAnalysisResult, output_path: str | Path) -> None:
    """Write analysis result as JSON (keys stringified for sample indices)."""
    payload = {str(k): v for k, v in result.items()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_unlabeled_answer_analysis(path: str | Path) -> dict[str, Any]:
    """Load ``unlabeled_answer_analysis.json`` (sample index keys are strings)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)



def iter_flat_answer_records(analysis: dict[str, Any]) -> list[FlatAnswerRecord]:
    """Flatten per-sample answer stats into one record per (sample, answer)."""
    rows: list[FlatAnswerRecord] = []
    for sample_key, sample in analysis.items():
        answers = sample.get("answers") or {}
        for answer_str, stats in answers.items():
            if not isinstance(stats, (list, tuple)) or len(stats) < 3:
                continue
            passrate_traj, slope, is_correct = stats[0], stats[1], stats[2]
            final_passrate = float(passrate_traj[-1]) if passrate_traj else 0.0
            rows.append(
                {
                    "sample_index": int(sample_key),
                    "answer": answer_str,
                    "passrate_trajectory": list(passrate_traj),
                    "final_passrate": final_passrate,
                    "slope": float(slope),
                    "is_correct": bool(is_correct),
                }
            )
    return rows


def _mean_final_passrate(rows: list[EnrichedRecord]) -> float | None:
    if not rows:
        return None
    return float(np.mean([r["final_passrate"] for r in rows]))




def majority_history_consistency(history: list[Any], hist_len: int = 8) -> float:
    """Same rule as OLR v2: fraction of recent prior majorities matching the latest one."""
    if len(history) < 2:
        return 1.0
    current = history[-1]
    if hist_len > 0:
        past = history[-1 - hist_len : -1]
    else:
        past = history[:-1]
    if not past:
        return 1.0
    ok = sum(1 for h in past if _answers_equal(h, current))
    return ok / len(past)


def build_sample_majority_meta(
    summary_records: list[dict],
    hist_len: int = 8,
    *,
    sample_type: str | None = "unlabeled",
) -> dict[int, dict[str, Any]]:
    """Per sample index: majority history, latest majority, consistency.

    ``sample_type``: ``labeled``, ``unlabeled``, ``all``, or ``None`` (no filter).
    """
    from .probes import _matches_sample_type

    by_index: dict[int, list[dict]] = defaultdict(list)
    for rec in summary_records:
        if sample_type is not None and not _matches_sample_type(rec, sample_type):  # type: ignore[arg-type]
            continue
        if "index" not in rec:
            continue
        by_index[int(rec["index"])].append(rec)

    meta: dict[int, dict[str, Any]] = {}
    for sample_index, rounds in by_index.items():
        rounds = sorted(rounds, key=_round_sort_key)
        history = [r.get("majority_answer") for r in rounds if r.get("majority_answer") is not None]
        current = history[-1] if history else None
        meta[sample_index] = {
            "majority_history": history,
            "current_majority": current,
            "majority_consistency": majority_history_consistency(history, hist_len),
            "n_probe_rounds": len(rounds),
        }
    return meta


def enrich_answer_records(
    analysis: dict[str, Any],
    sample_meta: dict[int, dict[str, Any]],
) -> list[EnrichedRecord]:
    """Attach sample-level consistency and whether this answer is the latest majority."""
    rows = iter_flat_answer_records(analysis)
    enriched: list[EnrichedRecord] = []
    for row in rows:
        idx = row["sample_index"]
        meta = sample_meta.get(idx, {})
        consistency = float(meta.get("majority_consistency", 1.0))
        current_majority = meta.get("current_majority")
        is_current_majority = (
            current_majority is not None and _answers_equal(row["answer"], current_majority)
        )
        enriched.append(
            {
                **row,
                "majority_consistency": consistency,
                "is_current_majority": is_current_majority,
                "current_majority": current_majority,
                "n_probe_rounds": int(meta.get("n_probe_rounds", 0)),
            }
        )
    return enriched


def _accuracy_metrics(rows: list[EnrichedRecord]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "correct_count": 0, "accuracy": None, "mean_final_passrate": None}
    correct = sum(1 for r in rows if r["is_correct"])
    n = len(rows)
    return {
        "count": n,
        "correct_count": correct,
        "accuracy": correct / n,
        "mean_final_passrate": _mean_final_passrate(rows),
    }


def _f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom <= 0:
        return None
    return 2.0 * precision * recall / denom


def _sample_level_prf(
    picked_rows: list[EnrichedRecord],
    n_samples: int,
) -> dict[str, Any]:
    """Sample-level precision / recall / F1 (denominator = all questions)."""
    correct = sum(1 for r in picked_rows if r["is_correct"])
    n_picked = len(picked_rows)
    precision = correct / n_picked if n_picked else None
    recall = correct / n_samples if n_samples else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1_score(precision, recall),
        "correct_count": correct,
    }


def _majority_row(candidates: list[EnrichedRecord]) -> EnrichedRecord | None:
    for row in candidates:
        if row["is_current_majority"]:
            return row
    return None


def composite_score(
    row: EnrichedRecord,
    *,
    w_slope: float = 1.0,
    w_final_passrate: float = 1.0,
    w_consistency: float = 0.0,
    w_majority: float = 0.5,
) -> float:
    maj_bonus = 1.0 if row["is_current_majority"] else 0.0
    return (
        w_slope * float(row["slope"])
        + w_final_passrate * float(row["final_passrate"])
        + w_consistency * float(row["majority_consistency"])
        + w_majority * maj_bonus
    )


COMPOSITE_WEIGHT_PRESETS: list[dict[str, Any]] = [
    {
        "name": "slope_only",
        "w_slope": 1.0,
        "w_final_passrate": 0.0,
        "w_consistency": 0.0,
        "w_majority": 0.0,
    },
    {
        "name": "passrate_only",
        "w_slope": 0.0,
        "w_final_passrate": 1.0,
        "w_consistency": 0.0,
        "w_majority": 0.0,
    },
    {
        "name": "slope_passrate",
        "w_slope": 1.0,
        "w_final_passrate": 1.0,
        "w_consistency": 0.0,
        "w_majority": 0.0,
    },
    {
        "name": "slope_passrate_maj",
        "w_slope": 1.0,
        "w_final_passrate": 1.0,
        "w_consistency": 0.0,
        "w_majority": 0.5,
    },
]


def _grid_threshold_analysis(
    rows: list[EnrichedRecord],
    slope_mins: list[float],
) -> list[dict[str, Any]]:
    """Accuracy for filters: slope > slope_min (optional majority)."""
    grid: list[dict[str, Any]] = []
    slopes = np.asarray([r["slope"] for r in rows], dtype=np.float64)
    final_passrates = np.asarray([r["final_passrate"] for r in rows], dtype=np.float64)
    correct = np.asarray([1.0 if r["is_correct"] else 0.0 for r in rows], dtype=np.float64)
    is_maj = np.asarray([1.0 if r["is_current_majority"] else 0.0 for r in rows], dtype=np.float64)

    for slope_min in slope_mins:
        for require_majority in (False, True):
            mask = slopes > slope_min
            if require_majority:
                mask = mask & (is_maj > 0.5)
            n = int(mask.sum())
            c = int(correct[mask].sum()) if n else 0
            mean_fp = float(final_passrates[mask].mean()) if n else None
            grid.append(
                {
                    "slope_min_exclusive": float(slope_min),
                    "require_current_majority": require_majority,
                    "count": n,
                    "correct_count": c,
                    "accuracy": c / n if n else None,
                    "mean_final_passrate": mean_fp,
                }
            )
    return grid


def _filter_eligible(
    candidates: list[EnrichedRecord],
    *,
    slope_min: float | None = None,
    passrate_min: float | None = None,
    require_current_majority: bool = False,
) -> list[EnrichedRecord]:
    eligible = candidates
    if slope_min is not None:
        eligible = [r for r in eligible if r["slope"] > slope_min]
    if passrate_min is not None:
        eligible = [r for r in eligible if r["final_passrate"] >= passrate_min]
    if require_current_majority:
        eligible = [r for r in eligible if r["is_current_majority"]]
    return eligible


def _per_sample_selection(
    rows: list[EnrichedRecord],
    *,
    slope_min: float | None = None,
    passrate_min: float | None = None,
    require_current_majority: bool = False,
    score_weights: dict[str, float] | None = None,
    fallback: str = "skip",
) -> dict[str, Any]:
    """Pick one answer per sample; optional composite score and majority fallback."""
    by_sample: dict[int, list[EnrichedRecord]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_index"]].append(row)

    weights = score_weights or {
        "w_slope": 1.0,
        "w_final_passrate": 0.0,
        "w_consistency": 0.0,
        "w_majority": 0.0,
    }
    use_composite = any(weights.get(k, 0.0) != 0.0 for k in ("w_final_passrate", "w_consistency", "w_majority"))

    def pick_score(r: EnrichedRecord) -> float:
        if use_composite:
            return composite_score(r, **weights)
        return float(r["slope"])

    picked_rows: list[EnrichedRecord] = []
    skipped_no_candidate = 0
    fallback_to_majority = 0
    for _, candidates in by_sample.items():
        eligible = _filter_eligible(
            candidates,
            slope_min=slope_min,
            passrate_min=passrate_min,
            require_current_majority=require_current_majority,
        )
        if eligible:
            picked_rows.append(max(eligible, key=pick_score))
            continue
        if fallback == "majority":
            maj = _majority_row(candidates)
            if maj is not None:
                picked_rows.append(maj)
                fallback_to_majority += 1
                continue
        skipped_no_candidate += 1

    n_samples = len(by_sample)
    metrics = _accuracy_metrics(picked_rows)
    metrics.update(_sample_level_prf(picked_rows, n_samples))
    metrics["n_samples"] = n_samples
    metrics["n_samples_with_pick"] = len(picked_rows)
    metrics["n_samples_skipped"] = skipped_no_candidate
    metrics["n_samples_fallback_majority"] = fallback_to_majority
    metrics["sample_coverage"] = len(picked_rows) / n_samples if n_samples else None
    metrics["fallback"] = fallback
    if score_weights:
        metrics["score_weights"] = dict(score_weights)
    return metrics


def _per_sample_two_tier(
    rows: list[EnrichedRecord],
    *,
    strict_slope_min: float,
    strict_passrate_min: float,
    require_current_majority: bool = False,
    score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Strict pool (composite argmax) with majority fallback when pool is empty."""
    by_sample: dict[int, list[EnrichedRecord]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_index"]].append(row)

    weights = score_weights or {
        "w_slope": 1.0,
        "w_final_passrate": 1.0,
        "w_consistency": 0.0,
        "w_majority": 0.5,
    }

    picked_rows: list[EnrichedRecord] = []
    strict_picks = 0
    fallback_picks = 0
    for _, candidates in by_sample.items():
        strict = _filter_eligible(
            candidates,
            slope_min=strict_slope_min,
            passrate_min=strict_passrate_min,
            require_current_majority=require_current_majority,
        )
        if strict:
            picked_rows.append(max(strict, key=lambda r: composite_score(r, **weights)))
            strict_picks += 1
            continue
        maj = _majority_row(candidates)
        if maj is not None:
            picked_rows.append(maj)
            fallback_picks += 1

    n_samples = len(by_sample)
    metrics = _accuracy_metrics(picked_rows)
    metrics.update(_sample_level_prf(picked_rows, n_samples))
    metrics["n_samples"] = n_samples
    metrics["n_samples_with_pick"] = len(picked_rows)
    metrics["n_samples_strict"] = strict_picks
    metrics["n_samples_fallback_majority"] = fallback_picks
    metrics["sample_coverage"] = len(picked_rows) / n_samples if n_samples else None
    metrics["strict_slope_min"] = strict_slope_min
    metrics["strict_passrate_min"] = strict_passrate_min
    metrics["require_current_majority"] = require_current_majority
    metrics["score_weights"] = dict(weights)
    return metrics


def _grid_per_sample_f1(
    rows: list[EnrichedRecord],
    slope_mins: list[float],
    passrate_mins: list[float],
) -> list[dict[str, Any]]:
    """Grid per-sample strategies; rank by sample-level F1 (precision/recall on all questions)."""
    grid: list[dict[str, Any]] = []

    for slope_min in slope_mins:
        for passrate_min in passrate_mins:
            for require_majority in (False, True):
                for fallback in ("skip", "majority"):
                    for preset in COMPOSITE_WEIGHT_PRESETS:
                        metrics = _per_sample_selection(
                            rows,
                            slope_min=slope_min if slope_min > 0 else None,
                            passrate_min=passrate_min if passrate_min > 0 else None,
                            require_current_majority=require_majority,
                            score_weights={
                                "w_slope": preset["w_slope"],
                                "w_final_passrate": preset["w_final_passrate"],
                                "w_consistency": preset["w_consistency"],
                                "w_majority": preset["w_majority"],
                            },
                            fallback=fallback,
                        )
                        grid.append(
                            {
                                "strategy": "filtered_composite",
                                "slope_min_exclusive": float(slope_min),
                                "passrate_min_inclusive": float(passrate_min),
                                "require_current_majority": require_majority,
                                "fallback": fallback,
                                "weight_preset": preset["name"],
                                **metrics,
                            }
                        )

    for slope_min in slope_mins:
        for passrate_min in passrate_mins:
            for require_majority in (False, True):
                metrics = _per_sample_two_tier(
                    rows,
                    strict_slope_min=slope_min,
                    strict_passrate_min=passrate_min,
                    require_current_majority=require_majority,
                )
                grid.append(
                    {
                        "strategy": "two_tier_strict_then_majority",
                        "slope_min_exclusive": float(slope_min),
                        "passrate_min_inclusive": float(passrate_min),
                        "require_current_majority": require_majority,
                        **metrics,
                    }
                )

    return grid


def analyze_slope_and_consistency(
    analysis: dict[str, Any],
    summary_records: list[dict],
    *,
    hist_len: int = 8,
    slope_mins: list[float] | None = None,
    passrate_mins: list[float] | None = None,
    include_f1_grid: bool = True,
) -> CombinedAnalysisResult:
    sample_meta = build_sample_majority_meta(summary_records, hist_len=hist_len)
    rows = enrich_answer_records(analysis, sample_meta)

    if slope_mins is None:
        slope_mins = [0.0, 0.02, 0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25]
    if passrate_mins is None:
        passrate_mins = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    f1_slope_mins = [0.0, 0.05, 0.08, 0.10, 0.15]
    f1_passrate_mins = [0.0, 0.15, 0.25, 0.35, 0.50]

    slopes = np.asarray([r["slope"] for r in rows], dtype=np.float64)
    cons = np.asarray([r["majority_consistency"] for r in rows], dtype=np.float64)
    final_passrates = np.asarray([r["final_passrate"] for r in rows], dtype=np.float64)
    correct = np.asarray([1.0 if r["is_correct"] else 0.0 for r in rows], dtype=np.float64)

    overall: dict[str, Any] = {
        "n_answer_records": len(rows),
        "n_samples": len(sample_meta),
        **_accuracy_metrics(rows),
    }
    if rows:
        correct_mask = correct > 0.5
        overall["mean_final_passrate_correct"] = (
            float(final_passrates[correct_mask].mean()) if correct_mask.any() else None
        )
        overall["mean_final_passrate_incorrect"] = (
            float(final_passrates[~correct_mask].mean()) if (~correct_mask).any() else None
        )
    else:
        overall["mean_final_passrate_correct"] = None
        overall["mean_final_passrate_incorrect"] = None
    if rows and np.std(slopes) > 0 and np.std(correct) > 0:
        overall["pearson_slope_correct"] = float(np.corrcoef(slopes, correct)[0, 1])
    else:
        overall["pearson_slope_correct"] = None
    if rows and np.std(cons) > 0 and np.std(correct) > 0:
        overall["pearson_consistency_correct"] = float(np.corrcoef(cons, correct)[0, 1])
    else:
        overall["pearson_consistency_correct"] = None

    result: CombinedAnalysisResult = {
        "hist_len": hist_len,
        "overall": overall,
        "answer_level": {
            "all_answers": _accuracy_metrics(rows),
            "current_majority_only": _accuracy_metrics([r for r in rows if r["is_current_majority"]]),
            "slope_positive": _accuracy_metrics([r for r in rows if r["slope"] > 0]),
            "slope_gt_0.05": _accuracy_metrics([r for r in rows if r["slope"] > 0.05]),
            "slope_gt_0.05_majority": _accuracy_metrics(
                [r for r in rows if r["slope"] > 0.05 and r["is_current_majority"]]
            ),
        },
        "threshold_grid": _grid_threshold_analysis(rows, slope_mins),
        "per_sample_selection": {
            "argmax_slope": _per_sample_selection(rows),
            "argmax_slope_majority_only": _per_sample_selection(
                rows, require_current_majority=True
            ),
            "argmax_slope_slope_gt_0.05": _per_sample_selection(rows, slope_min=0.05),
            "argmax_slope_slope_gt_0.05_majority": _per_sample_selection(
                rows,
                slope_min=0.05,
                require_current_majority=True,
            ),
            "argmax_slope_slope_gt_0.10_majority": _per_sample_selection(
                rows,
                slope_min=0.10,
                require_current_majority=True,
            ),
            "composite_slope_passrate_fallback_majority": _per_sample_selection(
                rows,
                slope_min=0.05,
                passrate_min=0.15,
                score_weights={
                    "w_slope": 1.0,
                    "w_final_passrate": 1.0,
                    "w_consistency": 0.0,
                    "w_majority": 0.5,
                },
                fallback="majority",
            ),
            "two_tier_slope0.05_pr0.15_fallback_majority": _per_sample_two_tier(
                rows,
                strict_slope_min=0.05,
                strict_passrate_min=0.15,
            ),
            "two_tier_slope0.08_pr0.25_maj_fallback": _per_sample_two_tier(
                rows,
                strict_slope_min=0.08,
                strict_passrate_min=0.25,
                require_current_majority=True,
            ),
        },
    }
    if include_f1_grid:
        f1_grid = _grid_per_sample_f1(
            rows,
            f1_slope_mins,
            f1_passrate_mins,
        )
        f1_ranked = sorted(
            f1_grid,
            key=lambda x: (-(x.get("f1") or -1), -(x.get("precision") or -1)),
        )
        result["per_sample_f1_grid"] = f1_grid
        result["per_sample_f1_best"] = [r for r in f1_ranked if r.get("f1") is not None][:20]
    return result


def analyze_from_files(
    analysis_path: str | Path,
    summary_path: str | Path,
    **kwargs: Any,
) -> CombinedAnalysisResult:
    return analyze_slope_and_consistency(
        load_unlabeled_answer_analysis(analysis_path),
        load_probe_summaries(summary_path),
        **kwargs,
    )


def save_combined_analysis(result: CombinedAnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _format_prf(metrics: dict[str, Any]) -> str:
    p, r, f1 = metrics.get("precision"), metrics.get("recall"), metrics.get("f1")
    p_s = f"{p:.4f}" if p is not None else "n/a"
    r_s = f"{r:.4f}" if r is not None else "n/a"
    f1_s = f"{f1:.4f}" if f1 is not None else "n/a"
    return f"P={p_s}  R={r_s}  F1={f1_s}"


def print_combined_summary(result: CombinedAnalysisResult) -> None:
    overall = result.get("overall") or {}
    print("=== Slope + majority consistency analysis ===")
    mfp_c = overall.get("mean_final_passrate_correct")
    mfp_w = overall.get("mean_final_passrate_incorrect")
    mfp_c_s = f"{mfp_c:.4f}" if mfp_c is not None else "n/a"
    mfp_w_s = f"{mfp_w:.4f}" if mfp_w is not None else "n/a"
    print(
        f"n_samples={overall.get('n_samples')}  n_answer_records={overall.get('n_answer_records')}  "
        f"answer_accuracy={overall.get('accuracy'):.4f}  "
        f"pearson_slope={overall.get('pearson_slope_correct')}  "
        f"pearson_consistency={overall.get('pearson_consistency_correct')}"
    )
    print(f"mean_final_passrate correct={mfp_c_s}  incorrect={mfp_w_s}")

    print("\n--- Answer-level baselines ---")
    for name, metrics in (result.get("answer_level") or {}).items():
        acc = metrics.get("accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        mfp = metrics.get("mean_final_passrate")
        mfp_s = f"{mfp:.4f}" if mfp is not None else "n/a"
        print(f"  {name}: count={metrics.get('count')}  accuracy={acc_s}  mean_pr={mfp_s}")

    print("\n--- Per-sample selection (one pick per question) ---")
    for name, metrics in (result.get("per_sample_selection") or {}).items():
        acc = metrics.get("accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        cov = metrics.get("sample_coverage")
        cov_s = f"{cov:.4f}" if cov is not None else "n/a"
        mfp = metrics.get("mean_final_passrate")
        mfp_s = f"{mfp:.4f}" if mfp is not None else "n/a"
        fb = metrics.get("n_samples_fallback_majority")
        fb_s = f"  fallback_maj={fb}" if fb else ""
        print(
            f"  {name}: picked={metrics.get('n_samples_with_pick')}/"
            f"{metrics.get('n_samples')}  coverage={cov_s}  acc={acc_s}  mean_pr={mfp_s}  "
            f"{_format_prf(metrics)}{fb_s}"
        )

    best_f1 = result.get("per_sample_f1_best") or []
    if best_f1:
        print("\n--- Best per-sample configs by F1 (top 10) ---")
        for i, row in enumerate(best_f1[:10], start=1):
            strat = row.get("strategy", "?")
            if strat == "two_tier_strict_then_majority":
                cfg = (
                    f"two_tier slope>{row['strict_slope_min']:.3f}  "
                    f"pr>={row['strict_passrate_min']:.2f}  "
                    f"maj={row.get('require_current_majority')}"
                )
            else:
                cfg = (
                    f"slope>{row['slope_min_exclusive']:.3f}  "
                    f"pr>={row['passrate_min_inclusive']:.2f}  "
                    f"maj={row.get('require_current_majority')}  "
                    f"fb={row.get('fallback')}  w={row.get('weight_preset')}"
                )
            acc = row.get("accuracy")
            acc_s = f"{acc:.4f}" if acc is not None else "n/a"
            mfp = row.get("mean_final_passrate")
            mfp_s = f"{mfp:.4f}" if mfp is not None else "n/a"
            print(f"  {i:>2}. {cfg}")
            print(
                f"      cov={row.get('sample_coverage', 0):.4f}  acc={acc_s}  mean_pr={mfp_s}  "
                f"{_format_prf(row)}"
            )

    print("\n--- Answer-level threshold grid (top 8 by accuracy, count>=50) ---")
    grid = sorted(
        result.get("threshold_grid") or [],
        key=lambda x: (-(x["accuracy"] or -1), -x["count"]),
    )
    shown = 0
    for row in grid:
        if row["count"] < 50:
            continue
        acc = row["accuracy"]
        mfp = row.get("mean_final_passrate")
        mfp_s = f"{mfp:.4f}" if mfp is not None else "n/a"
        maj = "maj" if row["require_current_majority"] else "any"
        print(
            f"  slope>{row['slope_min_exclusive']:.3f}  {maj}  "
            f"n={row['count']}  acc={acc:.4f}  mean_pr={mfp_s}"
        )
        shown += 1
        if shown >= 8:
            break


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze slope + majority history consistency for probe answers.",
    )
    parser.add_argument(
        "analysis_path",
        type=Path,
        help="unlabeled_answer_analysis.json",
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        nargs="?",
        default=None,
        help="all_probes_summary.jsonl (default: sibling of analysis file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <dir>/slope_consistency_analysis.json)",
    )
    parser.add_argument(
        "--hist-len",
        type=int,
        default=8,
        help="History length for majority consistency (OLR default 8)",
    )
    parser.add_argument(
        "--no-f1-grid",
        action="store_true",
        help="Skip per-sample F1 grid search (faster)",
    )
    return parser


def main(argv: list[str] | None = None) -> CombinedAnalysisResult:
    args = _build_argparser().parse_args(argv)
    summary_path = args.summary_path or (args.analysis_path.parent / "all_probes_summary.jsonl")
    result = analyze_from_files(
        args.analysis_path,
        summary_path,
        hist_len=args.hist_len,
        include_f1_grid=not args.no_f1_grid,
    )
    output = args.output or (args.analysis_path.parent / "slope_consistency_analysis.json")
    save_combined_analysis(result, output)
    print_combined_summary(result)
    print(f"Wrote -> {output}")
    return result


if __name__ == "__main__":
    main()
