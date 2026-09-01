"""Slope vs correctness among answers equal to the final-round majority only.

Requires ``unlabeled_answer_analysis.json`` and ``all_probes_summary.jsonl`` so
``is_current_majority`` is defined (latest probe round majority per sample).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .probe_slope_analysis import (
    _build_threshold_min_bins,
    _default_threshold_slope_mins,
    _make_slope_bin_edges,
    _mean_final_passrate_in_mask,
    load_unlabeled_answer_analysis,
)
from .probe_slope_history_consistency_analysis import (
    build_sample_majority_meta,
    enrich_answer_records,
)
from .probes import load_probe_summaries

FlatAnswerRecord = dict[str, Any]
SlopeFinalMajAnalysisResult = dict[str, Any]


def build_final_majority_answer_rows(
    analysis: dict[str, Any],
    summary_records: list[dict],
    *,
    hist_len: int = 8,
    sample_type: str = "unlabeled",
) -> tuple[list[FlatAnswerRecord], list[FlatAnswerRecord]]:
    """Return (all enriched rows, subset where answer == final-round majority)."""
    sample_meta = build_sample_majority_meta(
        summary_records, hist_len=hist_len, sample_type=sample_type
    )
    all_rows = enrich_answer_records(analysis, sample_meta)
    maj_rows = [r for r in all_rows if r.get("is_current_majority")]
    return all_rows, maj_rows


def _bin_slopes(
    rows: list[FlatAnswerRecord],
    *,
    n_bins: int,
    bin_edges: np.ndarray | None,
    threshold_slope_mins: list[float] | None,
    threshold_slope_max: float,
    include_threshold_bins: bool,
) -> dict[str, Any]:
    slopes = np.asarray([r["slope"] for r in rows], dtype=np.float64)
    final_passrates = np.asarray([r["final_passrate"] for r in rows], dtype=np.float64)
    correct = np.asarray([1.0 if r["is_correct"] else 0.0 for r in rows], dtype=np.float64)

    if bin_edges is None:
        edges = _make_slope_bin_edges(slopes, n_bins)
    else:
        edges = np.asarray(bin_edges, dtype=np.float64)
        n_bins = len(edges) - 1

    bin_rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        if not rows:
            mask = np.zeros(0, dtype=bool)
        elif b < n_bins - 1:
            mask = (slopes >= edges[b]) & (slopes < edges[b + 1])
        else:
            mask = (slopes >= edges[b]) & (slopes <= edges[b + 1])
        count = int(mask.sum())
        correct_count = int(correct[mask].sum()) if count else 0
        bin_rows.append(
            {
                "bin_id": b,
                "bin_type": "interval",
                "slope_min": float(edges[b]),
                "slope_max": float(edges[b + 1]),
                "slope_min_inclusive": True,
                "slope_max_inclusive": b == n_bins - 1,
                "count": count,
                "correct_count": correct_count,
                "incorrect_count": count - correct_count,
                "accuracy": correct_count / count if count else None,
                "mean_final_passrate": _mean_final_passrate_in_mask(final_passrates, mask),
            }
        )

    threshold_bin_rows: list[dict[str, Any]] = []
    if include_threshold_bins:
        mins = (
            list(threshold_slope_mins)
            if threshold_slope_mins is not None
            else _default_threshold_slope_mins(edges)
        )
        threshold_bin_rows = _build_threshold_min_bins(
            slopes, correct, final_passrates, mins, slope_max_right=threshold_slope_max
        )

    overall: dict[str, Any] = {
        "n_answers": len(rows),
        "n_correct": int(correct.sum()) if rows else 0,
        "accuracy": float(correct.mean()) if rows else None,
    }
    if rows and np.std(slopes) > 0 and np.std(correct) > 0:
        overall["pearson_slope_correct"] = float(np.corrcoef(slopes, correct)[0, 1])
    else:
        overall["pearson_slope_correct"] = None

    if rows:
        correct_mask = correct > 0.5
        overall["mean_slope_correct"] = (
            float(slopes[correct_mask].mean()) if correct_mask.any() else None
        )
        overall["mean_slope_incorrect"] = (
            float(slopes[~correct_mask].mean()) if (~correct_mask).any() else None
        )
        overall["mean_final_passrate_correct"] = (
            float(final_passrates[correct_mask].mean()) if correct_mask.any() else None
        )
        overall["mean_final_passrate_incorrect"] = (
            float(final_passrates[~correct_mask].mean()) if (~correct_mask).any() else None
        )
    else:
        overall["mean_slope_correct"] = None
        overall["mean_slope_incorrect"] = None
        overall["mean_final_passrate_correct"] = None
        overall["mean_final_passrate_incorrect"] = None

    out: dict[str, Any] = {
        "n_bins": n_bins,
        "slope_edges": [float(x) for x in edges],
        "bins": bin_rows,
        "overall": overall,
    }
    if include_threshold_bins:
        out["threshold_slope_max"] = float(threshold_slope_max)
        out["threshold_bins"] = threshold_bin_rows
    return out


def analyze_slope_given_final_majority(
    analysis: dict[str, Any],
    summary_records: list[dict],
    *,
    hist_len: int = 8,
    sample_type: str = "unlabeled",
    n_bins: int = 10,
    bin_edges: list[float] | np.ndarray | None = None,
    threshold_slope_mins: list[float] | None = None,
    threshold_slope_max: float = 1.0,
    include_threshold_bins: bool = True,
) -> SlopeFinalMajAnalysisResult:
    """Bin slope vs correctness on rows with ``is_current_majority=True`` only."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    all_rows, maj_rows = build_final_majority_answer_rows(
        analysis, summary_records, hist_len=hist_len, sample_type=sample_type
    )
    n_samples = len({r["sample_index"] for r in all_rows})
    n_maj_samples = len({r["sample_index"] for r in maj_rows})

    maj_correct = sum(1 for r in maj_rows if r["is_correct"])
    result: SlopeFinalMajAnalysisResult = {
        "filter": {
            "description": "answer string equals final probe round majority (current_majority)",
            "hist_len": hist_len,
            "sample_type": sample_type,
            "n_samples_all": n_samples,
            "n_samples_with_majority_row": n_maj_samples,
            "n_answers_all": len(all_rows),
            "n_answers_majority_only": len(maj_rows),
            "majority_answer_accuracy": maj_correct / len(maj_rows) if maj_rows else None,
        },
        **_bin_slopes(
            maj_rows,
            n_bins=n_bins,
            bin_edges=bin_edges,
            threshold_slope_mins=threshold_slope_mins,
            threshold_slope_max=threshold_slope_max,
            include_threshold_bins=include_threshold_bins,
        ),
    }

    # Reference: same binning on all answers (optional comparison in JSON)
    if all_rows:
        all_slopes = np.asarray([r["slope"] for r in all_rows], dtype=np.float64)
        all_correct = np.asarray([1.0 if r["is_correct"] else 0.0 for r in all_rows])
        if np.std(all_slopes) > 0 and np.std(all_correct) > 0:
            result["filter"]["pearson_slope_correct_all_answers"] = float(
                np.corrcoef(all_slopes, all_correct)[0, 1]
            )
    return result


def analyze_slope_final_maj_from_files(
    analysis_path: str | Path,
    summary_path: str | Path,
    **kwargs: Any,
) -> SlopeFinalMajAnalysisResult:
    return analyze_slope_given_final_majority(
        load_unlabeled_answer_analysis(analysis_path),
        load_probe_summaries(summary_path),
        **kwargs,
    )


def save_slope_final_maj_analysis(
    result: SlopeFinalMajAnalysisResult, output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _print_bin_table(title: str, rows: list[dict[str, Any]], *, threshold_min: bool = False) -> None:
    print(title)
    print(
        f"{'bin':>4}  {'slope_range':>24}  {'count':>7}  {'correct':>7}  "
        f"{'accuracy':>9}  {'mean_pr':>9}"
    )
    for row in rows:
        acc = row["accuracy"]
        acc_str = f"{acc:.4f}" if acc is not None else "n/a"
        mfp = row.get("mean_final_passrate")
        mfp_str = f"{mfp:.4f}" if mfp is not None else "n/a"
        if threshold_min:
            slope_range = f"({row['slope_min']:.4f}, {row['slope_max']:.4f}]"
        else:
            left = "(" if not row.get("slope_min_inclusive", True) else "["
            right = "]" if row.get("slope_max_inclusive", False) else ")"
            slope_range = f"{left}{row['slope_min']:.4f}, {row['slope_max']:.4f}{right}"
        print(
            f"{row['bin_id']:>4}  {slope_range:>24}  {row['count']:>7}  "
            f"{row['correct_count']:>7}  {acc_str:>9}  {mfp_str:>9}"
        )


def print_slope_final_maj_summary(result: SlopeFinalMajAnalysisResult) -> None:
    filt = result.get("filter") or {}
    overall = result.get("overall") or {}
    print("=== Slope vs correctness (final majority answers only) ===")
    print(
        f"filter: {filt.get('description')}  sample_type={filt.get('sample_type')}  "
        f"hist_len={filt.get('hist_len')}"
    )
    print(
        f"n_samples={filt.get('n_samples_with_majority_row')}/{filt.get('n_samples_all')}  "
        f"n_answers={filt.get('n_answers_majority_only')} (all candidates: {filt.get('n_answers_all')})  "
        f"majority_accuracy={filt.get('majority_answer_accuracy'):.4f}"
    )
    if filt.get("pearson_slope_correct_all_answers") is not None:
        print(
            f"pearson(slope,correct) all_answers={filt['pearson_slope_correct_all_answers']:.4f}  "
            f"majority_only={overall.get('pearson_slope_correct')}"
        )
    else:
        print(f"pearson(slope,correct) majority_only={overall.get('pearson_slope_correct')}")
    if overall.get("mean_slope_correct") is not None:
        print(
            f"mean_slope correct={overall['mean_slope_correct']:.6f}  "
            f"incorrect={overall.get('mean_slope_incorrect')}"
        )
    if overall.get("mean_final_passrate_correct") is not None:
        print(
            f"mean_pr correct={overall['mean_final_passrate_correct']:.4f}  "
            f"incorrect={overall.get('mean_final_passrate_incorrect')}"
        )
    _print_bin_table("--- Equal-width slope bins (majority rows) ---", result.get("bins") or [])
    threshold_rows = result.get("threshold_bins") or []
    if threshold_rows:
        _print_bin_table(
            f"--- Threshold bins slope > min (majority rows, right={result.get('threshold_slope_max', 1.0)}) ---",
            threshold_rows,
            threshold_min=True,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slope vs correctness for answers equal to final-round majority.",
    )
    parser.add_argument("analysis_path", type=Path, help="unlabeled_answer_analysis.json")
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
        help="Output JSON (default: <dir>/slope_final_maj_analysis.json)",
    )
    parser.add_argument("--hist-len", type=int, default=8, help="Majority history length")
    parser.add_argument(
        "--sample-type",
        choices=("labeled", "unlabeled", "all"),
        default="unlabeled",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--threshold-slope-mins",
        type=str,
        default=None,
        help="Comma-separated slope_min for slope > min bins",
    )
    parser.add_argument("--threshold-slope-max", type=float, default=1.0)
    parser.add_argument("--no-threshold-bins", action="store_true")
    return parser


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> SlopeFinalMajAnalysisResult:
    args = _build_argparser().parse_args(argv)
    summary_path = args.summary_path or (args.analysis_path.parent / "all_probes_summary.jsonl")
    threshold_mins = _parse_float_list(args.threshold_slope_mins) if args.threshold_slope_mins else None
    result = analyze_slope_final_maj_from_files(
        args.analysis_path,
        summary_path,
        hist_len=args.hist_len,
        sample_type=args.sample_type,
        n_bins=args.n_bins,
        threshold_slope_mins=threshold_mins,
        threshold_slope_max=args.threshold_slope_max,
        include_threshold_bins=not args.no_threshold_bins,
    )
    output = args.output or (args.analysis_path.parent / "slope_final_maj_analysis.json")
    save_slope_final_maj_analysis(result, output)
    print_slope_final_maj_summary(result)
    print(f"Wrote -> {output}")
    return result


if __name__ == "__main__":
    main()
