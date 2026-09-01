"""Slope-bin analysis of unlabeled answer trajectories (``unlabeled_answer_analysis.json``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

FlatAnswerRecord = dict[str, Any]
SlopeBinAnalysisResult = dict[str, Any]


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
            first_passrate = float(passrate_traj[0]) if passrate_traj else 0.0
            rows.append(
                {
                    "sample_index": int(sample_key),
                    "answer": answer_str,
                    "passrate_trajectory": list(passrate_traj),
                    "first_passrate": first_passrate,
                    "final_passrate": final_passrate,
                    "slope": float(slope),
                    "is_correct": bool(is_correct),
                }
            )
    return rows


def _mean_in_mask(values: np.ndarray, mask: np.ndarray) -> float | None:
    if values.size == 0 or not mask.any():
        return None
    return float(values[mask].mean())


def _mean_final_passrate_in_mask(final_passrates: np.ndarray, mask: np.ndarray) -> float | None:
    return _mean_in_mask(final_passrates, mask)


def _bin_trajectory_means(
    mask: np.ndarray,
    final_passrates: np.ndarray,
    first_passrates: np.ndarray,
    slopes: np.ndarray,
) -> dict[str, float | None]:
    return {
        "mean_final_passrate": _mean_in_mask(final_passrates, mask),
        "mean_first_passrate": _mean_in_mask(first_passrates, mask),
        "mean_slope": _mean_in_mask(slopes, mask),
    }


def _make_slope_bin_edges(slopes: np.ndarray, n_bins: int) -> np.ndarray:
    if slopes.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo, hi = float(np.min(slopes)), float(np.max(slopes))
    if lo == hi:
        pad = 1e-6 if lo == 0.0 else abs(lo) * 1e-3
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_bins + 1)


def _default_threshold_slope_mins(edges: np.ndarray) -> list[float]:
    """Use equal-width bin left edges as ``slope > min`` thresholds."""
    return sorted({float(x) for x in edges[:-1]})


def _build_threshold_min_bins(
    slopes: np.ndarray,
    correct: np.ndarray,
    final_passrates: np.ndarray,
    first_passrates: np.ndarray,
    slope_mins: list[float],
    slope_max_right: float = 1.0,
) -> list[dict[str, Any]]:
    """Bins ``(slope_min, slope_max_right]``: keep rows with ``slope > slope_min`` only."""
    bin_rows: list[dict[str, Any]] = []
    for bin_id, slope_min in enumerate(slope_mins):
        if slopes.size == 0:
            mask = np.zeros(0, dtype=bool)
        else:
            mask = slopes > slope_min
        count = int(mask.sum())
        correct_count = int(correct[mask].sum()) if count else 0
        bin_rows.append(
            {
                "bin_id": bin_id,
                "bin_type": "threshold_min",
                "slope_min": float(slope_min),
                "slope_max": float(slope_max_right),
                "slope_min_exclusive": True,
                "slope_max_inclusive": True,
                "count": count,
                "correct_count": correct_count,
                "incorrect_count": count - correct_count,
                "accuracy": correct_count / count if count else None,
                **_bin_trajectory_means(mask, final_passrates, first_passrates, slopes),
            }
        )
    return bin_rows


def analyze_slope_vs_correctness(
    analysis: dict[str, Any],
    n_bins: int = 10,
    bin_edges: list[float] | np.ndarray | None = None,
    threshold_slope_mins: list[float] | None = None,
    threshold_slope_max: float = 1.0,
    include_threshold_bins: bool = True,
) -> SlopeBinAnalysisResult:
    """Bin answers by passrate-trajectory slope and summarize correctness per bin.

    Each entry in ``analysis`` is one unlabeled sample; each answer maps to
    ``[passrate_list, slope, is_correct]``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    rows = iter_flat_answer_records(analysis)
    slopes = np.asarray([r["slope"] for r in rows], dtype=np.float64)
    final_passrates = np.asarray([r["final_passrate"] for r in rows], dtype=np.float64)
    first_passrates = np.asarray([r["first_passrate"] for r in rows], dtype=np.float64)
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
                **_bin_trajectory_means(mask, final_passrates, first_passrates, slopes),
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
            slopes,
            correct,
            final_passrates,
            first_passrates,
            mins,
            slope_max_right=threshold_slope_max,
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
        overall["mean_slope_correct"] = float(slopes[correct_mask].mean()) if correct_mask.any() else None
        overall["mean_slope_incorrect"] = (
            float(slopes[~correct_mask].mean()) if (~correct_mask).any() else None
        )
    else:
        overall["mean_slope_correct"] = None
        overall["mean_slope_incorrect"] = None

    if rows:
        all_mask = np.ones(len(rows), dtype=bool)
        overall.update(_bin_trajectory_means(all_mask, final_passrates, first_passrates, slopes))
    else:
        overall["mean_final_passrate"] = None
        overall["mean_first_passrate"] = None
        overall["mean_slope"] = None

    result: SlopeBinAnalysisResult = {
        "n_bins": n_bins,
        "slope_edges": [float(x) for x in edges],
        "bins": bin_rows,
        "overall": overall,
    }
    if include_threshold_bins:
        result["threshold_slope_max"] = float(threshold_slope_max)
        result["threshold_bins"] = threshold_bin_rows
    return result


def analyze_slope_bins_from_file(
    analysis_path: str | Path,
    n_bins: int = 10,
    bin_edges: list[float] | np.ndarray | None = None,
    threshold_slope_mins: list[float] | None = None,
    threshold_slope_max: float = 1.0,
    include_threshold_bins: bool = True,
) -> SlopeBinAnalysisResult:
    """Load ``unlabeled_answer_analysis.json`` and run slope-bin correctness analysis."""
    return analyze_slope_vs_correctness(
        load_unlabeled_answer_analysis(analysis_path),
        n_bins=n_bins,
        bin_edges=bin_edges,
        threshold_slope_mins=threshold_slope_mins,
        threshold_slope_max=threshold_slope_max,
        include_threshold_bins=include_threshold_bins,
    )


def save_slope_bin_analysis(result: SlopeBinAnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _fmt_mean(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _print_bin_table(title: str, rows: list[dict[str, Any]], *, threshold_min: bool = False) -> None:
    print(title)
    print(
        f"{'bin':>4}  {'slope_range':>24}  {'count':>7}  {'correct':>7}  "
        f"{'accuracy':>9}  {'mean_final':>10}  {'mean_first':>10}  {'mean_slope':>10}"
    )
    for row in rows:
        acc = row["accuracy"]
        acc_str = f"{acc:.4f}" if acc is not None else "n/a"
        if threshold_min:
            slope_range = f"({row['slope_min']:.4f}, {row['slope_max']:.4f}]"
        else:
            left = "(" if not row.get("slope_min_inclusive", True) else "["
            right = "]" if row.get("slope_max_inclusive", False) else ")"
            slope_range = f"{left}{row['slope_min']:.4f}, {row['slope_max']:.4f}{right}"
        print(
            f"{row['bin_id']:>4}  {slope_range:>24}  {row['count']:>7}  "
            f"{row['correct_count']:>7}  {acc_str:>9}  "
            f"{_fmt_mean(row.get('mean_final_passrate')):>10}  "
            f"{_fmt_mean(row.get('mean_first_passrate')):>10}  "
            f"{_fmt_mean(row.get('mean_slope')):>10}"
        )


def print_slope_bin_summary(result: SlopeBinAnalysisResult) -> None:
    """Print equal-width interval bins and optional ``slope > min`` threshold bins."""
    overall = result.get("overall") or {}
    print("=== Slope bin vs answer correctness ===")
    print(
        f"n_answers={overall.get('n_answers')}  "
        f"accuracy={overall.get('accuracy')}  "
        f"pearson(slope,correct)={overall.get('pearson_slope_correct')}"
    )
    if overall.get("mean_final_passrate") is not None:
        print(
            "selected (all rows): "
            f"mean_final_passrate={overall['mean_final_passrate']:.6f}  "
            f"mean_first_passrate={_fmt_mean(overall.get('mean_first_passrate'))}  "
            f"mean_slope={_fmt_mean(overall.get('mean_slope'))}"
        )
    if overall.get("mean_slope_correct") is not None:
        print(
            f"mean_slope correct={overall['mean_slope_correct']:.6f}  "
            f"incorrect={overall.get('mean_slope_incorrect')}"
        )
    _print_bin_table("--- Equal-width interval bins ---", result.get("bins") or [])
    threshold_rows = result.get("threshold_bins") or []
    if threshold_rows:
        _print_bin_table(
            f"--- Threshold bins (slope > slope_min, right end={result.get('threshold_slope_max', 1.0)}) ---",
            threshold_rows,
            threshold_min=True,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze slope vs correctness from unlabeled_answer_analysis.json.",
    )
    parser.add_argument(
        "analysis_path",
        type=Path,
        help="Path to unlabeled_answer_analysis.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <dir>/unlabeled_slope_bin_analysis.json)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal-width slope bins",
    )
    parser.add_argument(
        "--threshold-slope-mins",
        type=str,
        default=None,
        help="Comma-separated slope_min list for (min, 1.0] threshold bins; "
        "default: left edges of equal-width bins",
    )
    parser.add_argument(
        "--threshold-slope-max",
        type=float,
        default=1.0,
        help="Displayed right endpoint for threshold bins (selection is slope > min only)",
    )
    parser.add_argument(
        "--no-threshold-bins",
        action="store_true",
        help="Skip threshold_min bins in output",
    )
    return parser


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> SlopeBinAnalysisResult:
    args = _build_argparser().parse_args(argv)
    threshold_mins = _parse_float_list(args.threshold_slope_mins) if args.threshold_slope_mins else None
    result = analyze_slope_bins_from_file(
        args.analysis_path,
        n_bins=args.n_bins,
        threshold_slope_mins=threshold_mins,
        threshold_slope_max=args.threshold_slope_max,
        include_threshold_bins=not args.no_threshold_bins,
    )
    output = args.output or (args.analysis_path.parent / "unlabeled_slope_bin_analysis.json")
    save_slope_bin_analysis(result, output)
    print_slope_bin_summary(result)
    print(f"Wrote slope-bin analysis -> {output}")
    return result


if __name__ == "__main__":
    main()
