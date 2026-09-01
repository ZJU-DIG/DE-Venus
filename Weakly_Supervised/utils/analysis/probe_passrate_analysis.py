"""Final-passrate bin analysis of unlabeled probe answers (``unlabeled_answer_analysis.json``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# from .probe_slope_analysis import load_unlabeled_answer_analysis

FlatAnswerRecord = dict[str, Any]
PassrateBinAnalysisResult = dict[str, Any]

def load_unlabeled_answer_analysis(path: str | Path) -> dict[str, Any]:
    """Load ``unlabeled_answer_analysis.json`` (sample index keys are strings)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_flat_passrate_records(analysis: dict[str, Any]) -> list[FlatAnswerRecord]:
    """One row per (sample, answer); ``final_passrate`` = last element of passrate trajectory."""
    rows: list[FlatAnswerRecord] = []
    for sample_key, sample in analysis.items():
        answers = sample.get("answers") or {}
        for answer_str, stats in answers.items():
            if not isinstance(stats, (list, tuple)) or len(stats) < 3:
                continue
            passrate_traj = stats[0]
            if not passrate_traj:
                continue
            slope, is_correct = stats[1], stats[2]
            final_passrate = float(passrate_traj[-1])
            first_passrate = float(passrate_traj[0])
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


def _make_passrate_bin_edges(passrates: np.ndarray, n_bins: int) -> np.ndarray:
    if passrates.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo, hi = float(np.min(passrates)), float(np.max(passrates))
    if lo == hi:
        pad = 1e-6 if lo == 0.0 else abs(lo) * 1e-3
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_bins + 1)


def _default_threshold_passrate_mins(edges: np.ndarray) -> list[float]:
    return sorted({float(x) for x in edges[:-1]})


def _build_threshold_min_bins(
    passrates: np.ndarray,
    correct: np.ndarray,
    first_passrates: np.ndarray,
    slopes: np.ndarray,
    passrate_mins: list[float],
    passrate_max_right: float = 1.0,
) -> list[dict[str, Any]]:
    """Keep rows with ``final_passrate > passrate_min`` (display right end ``passrate_max_right``)."""
    bin_rows: list[dict[str, Any]] = []
    for bin_id, passrate_min in enumerate(passrate_mins):
        if passrates.size == 0:
            mask = np.zeros(0, dtype=bool)
        else:
            mask = passrates > passrate_min
        count = int(mask.sum())
        correct_count = int(correct[mask].sum()) if count else 0
        bin_rows.append(
            {
                "bin_id": bin_id,
                "bin_type": "threshold_min",
                "passrate_min": float(passrate_min),
                "passrate_max": float(passrate_max_right),
                "passrate_min_exclusive": True,
                "passrate_max_inclusive": True,
                "count": count,
                "correct_count": correct_count,
                "incorrect_count": count - correct_count,
                "accuracy": correct_count / count if count else None,
                **_bin_trajectory_means(mask, passrates, first_passrates, slopes),
            }
        )
    return bin_rows


def analyze_final_passrate_vs_correctness(
    analysis: dict[str, Any],
    n_bins: int = 10,
    bin_edges: list[float] | np.ndarray | None = None,
    threshold_passrate_mins: list[float] | None = None,
    threshold_passrate_max: float = 1.0,
    include_threshold_bins: bool = True,
) -> PassrateBinAnalysisResult:
    """Bin by last passrate in trajectory and summarize correctness per bin."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    rows = iter_flat_passrate_records(analysis)
    passrates = np.asarray([r["final_passrate"] for r in rows], dtype=np.float64)
    first_passrates = np.asarray([r["first_passrate"] for r in rows], dtype=np.float64)
    slopes = np.asarray([r["slope"] for r in rows], dtype=np.float64)
    correct = np.asarray([1.0 if r["is_correct"] else 0.0 for r in rows], dtype=np.float64)

    if bin_edges is None:
        edges = _make_passrate_bin_edges(passrates, n_bins)
    else:
        edges = np.asarray(bin_edges, dtype=np.float64)
        n_bins = len(edges) - 1

    bin_rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        if not rows:
            mask = np.zeros(0, dtype=bool)
        elif b < n_bins - 1:
            mask = (passrates >= edges[b]) & (passrates < edges[b + 1])
        else:
            mask = (passrates >= edges[b]) & (passrates <= edges[b + 1])
        count = int(mask.sum())
        correct_count = int(correct[mask].sum()) if count else 0
        bin_rows.append(
            {
                "bin_id": b,
                "bin_type": "interval",
                "passrate_min": float(edges[b]),
                "passrate_max": float(edges[b + 1]),
                "passrate_min_inclusive": True,
                "passrate_max_inclusive": b == n_bins - 1,
                "count": count,
                "correct_count": correct_count,
                "incorrect_count": count - correct_count,
                "accuracy": correct_count / count if count else None,
                **_bin_trajectory_means(mask, passrates, first_passrates, slopes),
            }
        )

    threshold_bin_rows: list[dict[str, Any]] = []
    if include_threshold_bins:
        mins = (
            list(threshold_passrate_mins)
            if threshold_passrate_mins is not None
            else _default_threshold_passrate_mins(edges)
        )
        threshold_bin_rows = _build_threshold_min_bins(
            passrates,
            correct,
            first_passrates,
            slopes,
            mins,
            passrate_max_right=threshold_passrate_max,
        )

    overall: dict[str, Any] = {
        "n_answers": len(rows),
        "n_correct": int(correct.sum()) if rows else 0,
        "accuracy": float(correct.mean()) if rows else None,
    }
    if rows and np.std(passrates) > 0 and np.std(correct) > 0:
        overall["pearson_final_passrate_correct"] = float(np.corrcoef(passrates, correct)[0, 1])
    else:
        overall["pearson_final_passrate_correct"] = None

    if rows:
        correct_mask = correct > 0.5
        overall["mean_final_passrate_correct"] = (
            float(passrates[correct_mask].mean()) if correct_mask.any() else None
        )
        overall["mean_final_passrate_incorrect"] = (
            float(passrates[~correct_mask].mean()) if (~correct_mask).any() else None
        )
    else:
        overall["mean_final_passrate_correct"] = None
        overall["mean_final_passrate_incorrect"] = None

    if rows:
        all_mask = np.ones(len(rows), dtype=bool)
        overall.update(_bin_trajectory_means(all_mask, passrates, first_passrates, slopes))
    else:
        overall["mean_final_passrate"] = None
        overall["mean_first_passrate"] = None
        overall["mean_slope"] = None

    result: PassrateBinAnalysisResult = {
        "n_bins": n_bins,
        "passrate_edges": [float(x) for x in edges],
        "bins": bin_rows,
        "overall": overall,
    }
    if include_threshold_bins:
        result["threshold_passrate_max"] = float(threshold_passrate_max)
        result["threshold_bins"] = threshold_bin_rows
    return result


def analyze_passrate_bins_from_file(
    analysis_path: str | Path,
    n_bins: int = 10,
    bin_edges: list[float] | np.ndarray | None = None,
    threshold_passrate_mins: list[float] | None = None,
    threshold_passrate_max: float = 1.0,
    include_threshold_bins: bool = True,
) -> PassrateBinAnalysisResult:
    """Load ``unlabeled_answer_analysis.json`` and run final-passrate bin analysis."""
    return analyze_final_passrate_vs_correctness(
        load_unlabeled_answer_analysis(analysis_path),
        n_bins=n_bins,
        bin_edges=bin_edges,
        threshold_passrate_mins=threshold_passrate_mins,
        threshold_passrate_max=threshold_passrate_max,
        include_threshold_bins=include_threshold_bins,
    )


def save_passrate_bin_analysis(result: PassrateBinAnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _fmt_mean(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _print_bin_table(title: str, rows: list[dict[str, Any]], *, threshold_min: bool = False) -> None:
    print(title)
    print(
        f"{'bin':>4}  {'passrate_range':>24}  {'count':>7}  {'correct':>7}  "
        f"{'accuracy':>9}  {'mean_final':>10}  {'mean_first':>10}  {'mean_slope':>10}"
    )
    for row in rows:
        acc = row["accuracy"]
        acc_str = f"{acc:.4f}" if acc is not None else "n/a"
        if threshold_min:
            pr_range = f"({row['passrate_min']:.4f}, {row['passrate_max']:.4f}]"
        else:
            left = "[" if row.get("passrate_min_inclusive", True) else "("
            right = "]" if row.get("passrate_max_inclusive", False) else ")"
            pr_range = f"{left}{row['passrate_min']:.4f}, {row['passrate_max']:.4f}{right}"
        print(
            f"{row['bin_id']:>4}  {pr_range:>24}  {row['count']:>7}  "
            f"{row['correct_count']:>7}  {acc_str:>9}  "
            f"{_fmt_mean(row.get('mean_final_passrate')):>10}  "
            f"{_fmt_mean(row.get('mean_first_passrate')):>10}  "
            f"{_fmt_mean(row.get('mean_slope')):>10}"
        )


def print_passrate_bin_summary(result: PassrateBinAnalysisResult) -> None:
    """Print equal-width and optional threshold-min passrate bins."""
    overall = result.get("overall") or {}
    print("=== Final passrate bin vs answer correctness ===")
    print(
        f"n_answers={overall.get('n_answers')}  "
        f"accuracy={overall.get('accuracy')}  "
        f"pearson(final_passrate,correct)={overall.get('pearson_final_passrate_correct')}"
    )
    if overall.get("mean_final_passrate") is not None:
        print(
            "selected (all rows): "
            f"mean_final_passrate={overall['mean_final_passrate']:.6f}  "
            f"mean_first_passrate={_fmt_mean(overall.get('mean_first_passrate'))}  "
            f"mean_slope={_fmt_mean(overall.get('mean_slope'))}"
        )
    if overall.get("mean_final_passrate_correct") is not None:
        print(
            f"mean_final_passrate correct={overall['mean_final_passrate_correct']:.6f}  "
            f"incorrect={overall.get('mean_final_passrate_incorrect')}"
        )
    _print_bin_table("--- Equal-width interval bins ---", result.get("bins") or [])
    threshold_rows = result.get("threshold_bins") or []
    if threshold_rows:
        _print_bin_table(
            f"--- Threshold bins (passrate > min, right end={result.get('threshold_passrate_max', 1.0)}) ---",
            threshold_rows,
            threshold_min=True,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze final passrate vs correctness from unlabeled_answer_analysis.json.",
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
        help="Output JSON (default: <dir>/unlabeled_passrate_bin_analysis.json)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal-width final-passrate bins",
    )
    parser.add_argument(
        "--threshold-passrate-mins",
        type=str,
        default=None,
        help="Comma-separated passrate_min for (min, 1.0] threshold bins",
    )
    parser.add_argument(
        "--threshold-passrate-max",
        type=float,
        default=1.0,
        help="Displayed right endpoint for threshold bins",
    )
    parser.add_argument(
        "--no-threshold-bins",
        action="store_true",
        help="Skip threshold_min bins",
    )
    return parser


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> PassrateBinAnalysisResult:
    args = _build_argparser().parse_args(argv)
    threshold_mins = (
        _parse_float_list(args.threshold_passrate_mins) if args.threshold_passrate_mins else None
    )
    result = analyze_passrate_bins_from_file(
        args.analysis_path,
        n_bins=args.n_bins,
        threshold_passrate_mins=threshold_mins,
        threshold_passrate_max=args.threshold_passrate_max,
        include_threshold_bins=not args.no_threshold_bins,
    )
    output = args.output or (args.analysis_path.parent / "unlabeled_passrate_bin_analysis.json")
    save_passrate_bin_analysis(result, output)
    print_passrate_bin_summary(result)
    print(f"Wrote passrate-bin analysis -> {output}")
    return result


if __name__ == "__main__":
    main()
