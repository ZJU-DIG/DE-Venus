"""Analyze TraPO probe summary dumps (``all_probes_summary.jsonl``)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from math_verify import parse, verify
from scipy.stats import linregress

AnswerStats = list[Any]  # [passrate_trajectory, slope, is_correct]
SampleAnswerAnalysis = dict[str, AnswerStats]
AnswerAnalysisResult = dict[int, dict[str, Any]]

SampleTypeFilter = Literal["labeled", "unlabeled", "all"]


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


def _is_labeled_record(record: dict) -> bool:
    sample_type = record.get("sample_type")
    if sample_type is not None:
        return str(sample_type).lower() == "labeled"
    labeled = record.get("labeled")
    if labeled is None:
        return False
    return labeled is True or labeled == 1 or str(labeled).lower() == "true"


def _is_unlabeled_record(record: dict) -> bool:
    sample_type = record.get("sample_type")
    if sample_type is not None:
        return str(sample_type).lower() == "unlabeled"
    labeled = record.get("labeled")
    if labeled is None:
        return False
    return labeled is False or labeled == 0 or str(labeled).lower() == "false"


def _matches_sample_type(record: dict, sample_type: SampleTypeFilter) -> bool:
    if sample_type == "all":
        return True
    if sample_type == "labeled":
        return _is_labeled_record(record)
    return _is_unlabeled_record(record)


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


def analyze_answer_trajectories(
    records: list[dict],
    *,
    sample_type: SampleTypeFilter = "unlabeled",
) -> AnswerAnalysisResult:
    """Per sample index, analyze each canonical answer across probe rounds.

    ``sample_type``: ``labeled``, ``unlabeled``, or ``all``.
    """
    by_index: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if not _matches_sample_type(record, sample_type):
            continue
        if "index" not in record:
            continue
        by_index[int(record["index"])].append(record)

    result: AnswerAnalysisResult = {}
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


def analyze_unlabeled_answer_trajectories(records: list[dict]) -> AnswerAnalysisResult:
    """Backward-compatible wrapper for unlabeled-only analysis."""
    return analyze_answer_trajectories(records, sample_type="unlabeled")


def analyze_labeled_answer_trajectories(records: list[dict]) -> AnswerAnalysisResult:
    return analyze_answer_trajectories(records, sample_type="labeled")


def analyze_probe_summary_file(
    path: str | Path,
    *,
    sample_type: SampleTypeFilter = "unlabeled",
) -> AnswerAnalysisResult:
    """Load ``all_probes_summary.jsonl`` and run answer-trajectory analysis."""
    return analyze_answer_trajectories(load_probe_summaries(path), sample_type=sample_type)


def load_answer_analysis(path: str | Path) -> dict[str, Any]:
    """Load answer analysis JSON (sample index keys are strings)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_unlabeled_answer_analysis(path: str | Path) -> dict[str, Any]:
    """Load ``unlabeled_answer_analysis.json``."""
    return load_answer_analysis(path)


def load_labeled_answer_analysis(path: str | Path) -> dict[str, Any]:
    """Load ``labeled_answer_analysis.json``."""
    return load_answer_analysis(path)


def save_analysis(result: AnswerAnalysisResult, output_path: str | Path) -> None:
    """Write analysis result as JSON (keys stringified for sample indices)."""
    payload = {str(k): v for k, v in result.items()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _default_output_path(summary_path: Path, sample_type: SampleTypeFilter) -> Path:
    if sample_type == "labeled":
        return summary_path.parent / "labeled_answer_analysis.json"
    if sample_type == "all":
        return summary_path.parent / "answer_analysis.json"
    return summary_path.parent / "unlabeled_answer_analysis.json"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze answer passrate trajectories from probe summary JSONL.",
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to all_probes_summary.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default depends on --sample-type)",
    )
    parser.add_argument(
        "--sample-type",
        choices=("labeled", "unlabeled", "all"),
        default="unlabeled",
        help="Which samples to include (default: unlabeled)",
    )
    return parser


def main(argv: list[str] | None = None) -> AnswerAnalysisResult:
    args = _build_argparser().parse_args(argv)
    sample_type: SampleTypeFilter = args.sample_type
    result = analyze_probe_summary_file(args.summary_path, sample_type=sample_type)

    output = args.output or _default_output_path(args.summary_path, sample_type)
    save_analysis(result, output)

    n_samples = len(result)
    n_answers = sum(len(v["answers"]) for v in result.values())
    print(
        f"Analyzed {n_samples} {sample_type} samples, {n_answers} canonical answers -> {output}"
    )
    return result


if __name__ == "__main__":
    main()
