"""Train classifiers on passrate trajectories to predict answer correctness.

Modes:
  - Single pool: grouped CV on one ``*_answer_analysis.json`` (legacy).
  - Train / predict: fit on ``labeled_answer_analysis.json``, evaluate on
    ``unlabeled_answer_analysis.json`` (labels on unlabeled used for metrics only).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .probes import load_answer_analysis, load_probe_summaries
    from .probe_slope_history_consistency_analysis import (
        _majority_row,
        _passrate_slope,
        _sample_level_prf,
        build_sample_majority_meta,
        enrich_answer_records,
    )
except ImportError:  # ``python path/to/probe_ml_analysis.py`` from analysis/
    from .probes import load_answer_analysis, load_probe_summaries
    from probe_slope_history_consistency_analysis import (
        _majority_row,
        _passrate_slope,
        _sample_level_prf,
        build_sample_majority_meta,
        enrich_answer_records,
    )

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "probe_ml_analysis requires scikit-learn. Install with: pip install scikit-learn"
    ) from exc

MlAnalysisResult = dict[str, Any]
RowDict = dict[str, Any]


def _trajectory_engineered(traj: list[float]) -> dict[str, float]:
    if not traj:
        return {
            "traj_len": 0.0,
            "passrate_first": 0.0,
            "passrate_last": 0.0,
            "passrate_mean": 0.0,
            "passrate_std": 0.0,
            "passrate_min": 0.0,
            "passrate_max": 0.0,
            "passrate_range": 0.0,
            "passrate_slope": 0.0,
            "passrate_delta": 0.0,
        }
    y = np.asarray(traj, dtype=np.float64)
    return {
        "traj_len": float(len(traj)),
        "passrate_first": float(y[0]),
        "passrate_last": float(y[-1]),
        "passrate_mean": float(y.mean()),
        "passrate_std": float(y.std()) if len(y) > 1 else 0.0,
        "passrate_min": float(y.min()),
        "passrate_max": float(y.max()),
        "passrate_range": float(y.max() - y.min()),
        "passrate_slope": _passrate_slope(list(traj)),
        "passrate_delta": float(y[-1] - y[0]),
    }


ENGINEERED_NAMES = list(_trajectory_engineered([]).keys())


def _infer_max_trajectory_len(rows: list[RowDict]) -> int:
    return max((len(r["passrate_trajectory"]) for r in rows), default=0)


def build_feature_matrix(
    rows: list[RowDict],
    *,
    max_traj_len: int | None = None,
    include_meta: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Features: padded trajectory + engineered stats (+ optional majority meta)."""
    if max_traj_len is None:
        max_traj_len = _infer_max_trajectory_len(rows)
    traj_names = [f"passrate_t{i}" for i in range(max_traj_len)]
    meta_names = ["is_current_majority", "majority_consistency"] if include_meta else []
    feature_names = traj_names + ENGINEERED_NAMES + meta_names

    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        traj = list(row.get("passrate_trajectory") or [])
        for j in range(max_traj_len):
            X[i, j] = float(traj[j]) if j < len(traj) else 0.0
        off = max_traj_len
        eng = _trajectory_engineered(traj)
        for k, name in enumerate(ENGINEERED_NAMES):
            X[i, off + k] = eng[name]
        off += len(ENGINEERED_NAMES)
        if include_meta:
            X[i, off] = 1.0 if row.get("is_current_majority") else 0.0
            X[i, off + 1] = float(row.get("majority_consistency", 1.0))
    return X, feature_names


def _make_model(name: str) -> Any:
    if name == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=0,
                    ),
                ),
            ]
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        )
    if name == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=0,
        )
    raise ValueError(f"Unknown model: {name}. Choose from: logistic, rf, gbdt")


def _answer_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _per_sample_pick(
    rows: list[RowDict],
    prob_correct: np.ndarray,
    *,
    threshold: float | None = None,
    fallback: str = "majority",
    require_majority: bool = False,
) -> list[RowDict]:
    """One pseudo-label per sample: highest P(correct), optional threshold / fallback."""
    by_sample: dict[int, list[tuple[int, RowDict]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_sample[row["sample_index"]].append((idx, row))

    picked: list[RowDict] = []
    for candidates in by_sample.values():
        pool = list(candidates)
        if require_majority:
            pool = [(i, r) for i, r in pool if r.get("is_current_majority")]
        if not pool:
            pool = list(candidates)
        best_i, best_row = max(pool, key=lambda t: prob_correct[t[0]])
        if threshold is not None and prob_correct[best_i] < threshold:
            if fallback == "majority":
                maj = _majority_row([r for _, r in candidates])
                if maj is not None:
                    picked.append(maj)
            continue
        picked.append(best_row)
    return picked


def _sample_metrics_from_pick(
    rows: list[RowDict],
    picked: list[RowDict],
) -> dict[str, Any]:
    by_sample: dict[int, list[RowDict]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_index"]].append(row)
    n_samples = len(by_sample)
    out = _sample_level_prf(picked, n_samples)
    out["n_samples"] = n_samples
    out["n_samples_with_pick"] = len(picked)
    out["sample_coverage"] = len(picked) / n_samples if n_samples else None
    out["accuracy"] = out["precision"]
    return out


def _sweep_sample_threshold(
    rows: list[RowDict],
    prob_correct: np.ndarray,
    *,
    fallback: str = "majority",
    require_majority: bool = False,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    grid: list[dict[str, Any]] = []
    for thr in thresholds:
        picked = _per_sample_pick(
            rows,
            prob_correct,
            threshold=float(thr),
            fallback=fallback,
            require_majority=require_majority,
        )
        m = _sample_metrics_from_pick(rows, picked)
        m["threshold"] = float(thr)
        m["fallback"] = fallback
        m["require_majority"] = require_majority
        grid.append(m)
    return grid


def _oof_group_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    *,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold predicted P(correct) with GroupKFold by sample index."""
    prob = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = _make_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        # class 1 = correct
        pos_col = 1 if proba.shape[1] > 1 else 0
        prob[val_idx] = proba[:, pos_col]
    return prob


def _rows_for_analysis(
    analysis: dict[str, Any],
    summary_records: list[dict] | None,
    *,
    hist_len: int,
    sample_type: str,
    include_meta: bool,
) -> list[RowDict]:
    if summary_records and include_meta:
        sample_meta = build_sample_majority_meta(
            summary_records, hist_len=hist_len, sample_type=sample_type
        )
        return enrich_answer_records(analysis, sample_meta)
    return _rows_without_meta(analysis)


def _evaluate_model_on_rows(
    rows: list[RowDict],
    prob: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Answer-level + per-sample metrics for a fixed probability vector."""
    y = np.asarray([1 if r["is_correct"] else 0 for r in rows], dtype=np.int32)
    pred = (prob >= 0.5).astype(np.int32)
    picked_argmax = _per_sample_pick(rows, prob, threshold=None)
    out: dict[str, Any] = {
        "answer_level": _answer_level_metrics(y, pred, prob),
        "per_sample_argmax": _sample_metrics_from_pick(rows, picked_argmax),
    }
    if threshold is not None:
        picked_thr = _per_sample_pick(
            rows, prob, threshold=threshold, fallback="majority"
        )
        out["per_sample_tuned_threshold"] = _sample_metrics_from_pick(rows, picked_thr)
        out["threshold"] = float(threshold)
    thr_grid = _sweep_sample_threshold(rows, prob, fallback="majority")
    best_f1 = max(thr_grid, key=lambda m: m.get("f1") or -1.0)
    out["best_threshold_on_this_set"] = best_f1
    return out


def analyze_passrate_ml(
    analysis: dict[str, Any],
    summary_records: list[dict] | None = None,
    *,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    hist_len: int = 8,
    include_meta: bool = True,
    sample_type: str = "unlabeled",
) -> MlAnalysisResult:
    """Train/evaluate ML models with grouped CV on one analysis pool."""
    if model_names is None:
        model_names = ["logistic", "rf", "gbdt"]

    rows = _rows_for_analysis(
        analysis,
        summary_records,
        hist_len=hist_len,
        sample_type=sample_type,
        include_meta=include_meta,
    )

    max_len = _infer_max_trajectory_len(rows)
    X, feature_names = build_feature_matrix(rows, max_traj_len=max_len, include_meta=include_meta)
    y = np.asarray([1 if r["is_correct"] else 0 for r in rows], dtype=np.int32)
    groups = np.asarray([r["sample_index"] for r in rows], dtype=np.int32)

    baselines = _baseline_metrics(rows)

    result: MlAnalysisResult = {
        "mode": "single_pool_cv",
        "sample_type": sample_type,
        "n_samples": len({r["sample_index"] for r in rows}),
        "n_answer_records": len(rows),
        "positive_rate": float(y.mean()),
        "max_trajectory_len": max_len,
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "cv_n_splits": n_splits,
        "include_meta_features": include_meta,
        "baselines": baselines,
        "models": {},
    }

    for model_name in model_names:
        oof_prob = _oof_group_cv_predict(X, y, groups, model_name, n_splits=n_splits)
        evals = _evaluate_model_on_rows(rows, oof_prob)
        thr_grid = _sweep_sample_threshold(rows, oof_prob, fallback="majority")
        best_f1 = max(thr_grid, key=lambda m: m.get("f1") or -1.0)
        picked_best = _per_sample_pick(
            rows, oof_prob, threshold=best_f1["threshold"], fallback="majority"
        )

        full_model = _make_model(model_name)
        full_model.fit(X, y)
        importances = _feature_importances(full_model, feature_names)

        result["models"][model_name] = {
            "answer_level_oof": evals["answer_level"],
            "per_sample_argmax_oof": evals["per_sample_argmax"],
            "per_sample_best_threshold_oof": _sample_metrics_from_pick(rows, picked_best),
            "best_threshold": best_f1,
            "best_threshold_skip_fallback": max(
                _sweep_sample_threshold(rows, oof_prob, fallback="skip"),
                key=lambda m: m.get("f1") or -1.0,
            ),
            "best_threshold_require_majority": max(
                _sweep_sample_threshold(
                    rows, oof_prob, fallback="majority", require_majority=True
                ),
                key=lambda m: m.get("f1") or -1.0,
            ),
            "threshold_sweep": thr_grid,
            "feature_importances_top20": importances[:20],
        }

    return result


def analyze_labeled_train_unlabeled_predict(
    labeled_analysis: dict[str, Any],
    unlabeled_analysis: dict[str, Any],
    summary_records: list[dict],
    *,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    hist_len: int = 8,
    include_meta: bool = True,
) -> MlAnalysisResult:
    """Fit on labeled samples (GT labels); predict and evaluate on unlabeled (GT for metrics only)."""
    if model_names is None:
        model_names = ["logistic", "rf", "gbdt"]

    rows_train = _rows_for_analysis(
        labeled_analysis,
        summary_records,
        hist_len=hist_len,
        sample_type="labeled",
        include_meta=include_meta,
    )
    rows_predict = _rows_for_analysis(
        unlabeled_analysis,
        summary_records,
        hist_len=hist_len,
        sample_type="unlabeled",
        include_meta=include_meta,
    )

    max_len = max(
        _infer_max_trajectory_len(rows_train),
        _infer_max_trajectory_len(rows_predict),
    )
    X_train, feature_names = build_feature_matrix(
        rows_train, max_traj_len=max_len, include_meta=include_meta
    )
    X_pred, _ = build_feature_matrix(
        rows_predict, max_traj_len=max_len, include_meta=include_meta
    )
    y_train = np.asarray([1 if r["is_correct"] else 0 for r in rows_train], dtype=np.int32)
    groups_train = np.asarray([r["sample_index"] for r in rows_train], dtype=np.int32)

    result: MlAnalysisResult = {
        "mode": "labeled_train_unlabeled_predict",
        "n_train_samples": len({r["sample_index"] for r in rows_train}),
        "n_train_answers": len(rows_train),
        "train_positive_rate": float(y_train.mean()) if len(y_train) else None,
        "n_predict_samples": len({r["sample_index"] for r in rows_predict}),
        "n_predict_answers": len(rows_predict),
        "max_trajectory_len": max_len,
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "cv_n_splits": n_splits,
        "include_meta_features": include_meta,
        "train_cv": {},
        "predict_baselines": _baseline_metrics(rows_predict),
        "predict": {},
    }

    for model_name in model_names:
        oof_prob_train = _oof_group_cv_predict(
            X_train, y_train, groups_train, model_name, n_splits=n_splits
        )
        train_cv = _evaluate_model_on_rows(rows_train, oof_prob_train)
        best_thr_train = train_cv["best_threshold_on_this_set"]
        tune_thr = float(best_thr_train.get("threshold", 0.5))

        model = _make_model(model_name)
        model.fit(X_train, y_train)
        proba_pred = model.predict_proba(X_pred)
        pos_col = 1 if proba_pred.shape[1] > 1 else 0
        prob_pred = proba_pred[:, pos_col]

        predict_eval = _evaluate_model_on_rows(
            rows_predict, prob_pred, threshold=tune_thr
        )
        importances = _feature_importances(model, feature_names)

        result["train_cv"][model_name] = {
            "answer_level_oof": train_cv["answer_level"],
            "per_sample_argmax_oof": train_cv["per_sample_argmax"],
            "best_threshold_tuned_on_labeled": best_thr_train,
            "feature_importances_top20": importances[:20],
        }
        result["predict"][model_name] = {
            "threshold_from_labeled_cv": tune_thr,
            "per_sample_argmax": predict_eval["per_sample_argmax"],
            "per_sample_tuned_threshold": predict_eval.get("per_sample_tuned_threshold"),
            "answer_level": predict_eval["answer_level"],
            "best_threshold_if_tuned_on_unlabeled": predict_eval["best_threshold_on_this_set"],
        }

    return result


def _rows_without_meta(analysis: dict[str, Any]) -> list[RowDict]:
    try:
        from .probe_slope_history_consistency_analysis import iter_flat_answer_records
    except ImportError:
        from probe_slope_history_consistency_analysis import iter_flat_answer_records

    rows = iter_flat_answer_records(analysis)
    for row in rows:
        row.setdefault("majority_consistency", 1.0)
        row.setdefault("is_current_majority", False)
    return rows


def _feature_importances(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).ravel()
    else:
        return []
    order = np.argsort(-vals)
    return [
        {"feature": feature_names[i], "importance": float(vals[i])}
        for i in order
        if i < len(feature_names)
    ]


def _baseline_metrics(rows: list[RowDict]) -> dict[str, Any]:
    """Majority-only and argmax-slope sample-level baselines."""
    by_sample: dict[int, list[RowDict]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_index"]].append(row)

    maj_picked: list[RowDict] = []
    slope_picked: list[RowDict] = []
    for candidates in by_sample.values():
        maj = _majority_row(candidates)
        if maj is not None:
            maj_picked.append(maj)
        slope_picked.append(max(candidates, key=lambda r: r["slope"]))

    n_samples = len(by_sample)
    return {
        "current_majority": {
            **_sample_level_prf(maj_picked, n_samples),
            "n_samples": n_samples,
            "sample_coverage": len(maj_picked) / n_samples if n_samples else None,
            "accuracy": _sample_level_prf(maj_picked, n_samples).get("precision"),
        },
        "argmax_slope": {
            **_sample_level_prf(slope_picked, n_samples),
            "n_samples": n_samples,
            "sample_coverage": 1.0,
            "accuracy": _sample_level_prf(slope_picked, n_samples).get("precision"),
        },
    }


def analyze_from_files(
    analysis_path: str | Path,
    summary_path: str | Path | None = None,
    **kwargs: Any,
) -> MlAnalysisResult:
    analysis = load_answer_analysis(analysis_path)
    summaries = load_probe_summaries(summary_path) if summary_path else None
    return analyze_passrate_ml(analysis, summaries, **kwargs)


def analyze_labeled_train_unlabeled_from_files(
    labeled_analysis_path: str | Path,
    unlabeled_analysis_path: str | Path,
    summary_path: str | Path,
    **kwargs: Any,
) -> MlAnalysisResult:
    summaries = load_probe_summaries(summary_path)
    return analyze_labeled_train_unlabeled_predict(
        load_answer_analysis(labeled_analysis_path),
        load_answer_analysis(unlabeled_analysis_path),
        summaries,
        **kwargs,
    )


def save_ml_analysis(result: MlAnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_default)


def _fmt_prf(m: dict[str, Any]) -> str:
    p, r, f1 = m.get("precision"), m.get("recall"), m.get("f1")
    return (
        f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
        if p is not None and r is not None and f1 is not None
        else "P/R/F1=n/a"
    )


def print_ml_summary(result: MlAnalysisResult) -> None:
    mode = result.get("mode", "single_pool_cv")
    print(f"=== Passrate trajectory ML ({mode}) ===")
    print(
        f"traj_len={result.get('max_trajectory_len')}  n_features={result.get('n_features')}"
    )

    if mode == "labeled_train_unlabeled_predict":
        _print_train_predict_summary(result)
        return

    print(
        f"n_samples={result.get('n_samples')}  n_answers={result.get('n_answer_records')}  "
        f"positive_rate={result.get('positive_rate', 0):.4f}"
    )
    baselines = result.get("baselines") or {}
    print("\n--- Baselines (per-sample) ---")
    for name, m in baselines.items():
        cov = m.get("sample_coverage")
        cov_s = f"{cov:.4f}" if cov is not None else "n/a"
        print(f"  {name}: coverage={cov_s}  {_fmt_prf(m)}")

    for model_name, payload in (result.get("models") or {}).items():
        _print_single_model_block(model_name, payload, oof_label="OOF")


def _print_train_predict_summary(result: MlAnalysisResult) -> None:
    print(
        f"train: n_samples={result.get('n_train_samples')}  n_answers={result.get('n_train_answers')}  "
        f"pos_rate={result.get('train_positive_rate')}"
    )
    print(
        f"predict (unlabeled): n_samples={result.get('n_predict_samples')}  "
        f"n_answers={result.get('n_predict_answers')}"
    )
    print("\n--- Unlabeled baselines (per-sample) ---")
    for name, m in (result.get("predict_baselines") or {}).items():
        print(f"  {name}: {_fmt_prf(m)}")

    for model_name in result.get("train_cv", {}):
        train_payload = result["train_cv"][model_name]
        pred_payload = result.get("predict", {}).get(model_name, {})
        print(f"\n--- Model: {model_name} ---")
        print("  [labeled CV / train]")
        ans = train_payload.get("answer_level_oof") or {}
        print(
            f"    answer-level OOF: P={ans.get('precision', 0):.4f}  "
            f"R={ans.get('recall', 0):.4f}  F1={ans.get('f1', 0):.4f}  "
            f"auc={ans.get('roc_auc')}"
        )
        argmax = train_payload.get("per_sample_argmax_oof") or {}
        print(f"    per-sample argmax OOF: {_fmt_prf(argmax)}")
        bthr = train_payload.get("best_threshold_tuned_on_labeled") or {}
        print(f"    best thr on labeled (for deploy): {bthr.get('threshold', 'n/a')}")

        print("  [unlabeled predict — GT metrics only]")
        pans = pred_payload.get("answer_level") or {}
        print(
            f"    answer-level: P={pans.get('precision', 0):.4f}  "
            f"R={pans.get('recall', 0):.4f}  F1={pans.get('f1', 0):.4f}  "
            f"auc={pans.get('roc_auc')}"
        )
        pargmax = pred_payload.get("per_sample_argmax") or {}
        print(f"    per-sample argmax: {_fmt_prf(pargmax)}")
        ptuned = pred_payload.get("per_sample_tuned_threshold") or {}
        thr = pred_payload.get("threshold_from_labeled_cv", "n/a")
        print(f"    per-sample thr={thr} (from labeled): {_fmt_prf(ptuned)}")
        top = train_payload.get("feature_importances_top20") or []
        if top:
            print("  top features (full labeled fit):")
            for item in top[:6]:
                print(f"    {item['feature']}: {item['importance']:.4f}")


def _print_single_model_block(
    model_name: str, payload: dict[str, Any], *, oof_label: str
) -> None:
    print(f"\n--- Model: {model_name} ({oof_label}) ---")
    ans = payload.get("answer_level_oof") or {}
    print(
        f"  answer-level: acc={ans.get('accuracy', 0):.4f}  "
        f"P={ans.get('precision', 0):.4f}  R={ans.get('recall', 0):.4f}  "
        f"F1={ans.get('f1', 0):.4f}  "
        f"roc_auc={ans.get('roc_auc')}  ap={ans.get('average_precision')}"
    )
    argmax = payload.get("per_sample_argmax_oof") or {}
    print(f"  per-sample argmax prob: coverage={argmax.get('sample_coverage', 0):.4f}  {_fmt_prf(argmax)}")
    best = payload.get("per_sample_best_threshold_oof") or {}
    thr = payload.get("best_threshold") or {}
    print(
        f"  per-sample best thr={thr.get('threshold', 'n/a'):.3f} (maj fallback): "
        f"coverage={best.get('sample_coverage', 0):.4f}  {_fmt_prf(best)}"
    )
    bmaj = payload.get("best_threshold_require_majority") or {}
    print(
        f"  per-sample best thr={bmaj.get('threshold', 'n/a'):.3f} (require maj): "
        f"coverage={bmaj.get('sample_coverage', 0):.4f}  {_fmt_prf(bmaj)}"
    )
    top = payload.get("feature_importances_top20") or []
    if top:
        print("  top features:")
        for item in top[:8]:
            print(f"    {item['feature']}: {item['importance']:.4f}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ML on passrate trajectories to predict answer correctness.",
    )
    parser.add_argument(
        "analysis_path",
        type=Path,
        nargs="?",
        default=None,
        help="Single-pool mode: unlabeled_answer_analysis.json",
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        nargs="?",
        default=None,
        help="all_probes_summary.jsonl (for majority meta features)",
    )
    parser.add_argument(
        "--train-analysis",
        type=Path,
        default=None,
        help="labeled_answer_analysis.json (train on labeled, predict on unlabeled)",
    )
    parser.add_argument(
        "--predict-analysis",
        type=Path,
        default=None,
        help="unlabeled_answer_analysis.json (with --train-analysis)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: probe_ml_analysis.json or probe_ml_labeled_train.json)",
    )
    parser.add_argument(
        "--hist-len",
        type=int,
        default=8,
        help="Majority history length for meta features",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="GroupKFold splits (by sample index)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logistic,rf,gbdt",
        help="Comma-separated: logistic, rf, gbdt",
    )
    parser.add_argument(
        "--no-meta-features",
        action="store_true",
        help="Do not use is_current_majority / majority_consistency features",
    )
    return parser


def main(argv: list[str] | None = None) -> MlAnalysisResult:
    args = _build_argparser().parse_args(argv)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    ml_kw = dict(
        model_names=model_names,
        n_splits=args.cv_splits,
        hist_len=args.hist_len,
        include_meta=not args.no_meta_features,
    )

    if args.train_analysis is not None or args.predict_analysis is not None:
        if args.train_analysis is None or args.predict_analysis is None:
            raise SystemExit(
                "Labeled-train mode requires both --train-analysis and --predict-analysis."
            )
        base = args.train_analysis.parent
        summary_path = args.summary_path or (base / "all_probes_summary.jsonl")
        if not summary_path.is_file():
            raise SystemExit(f"Missing summary JSONL: {summary_path}")
        result = analyze_labeled_train_unlabeled_from_files(
            args.train_analysis,
            args.predict_analysis,
            summary_path,
            **ml_kw,
        )
        output = args.output or (base / "probe_ml_labeled_train.json")
    else:
        if args.analysis_path is None:
            raise SystemExit(
                "Provide analysis_path (single-pool CV) or "
                "--train-analysis + --predict-analysis."
            )
        summary_path = args.summary_path or (
            args.analysis_path.parent / "all_probes_summary.jsonl"
        )
        result = analyze_from_files(
            args.analysis_path,
            summary_path if summary_path.is_file() else None,
            **ml_kw,
        )
        output = args.output or (args.analysis_path.parent / "probe_ml_analysis.json")

    save_ml_analysis(result, output)
    print_ml_summary(result)
    print(f"Wrote -> {output}")
    return result


if __name__ == "__main__":
    main()
