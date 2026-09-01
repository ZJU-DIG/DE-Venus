"""Common training utilities for weakly-supervised learning trainers."""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any


def make_tracking_logger(config):
    """Build verl ``Tracking`` with optional file logging.

    Uses ``trainer.output_log_path`` when set; otherwise ``{log_dir}/tracking.log``
    from ``trainer.log_dir`` or ``trainer.default_local_dir``. Omits ``log_file`` if
    the installed ``Tracking`` class does not accept it.
    """
    from omegaconf import OmegaConf
    from verl.utils.tracking import Tracking

    kwargs = {
        "project_name": config.trainer.project_name,
        "experiment_name": config.trainer.experiment_name,
        "default_backend": config.trainer.logger,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    log_file = config.trainer.get("output_log_path")
    if not log_file:
        log_dir = config.trainer.get("log_dir") or config.trainer.get("default_local_dir")
        if log_dir:
            log_file = os.path.join(str(log_dir), "tracking.log")
    if log_file and "log_file" in inspect.signature(Tracking.__init__).parameters:
        kwargs["log_file"] = log_file
    return Tracking(**kwargs)


def log_val_summary(config, global_steps, output_dict, log_dir_key="log_dir"):
    """Write validation summary metrics to a log file.

    Args:
        config: Trainer config (provides log_dir).
        global_steps: Current global step.
        output_dict: dict of metric_name -> value to write.
        log_dir_key: config key for log directory (default: "log_dir").
    """
    log_dir = config.trainer.get(log_dir_key, None)
    if not log_dir:
        return
    log_path = os.path.join(log_dir, f"valid/global_step_{global_steps}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for key, value in output_dict.items():
            f.write(f"{key}: {value}\n")


def log_train_metrics(config, global_steps, metrics, log_dir_key="log_dir"):
    """Append training step metrics to a log file.

    Args:
        config: Trainer config (provides log_dir).
        global_steps: Current global step.
        metrics: dict of metric_name -> value.
        log_dir_key: config key for log directory (default: "log_dir").
    """
    log_dir = config.trainer.get(log_dir_key, None)
    if not log_dir:
        return
    log_path = os.path.join(log_dir, "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        f.write(f"step{global_steps}: {metrics_str}\n")


def extract_core_val_metrics(metric_dict: dict) -> dict:
    """Extract core validation metrics from the full metric dict.

    Parses keys matching 'val-core/{data_source}/{var_name}/{metric_name}'
    where metric_name contains 'mean@', and returns a simplified mapping
    '{data_source}/{metric_name}' -> value along with average score.

    Args:
        metric_dict: Full validation metric dict from super()._validate().

    Returns:
        output_dict: dict with per-source metrics and 'average' key.
    """
    core_metric_dict = {}
    for key, val in metric_dict.items():
        if key.startswith("val-core/") and "mean@" in key:
            parts = key.split("/")
            data_source, metric_name = parts[1], parts[3]
            core_metric_dict[f"{data_source}/{metric_name}"] = val

    if core_metric_dict:
        avg_score = sum(core_metric_dict.values()) / len(core_metric_dict)
    else:
        avg_score = 0.0

    return {**core_metric_dict, "average": avg_score}


_VAL_STEP_RE = re.compile(r"global_step_(\d+)\.log$")


def read_val_log_file(path: str | Path) -> dict[str, float]:
    """Parse ``valid/global_step_*.log`` written by ``log_val_summary``."""
    metrics: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metrics[key.strip()] = float(value.strip())
    return metrics


def read_latest_val_average(log_dir: str | Path) -> dict[str, Any] | None:
    """Return the latest validation summary (``average`` = mean over val datasets).

    Reads files under ``{log_dir}/valid/global_step_*.log`` produced by
    ``validate_and_log`` / ``log_val_summary``.
    """
    valid_dir = Path(log_dir) / "valid"
    if not valid_dir.is_dir():
        return None

    best_step = -1
    best_path: Path | None = None
    for path in valid_dir.glob("global_step_*.log"):
        match = _VAL_STEP_RE.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step >= best_step:
            best_step = step
            best_path = path

    if best_path is None:
        return None

    metrics = read_val_log_file(best_path)
    if "average" not in metrics:
        return None

    per_source = {k: v for k, v in metrics.items() if k != "average"}
    return {
        "global_step": best_step,
        "average": float(metrics["average"]),
        "per_source": per_source,
        "log_path": str(best_path),
    }


def validate_and_log(config, global_steps, metric_dict):
    """Run the common post-validation logic: extract core metrics and write log.

    This replaces the identical _validate override in TTRL/Intuitor/EMRL/VMF trainers.

    Args:
        config: Trainer config (provides log_dir).
        global_steps: Current global step.
        metric_dict: Full validation metric dict returned by super()._validate().

    Returns:
        metric_dict: The original metric_dict (unchanged).
    """
    output_dict = extract_core_val_metrics(metric_dict)
    log_val_summary(config, global_steps, output_dict)
    return metric_dict


def compute_semi_metrics(rel_pseudo_num, rel_pseudo_correct_num,
                         unrel_pseudo_correct_num, unrel_pseudo_num=None,
                         prefix="semi"):
    """Compute pseudo-label precision/recall/f1 metrics.

    Args:
        rel_pseudo_num: Number of reliable pseudo-labeled uids.
        rel_pseudo_correct_num: Number of reliable pseudo-labeled uids that are correct.
        unrel_pseudo_correct_num: Number of unreliable uids that would have been correct.
        unrel_pseudo_num: Number of unreliable pseudo-labeled uids (optional, adds extra metrics).
        prefix: Metric key prefix (default: "semi").

    Returns:
        dict with keys: {prefix}/rel_pseudo_num, {prefix}/rel_pseudo_correct_num,
        {prefix}/rel_pseudo_precision, {prefix}/rel_pseudo_recall, {prefix}/rel_pseudo_f1.
        If unrel_pseudo_num is provided, also includes:
        {prefix}/unrel_pseudo_num, {prefix}/unrel_pseudo_correct_num,
        {prefix}/total_pseudo_correct_num.
    """
    total_correct = rel_pseudo_correct_num + unrel_pseudo_correct_num
    precision = rel_pseudo_correct_num / rel_pseudo_num if rel_pseudo_num > 0 else 0
    recall = rel_pseudo_correct_num / total_correct if total_correct > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        f"{prefix}/rel_pseudo_num": rel_pseudo_num,
        f"{prefix}/rel_pseudo_correct_num": rel_pseudo_correct_num,
        f"{prefix}/rel_pseudo_precision": precision,
        f"{prefix}/rel_pseudo_recall": recall,
        f"{prefix}/rel_pseudo_f1": f1,
    }

    if unrel_pseudo_num is not None:
        result[f"{prefix}/unrel_pseudo_num"] = unrel_pseudo_num
        result[f"{prefix}/unrel_pseudo_correct_num"] = unrel_pseudo_correct_num
        result[f"{prefix}/total_pseudo_correct_num"] = total_correct

    return result