"""Common data utilities for weakly-supervised learning trainers."""

import logging
from collections import defaultdict

import torch
from verl import DataProto

logger = logging.getLogger(__name__)


def is_labeled_extra_info(extra_info) -> bool:
    """Return True if ``extra_info`` marks a labeled (supervised GT) sample."""
    if extra_info is None:
        return False
    if isinstance(extra_info, dict):
        labeled = extra_info.get("labeled", False)
        return labeled is True or labeled == 1 or labeled == "true"
    return False


def collect_labeled_dataset_indices(dataset) -> list[int]:
    """Collect row indices in an RLHF dataset where ``extra_info['labeled']`` is True.

    Args:
        dataset: Verl ``RLHFDataset`` (or compatible) with a ``dataframe`` attribute.

    Returns:
        Sorted list of dataset indices for labeled prompts.
    """
    if not hasattr(dataset, "dataframe"):
        raise TypeError(
            "collect_labeled_dataset_indices expects a dataset with a .dataframe attribute "
            f"(got {type(dataset).__name__})"
        )

    hf_ds = dataset.dataframe
    labeled_indices: list[int] = []
    for i in range(len(hf_ds)):
        row = hf_ds[i]
        extra_info = row.get("extra_info") if isinstance(row, dict) else None
        if is_labeled_extra_info(extra_info):
            labeled_indices.append(i)

    logger.info(
        "collect_labeled_dataset_indices: %d labeled / %d total (%.1f%%)",
        len(labeled_indices),
        len(hf_ds),
        100.0 * len(labeled_indices) / max(len(hf_ds), 1),
    )
    return labeled_indices


def get_labeled_unlabeled_indices(data: DataProto):
    """Split batch indices into labeled and unlabeled groups by uid.

    Args:
        data: DataProto batch with non_tensor_batch["extra_info"] and non_tensor_batch["uid"].

    Returns:
        labeled_uid2idx: dict mapping uid -> list of indices for labeled samples.
        unlabeled_uid2idx: dict mapping uid -> list of indices for unlabeled samples.
    """
    labeled_uid2idx = defaultdict(list)
    unlabeled_uid2idx = defaultdict(list)

    for i in range(len(data)):
        data_item = data[i]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        assert extra_info and extra_info["labeled"] in (True, False)
        uid = data_item.non_tensor_batch.get("uid")
        if extra_info["labeled"] is True:
            labeled_uid2idx[uid].append(i)
        else:
            unlabeled_uid2idx[uid].append(i)

    return labeled_uid2idx, unlabeled_uid2idx


def uid2idx_to_flat_list(uid2idx: dict) -> list:
    """Flatten a uid2idx dict into a sorted flat list of indices.

    Args:
        uid2idx: dict mapping uid -> list of indices.

    Returns:
        Flat sorted list of all indices.

    Example:
        >>> uid2idx_to_flat_list({"a": [0, 3], "b": [1, 2]})
        [0, 1, 2, 3]
    """
    flat = []
    for indices in uid2idx.values():
        flat.extend(indices)
    flat.sort()
    return flat


def extract_response_strings(data: DataProto, indices: list, tokenizer) -> list:
    """Decode valid response token IDs into strings for the given indices.

    Args:
        data: DataProto batch with batch["prompts"], batch["responses"], batch["attention_mask"].
        indices: list of sample indices to extract.
        tokenizer: huggingface tokenizer for decoding.

    Returns:
        list of decoded response strings.
    """
    responses = []
    for idx in indices:
        data_item = data[idx]
        prompt_length = data_item.batch["prompts"].shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        responses.append(response_str)
    return responses