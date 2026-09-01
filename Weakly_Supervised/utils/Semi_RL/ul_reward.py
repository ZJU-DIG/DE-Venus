"""Unlabeled reward computation utilities for semi-supervised RL trainers.

Provides:
- compute_klu_seq_reward: KL(U || π_θ) reward (Intuitor)
- compute_entropy_seq_reward: entropy-based reward (EMRL)
- apply_ul_seq_reward: write sequence-level rewards into token-level reward tensor
"""

import torch


def compute_klu_seq_reward(data, ul_indices, vocab_size, device):
    """Compute sequence-level KL(U || π_θ) reward for unlabeled samples.

    For each token position t:
        KL(U || π_θ(·|y_{<t})) = uniform_ce_t - log(V)
    The sequence reward is the mean KL over valid response tokens.

    Args:
        data: DataProto with batch["response_mask"] and batch["uniform_ces"].
        ul_indices: sorted flat list of unlabeled sample indices.
        vocab_size: tokenizer vocab size.
        device: torch device.

    Returns:
        seq_reward: [U] tensor of sequence-level rewards.
    """
    response_mask = data.batch["response_mask"].to(device)
    uniform_ces = data.batch["uniform_ces"].to(device)

    idx_tensor = torch.tensor(ul_indices, dtype=torch.long, device=device)
    mask_ul = response_mask[idx_tensor].float()
    uce_ul = uniform_ces[idx_tensor] * mask_ul

    log_V = torch.tensor(vocab_size, dtype=uce_ul.dtype, device=device).log()
    kl_per_token = uce_ul - log_V * mask_ul

    valid_lengths = mask_ul.sum(dim=-1).clamp(min=1)
    seq_reward = kl_per_token.sum(dim=-1) / valid_lengths
    return seq_reward


def compute_entropy_seq_reward(data, ul_indices, entropys, ul_reward_type="tok-entropy", device=None):
    """Compute sequence-level entropy-based reward for unlabeled samples.

    Supports two modes:
    - "tok-entropy": reward = -sum_t H(π_θ(·|y_{<t}))  (token-level entropy)
    - "seq-entropy": reward = sum_t log π_θ(y_t|y_{<t})  (sequence-level log-prob)

    Args:
        data: DataProto with batch["response_mask"] and batch["old_log_probs"].
        ul_indices: sorted flat list of unlabeled sample indices.
        entropys: [bs, response_length] per-token entropy tensor.
        ul_reward_type: "tok-entropy" or "seq-entropy".
        device: torch device (inferred from entropys if None).

    Returns:
        seq_reward: [U] tensor of sequence-level rewards.
    """
    assert ul_reward_type in ("tok-entropy", "seq-entropy"), (
        f"Unsupported ul_reward_type: {ul_reward_type}"
    )
    if device is None:
        device = entropys.device

    response_mask = data.batch["response_mask"].to(device)
    old_log_probs = data.batch["old_log_probs"].to(device)
    entropys = entropys.to(device)

    idx_tensor = torch.tensor(ul_indices, dtype=torch.long, device=device)
    mask_ul = response_mask[idx_tensor].float()
    logp_ul = old_log_probs[idx_tensor] * mask_ul
    entropy_ul = entropys[idx_tensor] * mask_ul

    if ul_reward_type == "tok-entropy":
        seq_reward = -(entropy_ul.sum(dim=-1))
    else:
        seq_reward = logp_ul.sum(dim=-1)
    return seq_reward


def apply_ul_seq_reward(data, reward_tensor, ul_indices, seq_rewards):
    """Write sequence-level rewards for unlabeled samples into the token-level reward tensor.

    Clears existing rewards for unlabeled rows and writes seq_rewards at the last
    valid response token position.

    Args:
        data: DataProto batch with batch["response_mask"].
        reward_tensor: [bs, response_length] token-level reward tensor (modified in-place).
        ul_indices: sorted flat list of unlabeled sample indices.
        seq_rewards: [U] sequence-level reward tensor for unlabeled samples,
                     order must match ul_indices.

    Returns:
        The (in-place modified) reward_tensor.
    """
    if not ul_indices:
        return reward_tensor

    idx_tensor = torch.tensor(ul_indices, dtype=torch.long, device=reward_tensor.device)
    response_mask = data.batch["response_mask"].to(reward_tensor.device)

    # Clear existing reward for unlabeled rows
    reward_tensor[idx_tensor] = 0.0

    # Determine last valid token per unlabeled row
    mask_ul = response_mask[idx_tensor].float()
    valid_lengths = mask_ul.sum(dim=-1).long()
    last_idx = torch.clamp(valid_lengths - 1, min=0)
    has_valid = valid_lengths > 0

    if has_valid.any():
        rows = idx_tensor[has_valid]
        cols = last_idx[has_valid]
        reward_tensor[rows, cols] = seq_rewards[has_valid].to(reward_tensor.dtype)

    return reward_tensor