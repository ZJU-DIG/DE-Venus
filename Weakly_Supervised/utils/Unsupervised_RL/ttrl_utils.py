# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from collections import Counter, defaultdict
import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None", 0.0
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    return majority_answer, majority_ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric


# =============================================================================
# Unsupervised RLVR Extensions
# =============================================================================


def _get_ground_truth_scores_array(batch):
    """Per-sample ground-truth scores for metric computation."""
    for key in ("score", "acc", "reward"):
        if key in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch[key], dtype=np.float64).reshape(-1)
    if "token_level_scores_original" in batch.batch:
        return (
            batch.batch["token_level_scores_original"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        )
    raise KeyError(
        "Ground-truth scores not found. Expected non_tensor_batch['score'|'acc'|'reward'] "
        f"or batch['token_level_scores_original']. Keys: {list(batch.non_tensor_batch.keys())}"
    )


def as_plain_tensor(t) -> torch.Tensor:
    """Convert TensorDict / dict wrappers to a plain torch.Tensor."""
    if isinstance(t, dict):
        if "reward_tensor" in t:
            t = t["reward_tensor"]
        elif len(t) == 1:
            t = next(iter(t.values()))
    try:
        from tensordict import TensorDict

        if isinstance(t, TensorDict):
            if len(t.keys()) == 1:
                t = t[t.keys()[0]]
            else:
                for key in ("rm_scores", "token_level_scores", "token_level_rewards", "responses"):
                    if key in t.keys():
                        t = t[key]
                        break
    except ImportError:
        pass
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.contiguous()


def normalize_token_level_reward_tensor(
    reward_tensor: torch.Tensor, response_mask: torch.Tensor, responses: torch.Tensor
) -> torch.Tensor:
    """Ensure token-level rewards have shape (batch_size, response_length) for GRPO."""
    reward_tensor = as_plain_tensor(reward_tensor)
    response_mask = as_plain_tensor(response_mask)
    responses = as_plain_tensor(responses)

    if responses.dim() == 1:
        responses = responses.unsqueeze(0)
    if response_mask.dim() == 1:
        response_mask = response_mask.unsqueeze(0)

    batch_size, response_length = responses.shape[0], responses.shape[1]
    device = responses.device
    dtype = reward_tensor.dtype if reward_tensor.is_floating_point() else torch.float32

    if reward_tensor.dim() == 2 and reward_tensor.shape[0] == batch_size:
        if reward_tensor.shape[1] == response_length:
            return reward_tensor.to(dtype=dtype, device=device)
        if reward_tensor.shape[1] == 1:
            out = torch.zeros(batch_size, response_length, dtype=dtype, device=device)
            eos_indices = response_mask.sum(dim=-1).long().clamp(min=1) - 1
            out.scatter_(1, eos_indices.unsqueeze(1), reward_tensor.reshape(batch_size, 1).to(dtype))
            return out

    per_sample = None
    if reward_tensor.dim() == 0:
        per_sample = reward_tensor.reshape(1).expand(batch_size)
    elif reward_tensor.dim() == 1:
        if reward_tensor.shape[0] == batch_size:
            per_sample = reward_tensor.reshape(batch_size)
        elif reward_tensor.shape[0] == response_length and batch_size == 1:
            out = reward_tensor.reshape(1, response_length).to(dtype=dtype, device=device)
            return out
        elif reward_tensor.numel() == 1:
            per_sample = reward_tensor.reshape(1).expand(batch_size)

    if per_sample is None:
        raise ValueError(
            f"Cannot normalize reward_tensor shape {reward_tensor.shape} for "
            f"batch_size={batch_size}, response_length={response_length}"
        )

    out = torch.zeros(batch_size, response_length, dtype=dtype, device=device)
    eos_indices = response_mask.sum(dim=-1).long().clamp(min=1) - 1
    out.scatter_(1, eos_indices.unsqueeze(1), per_sample.to(dtype).reshape(batch_size, 1))
    return out


def _get_proxy_scores_array(batch, key: str = "pseudo_score"):
    """Per-sample proxy / pseudo reward scores for metric computation."""
    if key not in batch.non_tensor_batch:
        raise KeyError(
            f"Missing {key} in non_tensor_batch. Available keys: {list(batch.non_tensor_batch.keys())}"
        )
    val = batch.non_tensor_batch[key]
    if torch.is_tensor(val):
        return val.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(val, dtype=np.float64).reshape(-1)


def apply_hybrid_gt(batch, reward_extra_infos_dict, gen_batch_output, n, tokenizer):
    """
    Apply the hybrid ground truth to the batch.
    Only replaces GT with majority vote when no response in the group is correct
    (according to the reward function), but at least one response has correct format.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    # Get the majority vote ground truth
    model_outputs = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)

    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"

    # Apply the hybrid ground truth
    for i in range(num_prompts):
        data_item = batch[i]

        if "acc" in reward_extra_infos_dict:
            acc_list = reward_extra_infos_dict["acc"][n * i : n * (i + 1)]
        else:
            acc_list = reward_extra_infos_dict["score"][n * i : n * (i + 1)]
        if "format" in reward_extra_infos_dict:
            format_list = reward_extra_infos_dict["format"][n * i : n * (i + 1)]
        else:
            format_list = [1.0] * n

        if True in acc_list:
            continue
        elif sum(format_list) > 0:
            original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
            data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
            data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


# === Certainty-Based Reward ===


def compute_certainty_metrics(batch, n):
    """
    Calculates the point-biserial correlation and a custom label accuracy metric
    for certainty-based unsupervised rewards.

    Args:
        batch: An object containing the model's outputs with 'non_tensor_batch'
               containing "pseudo_score" and "score".
        n (int): The number of samples per prompt.
    """
    from scipy.stats import pointbiserialr

    proxy_rewards = _get_proxy_scores_array(batch, key="pseudo_score")
    ground_truth_rewards = _get_ground_truth_scores_array(batch)

    # 1. Calculate the point-biserial correlation coefficient
    if len(np.unique(ground_truth_rewards)) > 1:
        corr, _ = pointbiserialr(ground_truth_rewards, proxy_rewards)
    else:
        corr = float(0)

    # 2. Reshape the data to (number_of_prompts, n)
    assert len(proxy_rewards) % n == 0, "proxy_rewards length must be divisible by n"
    num_prompts = len(proxy_rewards) // n
    proxy_rewards_reshaped = proxy_rewards.reshape(num_prompts, n)
    ground_truth_rewards_reshaped = ground_truth_rewards.reshape(num_prompts, n)

    # 3. For each prompt, find the index of the response with the largest proxy reward.
    selected_indices = np.argmax(proxy_rewards_reshaped, axis=1)

    # 4. Select the ground truth reward of the chosen response for each prompt.
    rewards_of_selected_responses = ground_truth_rewards_reshaped[np.arange(num_prompts), selected_indices]

    # 5. Calculate the proportion (reward accuracy).
    correctly_identified_count = np.sum(rewards_of_selected_responses)
    label_acc = correctly_identified_count / num_prompts

    return {
        "point_biserial_correlation": corr,
        "pseudo_label_acc": label_acc,
        "train_acc": batch.batch["token_level_scores_original"].sum(dim=-1).float().mean().item(),
    }


def compute_certainty_reward(data, reward_type):
    """
    Compute reward based on model's certainty metrics.

    Args:
        data: DataProto containing model outputs with entropys, self_certaintys, old_log_probs.
        reward_type: One of "self_certainty", "token_level_entropy",
                     "trajectory_level_entropy", "probability".

    Returns:
        reward_tensor, reward_extra_info dict
    """
    reward_extra_info = defaultdict(list)
    response_mask = data.batch["response_mask"]
    responses = data.batch["responses"]
    if response_mask.dim() == 1:
        response_mask = response_mask.unsqueeze(0)
    if responses.dim() == 1:
        responses = responses.unsqueeze(0)
    from verl.utils.torch_functional import masked_mean, masked_sum

    if reward_type == "self_certainty":
        token_scores = data.batch.get("self_certaintys")
        if token_scores is None:
            # verl does not populate self_certaintys yet; use negative entropy as proxy
            entropys = data.batch.get("entropys")
            token_scores = -entropys if entropys is not None else torch.zeros_like(response_mask)
        scores = masked_mean(token_scores, response_mask, axis=-1).reshape(responses.shape[0])
    elif reward_type == "token_level_entropy":
        token_scores = -data.batch.get("entropys", torch.zeros_like(response_mask, device=response_mask.device))
        scores = masked_mean(token_scores, response_mask, axis=-1).reshape(responses.shape[0])
    elif reward_type == "trajectory_level_entropy":
        token_scores = data.batch.get("old_log_probs", torch.zeros_like(response_mask, device=response_mask.device))
        scores = masked_mean(token_scores, response_mask, axis=-1).reshape(responses.shape[0])
    elif reward_type == "probability":
        log_probs = data.batch.get("old_log_probs", torch.zeros_like(response_mask, device=response_mask.device))
        sentence_scores = masked_sum(log_probs, response_mask, axis=-1)  # (batch_size,)
        scores = torch.exp(sentence_scores).reshape(responses.shape[0])
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    batch_size, response_length = responses.shape[0], responses.shape[1]
    reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32, device=responses.device)
    eos_indices = response_mask.sum(dim=-1).long().clamp(min=1) - 1
    reward_tensor.scatter_(1, eos_indices.unsqueeze(1), scores.to(reward_tensor.dtype).unsqueeze(1))

    # save the pseudo scores for later metric computation (list for stable np conversion)
    reward_extra_info["pseudo_score"] = scores.detach().cpu().tolist()

    return reward_tensor, reward_extra_info


# === Self-Verify ===


def compute_self_verify_metrics(batch):
    """Compute metrics for self-verify based reward."""
    proxy_rewards = _get_proxy_scores_array(batch, key="verification_score")
    ground_truth_rewards = _get_ground_truth_scores_array(batch)

    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(proxy_rewards, ground_truth_rewards):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(proxy_rewards)

    return {
        "reward_accuracy": rewards_hit_rate,
        "self_verify_reward": sum(proxy_rewards) / len(proxy_rewards),
        "ground_truth_reward": sum(ground_truth_rewards) / len(ground_truth_rewards),
    }


def apply_self_verify(
    batch,
    tokenizer,
    actor_rollout_wg,
    verify_prompt=None,
    async_rollout_mode: bool = False,
    async_rollout_manager=None,
    max_prompt_length: int = None,
):
    """
    Apply self-verify ground truth to the batch using actor model for verification.
    Returns:
        reward_tensor, reward_extra_infos_dict
    """
    verify_prompt = '''You are given a question and its proposed solution. Your task is to EVALUATE whether the solution is correct.

Follow these steps carefully:
1. The expression in the solution only contains numbers that appear in the question.
2. Every number that appears in the question is used exactly once in the solution.
3. The solution is a valid arithmetic expression (not an equation).
4. The solution evaluates to the target value specified in the question.
5. At the end, output ONLY one of the following with your explanation:
- \\boxed{{True}}  (if the solution is correct)
- \\boxed{{False}} (if the solution is incorrect)

Question:
[{}]

Solution:
[{}]

Result:    
'''
    reward_tensor, reward_extra_infos_dict = _compute_self_verify_rewards(
        batch,
        tokenizer,
        actor_rollout_wg,
        verify_prompt,
        num_examine=5,
        reward_fn_key="data_source",
        async_rollout_mode=async_rollout_mode,
        async_rollout_manager=async_rollout_manager,
        max_prompt_length=max_prompt_length,
    )
    return reward_tensor, reward_extra_infos_dict


def _compute_self_verify_rewards(
    data,
    tokenizer,
    actor_rollout_wg,
    verify_prompt,
    num_examine=5,
    reward_fn_key="data_source",
    async_rollout_mode: bool = False,
    async_rollout_manager=None,
    max_prompt_length: int = None,
):
    """
    Compute self-verify rewards using actor model for verification.

    Args:
        data: DataProto containing prompts and responses
        tokenizer: Tokenizer for text processing
        actor_rollout_wg: Actor rollout worker group for self-verification
        verify_prompt: Template string for verification prompts
        num_examine: Number of samples to print for debugging
        reward_fn_key: Key for accessing data source

    Returns:
        tuple: (reward_tensor, reward_extra_infos_dict)
    """
    responses = as_plain_tensor(data.batch["responses"])
    reward_tensor = torch.zeros(responses.shape[0], responses.shape[1], dtype=torch.float32, device=responses.device)
    reward_extra_info = defaultdict(list)

    # Collect questions and solutions for batch processing
    questions = []
    solutions = []
    item_indices = []

    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]
        # decode
        prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(response_ids, skip_special_tokens=True)

        question, solution = _extract_question_and_solution(prompt_str, response_str)
        questions.append(question)
        solutions.append(solution)
        item_indices.append(i)

    # Batch verification using actor
    if questions:
        verification_batch = _create_verification_batch(
            questions,
            solutions,
            tokenizer,
            verify_prompt,
            for_agent_loop=async_rollout_mode,
            max_prompt_length=max_prompt_length,
        )
        verification_batch.meta_info.update(
            {
                "validate": True,
                "temperature": 0.5,
                "do_sample": True,
            }
        )
        # Generate verification responses using actor
        if not async_rollout_mode:
            verification_output = actor_rollout_wg.generate_sequences(verification_batch)
        else:
            assert async_rollout_manager is not None, "async_rollout_manager is required when async_rollout_mode=True"
            verification_output = async_rollout_manager.generate_sequences(verification_batch)

        # Parse verification responses (agent loop / vLLM output layout)
        verification_responses = verification_output.batch["responses"]
        verify_prompt_length = verification_output.batch["prompts"].shape[-1]
        for i, (item_idx, verification_response_ids) in enumerate(zip(item_indices, verification_responses)):
            verify_attention = verification_output.batch["attention_mask"][i]
            valid_verification_length = int(verify_attention[verify_prompt_length:].sum().item())
            valid_verification_ids = verification_response_ids[:valid_verification_length]
            verification_text = tokenizer.decode(valid_verification_ids, skip_special_tokens=True)

            # Parse score
            score = _parse_verification_response(verification_text)

            # Get original data item for logging
            data_item = data[item_idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            eos_idx = max(valid_response_length - 1, 0)

            # Store reward at EOS of the original rollout response
            reward_tensor[item_idx, eos_idx] = score

            # Store extra info
            reward_extra_info["verification_response"].append(verification_text)
            reward_extra_info["verification_score"].append(score)

    return reward_tensor, reward_extra_info


def _extract_question_and_solution(prompt_str: str, response_str: str) -> tuple[str, str]:
    """Extract user question and assistant solution from decoded chat strings."""
    markers = [
        ("user\n", "assistant\n"),
        ("<|im_start|>user", "<|im_start|>assistant"),
        ("<|user|>", "<|assistant|>"),
    ]
    for user_tag, asst_tag in markers:
        if user_tag in prompt_str and asst_tag in prompt_str:
            q_start = prompt_str.find(user_tag) + len(user_tag)
            q_end = prompt_str.find(asst_tag)
            return prompt_str[q_start:q_end].strip(), response_str.strip()
    return prompt_str.strip(), response_str.strip()


def _create_verification_batch(
    questions, solutions, tokenizer, prompt, for_agent_loop: bool = True, max_prompt_length: int = None
):
    """Create a DataProto batch of verification prompts.

    vLLM async agent loop requires ``raw_prompt`` in ``non_tensor_batch`` (see RLHFDataset).
    Legacy sync HF rollout expects tokenized ``input_ids`` / ``attention_mask`` / ``position_ids``.

    When ``max_prompt_length`` is set (required for agent loop mode), solutions are token-truncated
    so that the full verification prompt stays within ``max_prompt_length`` tokens.  Without this,
    long Q+A pairs produce prompt_ids of varying lengths that make ``_postprocess``'s ``torch.cat``
    fail with ``RuntimeError: Sizes of tensors must match``.
    """
    verification_messages = []
    for q, s in zip(questions, solutions):
        q_str = str(q)
        s_str = str(s)

        if max_prompt_length is not None:
            q_esc = q_str.replace("{", "{{").replace("}", "}}")
            s_esc = s_str.replace("{", "{{").replace("}", "}}")
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.format(q_esc, s_esc)}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(full_ids) > max_prompt_length:
                # Compute template+question overhead to find the solution token budget.
                overhead_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.format(q_esc, "")}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                # Initial estimate: leave 50 tokens of breathing room for subword
                # boundary artifacts that arise when decode→re-encode produces slightly
                # more tokens than the estimate. 10 tokens was too tight (2059 > 2048).
                sol_budget = max(max_prompt_length - len(overhead_ids) - 50, 20)
                sol_token_ids = tokenizer.encode(s_str)[:sol_budget]
                s_str = tokenizer.decode(sol_token_ids, skip_special_tokens=True)

                # Tight validation loop: re-tokenize and keep shrinking by 10% until
                # the assembled prompt provably fits within max_prompt_length.
                s_esc_check = s_str.replace("{", "{{").replace("}", "}}")
                check_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.format(q_esc, s_esc_check)}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                while len(check_ids) > max_prompt_length and len(s_str) > 20:
                    cur_sol_ids = tokenizer.encode(s_str)
                    keep = max(int(len(cur_sol_ids) * 0.9), 10)
                    s_str = tokenizer.decode(cur_sol_ids[:keep], skip_special_tokens=True)
                    s_esc_check = s_str.replace("{", "{{").replace("}", "}}")
                    check_ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt.format(q_esc, s_esc_check)}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )

        q_escaped = q_str.replace("{", "{{").replace("}", "}}")
        s_escaped = s_str.replace("{", "{{").replace("}", "}}")
        verification_messages.append([{"role": "user", "content": prompt.format(q_escaped, s_escaped)}])

    batch_size = len(verification_messages)
    if batch_size == 0:
        return DataProto(batch=TensorDict({}, batch_size=[0]))

    if for_agent_loop:
        non_tensor_batch = {
            "raw_prompt": np.array(verification_messages, dtype=object),
            # Stubs so optional reward-loop scoring during verify gen does not crash.
            "data_source": np.array(["self_verify"] * batch_size, dtype=object),
            "reward_model": np.array(
                [{"style": "rule", "ground_truth": ""} for _ in range(batch_size)], dtype=object
            ),
        }
        batch = TensorDict(
            {"dummy_tensor": torch.zeros(batch_size, dtype=torch.uint8)},
            batch_size=[batch_size],
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"validate": True})

    formatted_prompts = []
    for msg in verification_messages:
        formatted_prompts.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        )
    tokenizer.padding_side = "left"
    tokenized = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    position_ids = (tokenized["attention_mask"].cumsum(dim=1) - 1) * tokenized["attention_mask"]
    batch = TensorDict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "position_ids": position_ids,
        },
        batch_size=[batch_size],
    )
    return DataProto(batch=batch)


def _parse_verification_response(response):
    """
    Parse the verification response to get a score.

    Args:
        response: Verification response text

    Returns:
        float: 1.0 if verified as correct, 0.0 otherwise
    """
    response = response.strip().lower()

    # Look for boxed answers
    if "\\boxed{true}" in response or "\\boxed{true" in response:
        return 1.0
    elif "\\boxed{false}" in response or "\\boxed{false" in response:
        return 0.0
    # Fallback to keyword matching
    elif "true" in response and "false" not in response:
        return 1.0
    elif "false" in response and "true" not in response:
        return 0.0
    else:
        return 0.0