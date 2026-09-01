"""Compute uncertainty metrics for each question from rollout responses.

Designed for multi-GPU parallelism: each process loads the model on one GPU
and scores a 1/n_gpus slice of the data. A shell script launches all processes
and merges the per-split outputs.

Prerequisites:
  - responses.jsonl from generate_response.py
  - global_attn_heads.pkl from attn_probing.py

Computed metrics:
  - pivot_num:     number of attention pivot peaks (from global attention heads)
  - consistency:   answer agreement across rollouts, max_freq / n (math only)
  - entropy:       mean token-level entropy of output distribution
  - self_certainty: mean KL(uniform || p) across output tokens
  - coe_r:         cumulative per-layer magnitude/angle change ratio
  - coe_c:         resultant length of per-layer polar (mag, angle) vectors
  - cot_kinetics:  normalized speed + curvature across layers
"""

import os
import json
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_long_range_attn(attn_map, H_low, H_high):
    """Compute per-token long-range attention signal.

    For token k, averages attention received from tokens in [k+H_low, k+H_high).
    """
    seq_len = attn_map.shape[0]
    valid_len = max(0, seq_len - H_low)
    if valid_len == 0:
        return np.array([])

    cumsum = np.cumsum(attn_map, axis=0)

    signal = np.empty(valid_len, dtype=np.float64)
    for k in range(valid_len):
        start = k + H_low
        end = min(k + H_high, seq_len)
        col_sum = cumsum[end - 1, k]
        if start > 0:
            col_sum -= cumsum[start - 1, k]
        signal[k] = col_sum / (end - start)

    return signal


def compute_consistency(responses, golden_answer):
    """Compute answer consistency across rollouts.

    Clusters rollout answers via math_verify parse/verify and returns
    max_cluster_size / n_rollouts.
    """
    from math_verify import parse, verify

    predict_answers = list(map(parse, responses))

    freq_list = []
    for answer in predict_answers:
        found = False
        for k, (exist_answer, count) in enumerate(freq_list):
            try:
                if exist_answer[1] == answer[1] or verify(exist_answer, answer):
                    freq_list[k] = (exist_answer, count + 1)
                    found = True
                    break
            except Exception:
                pass
        if not found:
            freq_list.append((answer, 1))

    counts = [count for _, count in freq_list]
    max_count = max(counts)
    consistency = max_count / len(responses)

    return consistency


def compute_hidden_state_metrics(hidden_states, prompt_len):
    """Compute coe_r, coe_c, cot_kinetics from per-layer hidden states.

    Only response tokens (after prompt_len) are used.

    Returns:
        (coe_r, coe_c, cot_kinetics)
    """
    eps = 1e-8
    layer_num = len(hidden_states)

    # Extract response-only hidden states and compute per-layer mean representations
    avg_layer_features = []
    for hid in hidden_states:
        feat = hid.squeeze(0)[prompt_len:, :].float()
        avg_layer_features.append(feat.mean(dim=0))

    # --- coe_r and coe_c ---
    # Compare per-layer changes against the total (layer_0 -> layer_L) change
    feature0 = hidden_states[0].squeeze(0)[prompt_len:, :].float()
    featureL = hidden_states[-1].squeeze(0)[prompt_len:, :].float()

    diff_init = featureL - feature0
    mag_init = torch.norm(diff_init, dim=-1).mean()

    feat0_unit = feature0 / (torch.norm(feature0, dim=-1, keepdim=True) + eps)
    featL_unit = featureL / (torch.norm(featureL, dim=-1, keepdim=True) + eps)
    cos_sim_init = torch.clamp(torch.sum(feat0_unit * featL_unit, dim=-1), -1.0, 1.0)
    ang_init = torch.acos(cos_sim_init).mean()

    coe_r = 0.0
    cos_sum = 0.0
    sin_sum = 0.0
    for l in range(layer_num - 1):
        feat = hidden_states[l].squeeze(0)[prompt_len:, :].float()
        feat_next = hidden_states[l + 1].squeeze(0)[prompt_len:, :].float()

        diff = feat_next - feat
        mag = torch.norm(diff, dim=-1).mean()

        feat_unit = feat / (torch.norm(feat, dim=-1, keepdim=True) + eps)
        feat_next_unit = feat_next / (torch.norm(feat_next, dim=-1, keepdim=True) + eps)
        cos_sim = torch.clamp(torch.sum(feat_unit * feat_next_unit, dim=-1), -1.0, 1.0)
        ang = torch.acos(cos_sim).mean()

        coe_r += (mag / mag_init - ang / ang_init).item()
        sin_sum += (mag * torch.sin(ang)).item()
        cos_sum += (mag * torch.cos(ang)).item()

    coe_r = coe_r / layer_num
    coe_c = math.sqrt(cos_sum ** 2 + sin_sum ** 2) / layer_num

    # --- cot_kinetics ---
    cot_kinetics = 0.0
    norm_term = torch.norm(avg_layer_features[-1] - avg_layer_features[0], dim=-1).item()
    if norm_term < eps:
        norm_term = eps

    for m in range(1, len(avg_layer_features)):
        f_cur = avg_layer_features[m]
        f_prev = avg_layer_features[m - 1]
        tau = torch.norm(f_cur - f_prev, dim=-1).item()

        if m >= 2:
            f_prev2 = avg_layer_features[m - 2]
            kappa = torch.norm((f_cur - f_prev) - (f_prev - f_prev2), dim=-1).item()
            cot_kinetics += tau / norm_term + kappa / norm_term
        else:
            cot_kinetics += tau / norm_term

    return coe_r, coe_c, cot_kinetics


def compute_logit_metrics(logits, prompt_len):
    """Compute entropy and self_certainty from output logits.

    Only generation tokens (after prompt) are used.

    Returns:
        (entropy, self_certainty)
    """
    gen_logits = logits.squeeze(0)[prompt_len - 1:-1].float()
    probs = F.softmax(gen_logits, dim=-1)

    epsilon = 1e-8
    # Entropy
    entropy_per_token = -(probs * torch.log(probs + epsilon)).sum(dim=-1)
    entropy = entropy_per_token.mean().item()

    # Self-certainty: KL(uniform || p)
    vocab_size = probs.shape[-1]
    log_term = torch.log(vocab_size * probs + epsilon)
    kl_u_p = -log_term.mean(dim=-1)
    self_certainty = kl_u_p.mean().item()

    return entropy, self_certainty


def score_all(model, tokenizer, questions, global_attn_heads, H_low, H_high, max_len):
    """Compute all uncertainty metrics for each question.

    Uses the shortest rollout per question for the forward pass.
    Long responses are processed in sliding windows of max_len tokens.
    """
    results = []
    for q in tqdm(questions, desc="Scoring"):
        prompt = q["prompt"]
        raw_prompt = q["raw_prompt"]
        golden_answer = q["golden_answer"]
        accuracy = q["accuracy"]
        consistency = q["consistency"]
        rollout_input_ids = q["rollout_input_ids"]
        prompt_len = q["prompt_len"]

        # Pick the shortest rollout for forward pass
        shortest_idx = min(range(len(rollout_input_ids)), key=lambda k: len(rollout_input_ids[k]))
        full_input_ids = rollout_input_ids[shortest_idx]

        empty_result = {
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "golden_answer": golden_answer,
            "accuracy": accuracy,
            "consistency": consistency,
            "pivot_num": 0,
            "entropy": 0.0,
            "self_certainty": 0.0,
            "coe_r": 0.0,
            "coe_c": 0.0,
            "cot_kinetics": 0.0,
        }

        if len(full_input_ids) <= prompt_len:
            results.append(empty_result)
            continue

        # Sliding window: split into chunks of max_len
        window_size = prompt_len + max_len
        input_ids_chunks = []
        for start in range(0, len(full_input_ids), max_len):
            chunk = full_input_ids[start:start + window_size]
            if len(chunk) <= prompt_len:
                continue
            input_ids_chunks.append(chunk)

        if not input_ids_chunks:
            results.append(empty_result)
            continue

        total_pivot_num = 0
        total_entropy = 0.0
        total_self_certainty = 0.0
        total_coe_r = 0.0
        total_coe_c = 0.0
        # For cot_kinetics, accumulate avg_layer_features across chunks
        acc_layer_features = []
        n_chunks = 0

        for chunk in input_ids_chunks:
            inputs = tokenizer.pad(
                {"input_ids": [chunk]},
                return_tensors="pt",
                return_attention_mask=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

            # --- Pivot num (attention-based) ---
            attentions = [attn[:, :, prompt_len:, prompt_len:] for attn in outputs.attentions]

            seq_len = attentions[0].size(-1)
            if seq_len == 0:
                del outputs, attentions
                torch.cuda.empty_cache()
                continue

            n_chunks += 1

            # Aggregate global attention heads
            count = 0
            total_attn = None
            num_layers = len(attentions)
            num_heads = attentions[0].size(1)
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    if f"({layer_idx}, {head_idx})" not in global_attn_heads:
                        continue
                    attn_map = attentions[layer_idx][0, head_idx].float()
                    if total_attn is None:
                        total_attn = torch.zeros_like(attn_map)
                    total_attn += attn_map
                    count += 1

            if total_attn is not None and count > 0:
                attn_map_np = (total_attn / count).cpu().numpy()
                signal = compute_long_range_attn(attn_map_np, H_low, H_high)

                if signal.size != 0:
                    height_thresh = np.percentile(signal, 95)
                    prominence_thresh = (signal.max() - signal.min()) * 0.05
                    peaks, _ = find_peaks(
                        signal,
                        height=height_thresh,
                        prominence=prominence_thresh,
                        distance=10,
                    )
                    total_pivot_num += len(peaks)

            # --- Entropy and self_certainty (logit-based) ---
            entropy, self_certainty = compute_logit_metrics(outputs.logits, prompt_len)
            total_entropy += entropy
            total_self_certainty += self_certainty

            # --- coe_r, coe_c (hidden-state-based, per chunk) ---
            coe_r, coe_c, _ = compute_hidden_state_metrics(outputs.hidden_states, prompt_len)
            total_coe_r += coe_r
            total_coe_c += coe_c

            # --- Accumulate layer features for cot_kinetics ---
            hidden_states = outputs.hidden_states
            for m, hid in enumerate(hidden_states):
                feat = hid.squeeze(0)[prompt_len:, :].float().mean(dim=0)
                if len(acc_layer_features) <= m:
                    acc_layer_features.append(feat)
                else:
                    acc_layer_features[m] = acc_layer_features[m] + feat

            del outputs, attentions, total_attn
            torch.cuda.empty_cache()

        if n_chunks > 0:
            total_entropy /= n_chunks
            total_self_certainty /= n_chunks
            total_coe_r /= n_chunks
            total_coe_c /= n_chunks

            # Compute cot_kinetics from accumulated layer features
            for m in range(len(acc_layer_features)):
                acc_layer_features[m] /= n_chunks

            eps = 1e-8
            cot_kinetics = 0.0
            norm_term = torch.norm(acc_layer_features[-1] - acc_layer_features[0], dim=-1).item()
            if norm_term < eps:
                norm_term = eps
            for m in range(1, len(acc_layer_features)):
                f_cur = acc_layer_features[m]
                f_prev = acc_layer_features[m - 1]
                tau = torch.norm(f_cur - f_prev, dim=-1).item()
                if m >= 2:
                    f_prev2 = acc_layer_features[m - 2]
                    kappa = torch.norm((f_cur - f_prev) - (f_prev - f_prev2), dim=-1).item()
                    cot_kinetics += tau / norm_term + kappa / norm_term
                else:
                    cot_kinetics += tau / norm_term
        else:
            cot_kinetics = 0.0

        results.append({
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "golden_answer": golden_answer,
            "accuracy": accuracy,
            "consistency": consistency,
            "pivot_num": total_pivot_num,
            "entropy": total_entropy,
            "self_certainty": total_self_certainty,
            "coe_r": total_coe_r,
            "coe_c": total_coe_c,
            "cot_kinetics": cot_kinetics,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute uncertainty metrics for each question from rollout responses.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to responses.jsonl from generate_response.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for scored JSONL (per-split)")
    parser.add_argument("--global_attn_path", type=str, required=True,
                        help="Path to global_attn_heads.pkl from attn_probing.py")
    parser.add_argument("--split", type=int, required=True,
                        help="Data split index for this process (0-indexed)")
    parser.add_argument("--n_gpus", type=int, required=True,
                        help="Total number of parallel processes")
    parser.add_argument("--n_per_prompt", type=int, default=8,
                        help="Number of rollouts per prompt (default: 8)")
    parser.add_argument("--H_low", type=int, default=20,
                        help="Long-range attention window lower bound (default: 20)")
    parser.add_argument("--H_high", type=int, default=100,
                        help="Long-range attention window upper bound (default: 100)")
    parser.add_argument("--max_len", type=int, default=2048,
                        help="Max response tokens per forward pass (default: 2048)")
    args = parser.parse_args()

    # Load global attention heads
    print(f"[Split {args.split}/{args.n_gpus}] Loading global_attn_heads from {args.global_attn_path}")
    with open(args.global_attn_path, "rb") as f:
        global_attn_heads = pickle.load(f)
    print(f"Loaded {len(global_attn_heads)} global attention heads")

    # Load and partition data
    print("Loading data...")
    data_list = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    N = len(data_list) // args.n_per_prompt
    chunk_size = math.ceil(N / args.n_gpus)
    q_start = args.split * chunk_size
    q_end = min((args.split + 1) * chunk_size, N)
    start_idx = q_start * args.n_per_prompt
    end_idx = q_end * args.n_per_prompt
    split_data = data_list[start_idx:end_idx]

    print(f"[Split {args.split}/{args.n_gpus}] Total {N} questions, "
          f"this split: questions [{q_start}, {q_end}) = {q_end - q_start} questions")

    # Tokenize and group responses by question
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    questions = []
    n_split_questions = q_end - q_start
    for i in tqdm(range(n_split_questions)):
        start = i * args.n_per_prompt
        group = split_data[start:start + args.n_per_prompt]

        prompt = group[0]['prompt']
        golden_answer = group[0]['answer']
        raw_prompt = group[0].get('raw_prompt', None)

        # Accuracy from pre-computed 'correct' field
        n_correct = sum(1 for d in group if d.get('correct', False))
        accuracy = n_correct / len(group)

        # Consistency (math data only; None for other types)
        data_type = group[0].get('data_type', None)
        responses = [d['generated_text'] for d in group]
        if data_type == 'math' or data_type is None:
            consistency = compute_consistency(responses, golden_answer)
        else:
            consistency = None

        # Tokenize each rollout
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        rollout_input_ids = []
        for d in group:
            generation = d['generated_text']
            generated_ids = tokenizer.encode(generation, add_special_tokens=False)
            input_ids = prompt_ids + generated_ids
            rollout_input_ids.append(input_ids)

        questions.append({
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "golden_answer": golden_answer,
            "accuracy": accuracy,
            "consistency": consistency,
            "rollout_input_ids": rollout_input_ids,
            "prompt_len": prompt_len,
        })

    # Load model
    print(f"[Split {args.split}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).cuda()
    model.eval()

    # Score
    print(f"Scoring {len(questions)} questions...")
    results = score_all(model, tokenizer, questions, global_attn_heads,
                        args.H_low, args.H_high, args.max_len)

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for info in results:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')

    print(f"[Split {args.split}] Saved {len(results)} scored questions to {args.output_path}")


if __name__ == "__main__":
    main()
