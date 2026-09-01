import os
import json
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_long_range_attn_gpu(attn_maps, H_low, H_high):
    """Compute mean long-range attention for all heads at once on GPU.

    Args:
        attn_maps: tensor of shape (num_heads, seq_len, seq_len) on GPU.
        H_low: lower distance bound.
        H_high: upper distance bound.

    Returns:
        tensor of shape (num_heads,) — average long-range attention per head.
    """
    seq_len = attn_maps.shape[-1]
    valid_len = max(0, seq_len - H_low)
    if valid_len == 0:
        return torch.zeros(attn_maps.shape[0], device=attn_maps.device)

    # cumsum along rows (dim=1): cumsum[h, i, k] = sum of attn_maps[h, 0:i+1, k]
    cumsum = torch.cumsum(attn_maps, dim=1)

    # For each token k in [0, valid_len), compute mean attn from [k+H_low, k+H_high)
    ks = torch.arange(valid_len, device=attn_maps.device)
    starts = ks + H_low
    ends = torch.clamp(ks + H_high, max=seq_len)
    lengths = (ends - starts).float()

    # Gather: cumsum[:, end-1, k] - cumsum[:, start-1, k]
    end_vals = cumsum[:, ends - 1, ks]          # (num_heads, valid_len)
    start_vals = torch.where(
        starts > 0,
        cumsum[:, starts - 1, ks],
        torch.zeros_like(end_vals),
    )
    col_sums = end_vals - start_vals
    means = col_sums / lengths.unsqueeze(0)

    return means.mean(dim=1)                    # (num_heads,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to responses JSONL from generate_response.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save global_attn_heads pickle file")
    parser.add_argument("--n_per_prompt", type=int, default=8)
    parser.add_argument("--H_low", type=int, default=20)
    parser.add_argument("--H_high", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=2048,
                        help="Max sequence length for probing forward pass")
    args = parser.parse_args()

    # ---- Load data (only first n_per_prompt responses) ----
    print("Loading data...")
    data_list = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    probe_data = data_list[:args.n_per_prompt]
    print(f"Using {len(probe_data)} responses for probing")

    # ---- Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    all_input_ids = []
    for data in probe_data:
        prompt = data['prompt']
        generation = data['generated_text']

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = tokenizer.encode(generation, add_special_tokens=False)
        # Use only the generated part (response), truncated to max_len
        input_ids = (prompt_ids + generated_ids)[len(prompt_ids):len(prompt_ids) + args.max_len]
        all_input_ids.append(input_ids)

    # ---- Load model ----
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).cuda()
    model.eval()

    # ---- Probing (GPU-accelerated) ----
    print("Probing attention heads...")
    num_layers = None
    num_heads = None
    # Accumulate per-head scores: (num_layers, num_heads)
    score_accum = None

    for idx in tqdm(range(len(all_input_ids)), desc="Probing"):
        inputs = tokenizer.pad(
            {"input_ids": [all_input_ids[idx]]},
            return_tensors="pt",
            return_attention_mask=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

        if num_layers is None:
            num_layers = len(attentions)
            num_heads = attentions[0].size(1)
            score_accum = torch.zeros(num_layers, num_heads, device=model.device)

        for layer_idx in range(num_layers):
            # attentions[layer_idx] shape: (1, num_heads, seq_len, seq_len)
            attn_maps = attentions[layer_idx][0].float()  # (num_heads, seq_len, seq_len)
            head_scores = compute_long_range_attn_gpu(attn_maps, args.H_low, args.H_high)
            score_accum[layer_idx] += head_scores

        del outputs, attentions
        torch.cuda.empty_cache()

    score_accum /= len(all_input_ids)

    # ---- Select top 20% as global attention heads ----
    scores_np = score_accum.cpu().numpy()
    values = scores_np.flatten()
    threshold = np.percentile(values, 80)

    global_attn_heads = set()
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            if scores_np[layer_idx, head_idx] >= threshold:
                global_attn_heads.add(f"({layer_idx}, {head_idx})")

    print(f"Identified {len(global_attn_heads)} global attention heads "
          f"(threshold={threshold:.6f})")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(global_attn_heads, f)

    print(f"Saved global_attn_heads to {args.output_path}")


if __name__ == "__main__":
    main()
