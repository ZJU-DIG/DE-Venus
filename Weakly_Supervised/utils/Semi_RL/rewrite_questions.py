"""
Rewrite math questions using vLLM for Co-rewarding training.

This script takes a parquet file with math questions and generates semantically
equivalent but differently worded versions using a large language model.

Usage:
    python3 rewrite_questions.py \
        --input_path data/deepmath-hard/train.parquet \
        --model_path /path/to/Qwen3-32B \
        --tokenizer_path /path/to/Qwen3-32B \
        --output_parquet data/deepmath-hard/train_rewrite.parquet \
        --output_original_parquet data/deepmath-hard/train_original.parquet

Output:
    - train_rewrite.parquet: rewritten questions (same schema as input)
    - train_original.parquet: aligned original questions (only successfully rewritten rows)
    - train_rewrite.jsonl: raw original/rewritten pairs for inspection
"""

import argparse
import copy
import json
import os

import pandas as pd
from tqdm import tqdm

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


REWRITE_INSTRUCTION = """You are given a math problem. Please rewrite it using different wording and a different real-world scenario, while keeping the underlying mathematical meaning and answer exactly the same.

Guidelines:
1. Do not change the math logic or the final answer.
2. Use different words and a new context to make it look like a different problem.
3. Avoid copying phrases or sentence structures from the original.
4. Make sure the rewritten question is natural, clear, and solvable.
5. Output ONLY between the following markers, and strictly in this format (no extra explanation):

### RESULT_START
ORIGINAL:
<original question>
REWRITE:
<rewritten question>
### RESULT_END
"""


def build_prompt_from_message(content, tokenizer, enable_thinking=False):
    """Build a chat prompt for the rewriting task."""
    full_prompt = REWRITE_INSTRUCTION + "\n" + content.strip()
    messages = [{"role": "user", "content": full_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def extract_between_markers(text, start_marker="### RESULT_START", end_marker="### RESULT_END"):
    """Extract original and rewritten questions from the model output."""
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Failed to locate RESULT_START and RESULT_END markers.")
    block = text[start + len(start_marker) : end].strip()
    parts = block.split("REWRITE:", 1)
    if len(parts) != 2:
        raise ValueError("Failed to split ORIGINAL and REWRITE in output.")
    original = parts[0].replace("ORIGINAL:", "").strip()
    rewritten = parts[1].strip()
    return original, rewritten


def main(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load data
    df = pd.read_parquet(args.input_path)
    questions = df[args.question_column].tolist()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # Load model
    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=20480)

    successful_indices = []
    batch_size = args.batch_size

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as jsonl_writer:
        for i in tqdm(range(0, len(questions), batch_size), desc="Rewriting"):
            batch_indices = list(range(i, min(i + batch_size, len(questions))))
            batch_questions = [questions[idx] for idx in batch_indices]
            prompts = [
                build_prompt_from_message(q[1]["content"], tokenizer, args.enable_thinking)
                for q in batch_questions
            ]
            outputs = llm.generate(prompts, sampling_params)

            for j, output in enumerate(outputs):
                text = output.outputs[0].text
                try:
                    original, rewritten = extract_between_markers(text)
                    idx = batch_indices[j]
                    entry = copy.deepcopy(df.at[idx, args.question_column])
                    entry[1]["content"] = rewritten
                    df.at[idx, args.question_column] = entry
                    successful_indices.append(idx)
                    jsonl_writer.write(
                        json.dumps({"original": original, "rewritten": rewritten}, ensure_ascii=False) + "\n"
                    )
                except Exception as e:
                    print(f"[Warning] Failed to parse output for index {batch_indices[j]}: {e}")
                    continue

    print(f"Successfully rewrote {len(successful_indices)} / {len(questions)} questions.")

    if successful_indices:
        # Save aligned original parquet (only successfully rewritten rows)
        df_orig = pd.read_parquet(args.input_path)
        df_matched_orig = df_orig.iloc[successful_indices].reset_index(drop=True)
        df_matched_orig.to_parquet(args.output_original_parquet, index=False)
        print(f"Saved aligned original data to {args.output_original_parquet}")

        # Save rewritten parquet
        df_rewrite = df.iloc[successful_indices].reset_index(drop=True)
        df_rewrite.to_parquet(args.output_parquet, index=False)
        print(f"Saved rewritten data to {args.output_parquet}")
    else:
        print("Warning: No successful rewrites found! Output will be empty.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite math problems for Co-rewarding training.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Path to output JSONL (for inspection)")
    parser.add_argument("--output_parquet", type=str, default=None, help="Path to output rewritten parquet")
    parser.add_argument("--output_original_parquet", type=str, default=None, help="Path to aligned original parquet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to rewriting model (e.g., Qwen3-32B)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--question_column", type=str, default="prompt", help="Column name for questions")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for vLLM generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="vLLM tensor parallel size")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3's thinking mode")
    args = parser.parse_args()

    # Default paths
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    input_dir = os.path.dirname(args.input_path)
    input_stem = os.path.splitext(os.path.basename(args.input_path))[0]
    if args.output_jsonl is None:
        args.output_jsonl = os.path.join(input_dir, f"{input_stem}_rewrite.jsonl")
    if args.output_parquet is None:
        args.output_parquet = os.path.join(input_dir, f"{input_stem}_rewrite.parquet")
    if args.output_original_parquet is None:
        args.output_original_parquet = os.path.join(input_dir, f"{input_stem}_original.parquet")

    main(args)
