"""Generate multiple rollout responses per prompt using vLLM.

Reads a parquet dataset, generates n_per_prompt responses for each prompt,
optionally verifies correctness against ground truth, and writes results
to a JSONL file for downstream scoring.

Output fields per line:
  prompt, raw_prompt, generated_text, answer, correct, data_type
"""

import os
import json
import argparse
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verify_utils import verify_response


def main():
    parser = argparse.ArgumentParser(
        description="Generate rollout responses and optionally verify correctness.")

    # I/O
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input parquet with 'prompt', 'ability', 'reward_model' columns")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL path for generated responses")

    # Generation
    parser.add_argument("--n_per_prompt", type=int, default=8,
                        help="Number of rollouts per prompt (default: 8)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens per generation (default: 4096)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling (default: 1.0)")
    parser.add_argument("--stop", type=str, default=None,
                        help='Stop sequences as JSON list, e.g. \'["\\n\\n"]\'')

    # vLLM engine
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallel size for vLLM (default: 4)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (default: 0.9)")

    # Verification
    parser.add_argument("--labeled", action="store_true",
                        help="Data has ground truth labels; enable correctness verification")
    parser.add_argument("--strict_boxed", action="store_true",
                        help="Require the last \\boxed{} to match ground truth (math only)")

    args = parser.parse_args()

    # Parse stop sequences
    if args.stop is not None:
        args.stop = json.loads(args.stop)
        print(f"Stop sequences: {args.stop!r}")

    # Load dataset
    df = pd.read_parquet(args.input_file)

    # Extract prompts (convert numpy types to native Python for JSON serialization)
    messages = df['prompt'].tolist()
    messages = [
        [{"role": str(m["role"]), "content": str(m["content"])} for m in msg]
        for msg in messages
    ]

    # Read data_type from 'ability' column
    data_types = [str(a) for a in df['ability'].tolist()]
    n_by_type = {}
    for dt in data_types:
        n_by_type[dt] = n_by_type.get(dt, 0) + 1
    print(f"Data types: {n_by_type}")

    # Ground truth labels
    if args.labeled:
        answers = [row['ground_truth'] for row in df['reward_model'].tolist()]
        assert len(messages) == len(answers)
    else:
        answers = [None] * len(messages)
        print("Unlabeled mode: skipping verification")

    # Initialize tokenizer and vLLM engine
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_per_prompt,
        stop=args.stop,
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Apply chat template
    gen_prompts = []
    for cur_message in messages:
        gen_prompt = tokenizer.apply_chat_template(
            cur_message,
            tokenize=False,
            add_generation_prompt=True,
        )
        gen_prompts.append(gen_prompt)

    print(f"Example prompt: {gen_prompts[0]}")
    print(f"temperature={args.temperature}, top_p={args.top_p}, "
          f"max_tokens={args.max_tokens}, n={args.n_per_prompt}")

    outputs = llm.generate(gen_prompts, sampling_params)

    # Write results with optional verification
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    total_correct = 0
    total_responses = 0
    with open(args.output_file, 'w') as fo:
        for i, output in enumerate(outputs):
            prompt = output.prompt
            dt = data_types[i]
            for j in range(args.n_per_prompt):
                generated_text = output.outputs[j].text
                if args.labeled:
                    is_correct = verify_response(generated_text, answers[i], dt,
                                                  strict_boxed=args.strict_boxed)
                else:
                    is_correct = False
                total_correct += int(is_correct)
                total_responses += 1
                item = {
                    'prompt': prompt,
                    'raw_prompt': messages[i],
                    'generated_text': generated_text,
                    'answer': answers[i],
                    'correct': is_correct,
                    'data_type': dt,
                }
                fo.write(json.dumps(item) + '\n')

    acc = total_correct / total_responses if total_responses > 0 else 0.0
    print(f"Saved {total_responses} responses to {args.output_file}")
    print(f"Overall accuracy: {total_correct}/{total_responses} = {acc:.4f}")


if __name__ == "__main__":
    main()
