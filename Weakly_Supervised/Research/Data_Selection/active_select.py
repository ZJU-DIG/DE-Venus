"""Active data selection for semi-supervised RL training.

Two selection modes:

Labeled mode:
  1. Filter samples by accuracy in [acc_l, acc_h]
  2. Sort by uncertainty (most uncertain first), take top train_budget

Unlabeled mode:
  1. Sort by uncertainty ascending
  2. Uniformly sample probe_k probes, annotate (simulate / human)
  3. Sliding window on probes -> tau_1 (acc_h), tau_2 (acc_l) thresholds
  4. Filter to [tau_1, tau_2] range (probes excluded)
  5. Select label set via label_budget or acc_a threshold
  6. Output parquet: labeled + unlabeled samples

Supported --sort_by metrics:
  pivot_num, consistency, entropy, self_certainty, coe_r, coe_c,
  cot_kinetics, accuracy (labeled mode only)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd


# Whether to reverse-sort for "uncertainty ascending" order.
# reverse=False: raw value ascending = uncertainty ascending (e.g. pivot_num, entropy)
# reverse=True:  raw value descending = uncertainty ascending (e.g. consistency, accuracy)
SORT_REVERSE = {
    "pivot_num": False,
    "consistency": True,
    "entropy": False,
    "self_certainty": True,
    "coe_r": True,
    "coe_c": True,
    "cot_kinetics": True,
    "accuracy": True,
}


def load_scored_data(input_path):
    """Load scored JSONL file into a list of dicts."""
    data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list


def is_labeled_data(data_list):
    """Return True if all samples have non-None accuracy."""
    for d in data_list:
        if d["accuracy"] is None:
            return False
    return True


def uncertainty_value(d, metric, reverse):
    """Map metric value to uncertainty space (higher = more uncertain)."""
    val = d[metric]
    return -val if reverse else val


def sliding_window_threshold(sorted_data, window_size, target_acc, metric, reverse):
    """Find the uncertainty threshold where avg accuracy drops below target_acc.

    Scans sorted_data (uncertainty ascending) left-to-right. Returns the mean
    uncertainty value of the first window with avg accuracy < target_acc,
    or None if no such window exists.
    """
    if len(sorted_data) < window_size:
        avg_acc = np.mean([d["accuracy"] for d in sorted_data])
        if avg_acc < target_acc:
            return np.mean([uncertainty_value(d, metric, reverse) for d in sorted_data])
        return None

    for i in range(len(sorted_data) - window_size + 1):
        window = sorted_data[i:i + window_size]
        avg_acc = np.mean([d["accuracy"] for d in window])
        if avg_acc < target_acc:
            avg_uv = np.mean([uncertainty_value(d, metric, reverse) for d in window])
            return avg_uv
    return None


def uniform_sample(data_list, k):
    """Return k evenly-spaced indices from [0, len(data_list)-1]."""
    N = len(data_list)
    if k >= N:
        return list(range(N))
    indices = np.round(np.linspace(0, N - 1, k)).astype(int).tolist()
    return indices


def make_output_record(data, labeled_flag, data_source, ability, index):
    """Build a single output record for the training parquet."""
    raw_prompt = data["raw_prompt"]
    reward_model = {"ground_truth": data["golden_answer"], "style": "rule"}
    extra_info = {"index": index, "split": "train", "labeled": labeled_flag}
    return {
        "prompt": raw_prompt,
        "data_source": data_source,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def handle_labeled_mode(data_list, args):
    """Labeled mode: filter by accuracy range, select by uncertainty."""
    metric = args.sort_by
    reverse = SORT_REVERSE[metric]
    print(f"Labeled mode: {len(data_list)} total samples, sort_by={metric}")

    # Filter by accuracy range
    filtered = [d for d in data_list if args.acc_l <= d["accuracy"] <= args.acc_h]
    print(f"After filtering acc in [{args.acc_l}, {args.acc_h}]: {len(filtered)} samples")

    # Sort by uncertainty descending (most uncertain first)
    filtered.sort(key=lambda x: uncertainty_value(x, metric, reverse), reverse=True)

    if args.train_budget is not None:
        selected = filtered[:args.train_budget]
        print(f"Selected top {len(selected)} samples by {metric} (train_budget={args.train_budget})")
    else:
        selected = filtered
        print(f"No train_budget specified, keeping all {len(selected)} filtered samples")

    # All selected samples are labeled
    output_records = []
    for i, d in enumerate(selected):
        output_records.append(make_output_record(d, True, args.data_source, args.ability, i))

    return output_records


def handle_unlabeled_mode(data_list, args):
    """Unlabeled mode: probe -> threshold -> select -> output."""
    metric = args.sort_by
    reverse = SORT_REVERSE[metric]
    print(f"Unlabeled mode: {len(data_list)} total samples, sort_by={metric}")

    # Step 1: Sort by uncertainty ascending (certain/easy first)
    data_list.sort(key=lambda x: x[metric], reverse=reverse)
    print(f"Sorted by {metric} (uncertainty ascending). "
          f"Range: [{data_list[0][metric]}, {data_list[-1][metric]}]")

    # Step 2: Uniformly sample probe_k samples for annotation probing
    probe_indices = uniform_sample(data_list, args.probe_k)
    probe_data = [data_list[i] for i in probe_indices]
    print(f"Uniformly sampled {len(probe_data)} probe samples")

    # Step 3: Annotate probe samples
    if args.simulate:
        # Simulation: re-compute accuracy from original parquet + responses
        print("Simulation mode: computing accuracy from original parquet...")
        orig_df = pd.read_parquet(args.simulate_parquet_path)

        # Build user_content -> ground_truth lookup
        gt_lookup = {}
        for _, row in orig_df.iterrows():
            msgs = row['prompt']
            user_content = None
            for msg in msgs:
                if msg['role'] == 'user':
                    user_content = msg['content']
                    break
            if user_content is not None:
                gt_lookup[user_content] = row['reward_model']['ground_truth']

        # Load individual responses for accuracy computation
        print("Loading responses.jsonl for accuracy computation...")
        responses = []
        with open(args.responses_path, 'r', encoding='utf-8') as f:
            for line in f:
                responses.append(json.loads(line.strip()))

        # Group responses by prompt
        from collections import defaultdict
        resp_by_prompt = defaultdict(list)
        for r in responses:
            resp_by_prompt[r['prompt']].append(r)

        from verify_utils import verify_response, detect_data_type

        for d in probe_data:
            prompt = d["prompt"]
            golden_answer = d["golden_answer"]
            if golden_answer is None:
                # Fall back to original parquet lookup
                msgs = d["raw_prompt"]
                user_content = None
                for msg in msgs:
                    if msg['role'] == 'user':
                        user_content = msg['content']
                        break
                golden_answer = gt_lookup.get(user_content, None)
                if golden_answer is not None:
                    d["golden_answer"] = golden_answer

            if golden_answer is not None:
                group = resp_by_prompt.get(prompt, [])
                if group:
                    dt = detect_data_type(golden_answer)
                    n_correct = sum(1 for r in group
                                    if verify_response(r['generated_text'], golden_answer, dt))
                    d["accuracy"] = n_correct / len(group)
                else:
                    d["accuracy"] = 0.0
            else:
                d["accuracy"] = 0.0

        print(f"Probe accuracy stats: mean={np.mean([d['accuracy'] for d in probe_data]):.4f}, "
              f"median={np.median([d['accuracy'] for d in probe_data]):.4f}")

    elif args.probe_annotation_path is not None:
        # Load pre-annotated probe results
        print(f"Loading human annotations from {args.probe_annotation_path}...")
        annotated = []
        with open(args.probe_annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                annotated.append(json.loads(line.strip()))

        if len(annotated) != len(probe_data):
            print(f"ERROR: Expected {len(probe_data)} annotations, got {len(annotated)}")
            sys.exit(1)

        for d, a in zip(probe_data, annotated):
            d["accuracy"] = a["accuracy"]
            if "golden_answer" in a:
                d["golden_answer"] = a["golden_answer"]

        print(f"Loaded {len(annotated)} annotations")
    else:
        # No annotation source: export probes for human annotation and exit
        probe_output_path = os.path.splitext(args.output_path)[0] + "_probe.jsonl"
        print(f"Saving {len(probe_data)} probe samples for human annotation to {probe_output_path}")
        with open(probe_output_path, 'w', encoding='utf-8') as f:
            for d in probe_data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f"Please annotate the probe samples and re-run with --probe_annotation_path {probe_output_path}")
        sys.exit(0)

    # Step 4: Sliding window on probes to find tau_1 (acc_h) and tau_2 (acc_l)
    print(f"\nSliding window analysis (window_size={args.window_size}):")

    tau_1 = sliding_window_threshold(probe_data, args.window_size, args.acc_h, metric, reverse)
    tau_2 = sliding_window_threshold(probe_data, args.window_size, args.acc_l, metric, reverse)

    if tau_1 is None:
        print(f"WARNING: No window found with avg_acc < acc_h ({args.acc_h}). "
              f"Using min uncertainty as tau_1.")
        tau_1 = uncertainty_value(data_list[0], metric, reverse)
    if tau_2 is None:
        print(f"WARNING: No window found with avg_acc < acc_l ({args.acc_l}). "
              f"Using max uncertainty as tau_2.")
        tau_2 = uncertainty_value(data_list[-1], metric, reverse)

    print(f"tau_1 (acc_h={args.acc_h} threshold): {tau_1:.4f}")
    print(f"tau_2 (acc_l={args.acc_l} threshold): {tau_2:.4f}")

    # Step 5: Filter to [tau_1, tau_2] range, excluding probes
    probe_id_set = set(id(d) for d in probe_data)
    remaining = [d for d in data_list
                 if tau_1 <= uncertainty_value(d, metric, reverse) <= tau_2
                 and id(d) not in probe_id_set]
    print(f"After filtering by uncertainty in [{tau_1:.4f}, {tau_2:.4f}] (excl. probe): "
          f"{len(remaining)} samples")

    # Step 6: Determine label set (most uncertain first)
    remaining.sort(key=lambda x: uncertainty_value(x, metric, reverse), reverse=True)

    if args.label_budget is not None:
        # Fixed budget: reserve probe spots, fill rest from remaining
        budget_for_remaining = max(0, args.label_budget - len(probe_data))
        to_label = remaining[:budget_for_remaining]
        to_label_set = set(id(d) for d in to_label)
        not_labeled = [d for d in remaining if id(d) not in to_label_set]
        print(f"Label budget: {len(probe_data)} probe + {len(to_label)} selected "
              f"= {len(probe_data) + len(to_label)} labeled (budget={args.label_budget})")
    else:
        # Threshold-based: use acc_a sliding window to find tau_a
        probe_in_range = [d for d in probe_data
                          if tau_1 <= uncertainty_value(d, metric, reverse) <= tau_2]
        probe_in_range.sort(key=lambda x: uncertainty_value(x, metric, reverse))

        tau_a = sliding_window_threshold(probe_in_range, args.window_size, args.acc_a, metric, reverse)
        if tau_a is None:
            print(f"WARNING: No window found with avg_acc < acc_a ({args.acc_a}). "
                  f"Labeling all remaining samples.")
            to_label = remaining
            not_labeled = []
        else:
            print(f"tau_a (acc_a={args.acc_a} threshold): {tau_a:.4f}")
            to_label = [d for d in remaining
                        if uncertainty_value(d, metric, reverse) >= tau_a]
            not_labeled = [d for d in remaining
                           if uncertainty_value(d, metric, reverse) < tau_a]
            print(f"Selecting {len(to_label)} samples with uncertainty >= {tau_a:.4f} to label")

    # Step 7: Annotate to_label samples (probes already annotated)
    if args.simulate:
        print("Simulation: computing accuracy for selected samples...")
        from collections import defaultdict
        from verify_utils import verify_response, detect_data_type

        if 'resp_by_prompt' not in dir():
            responses = []
            with open(args.responses_path, 'r', encoding='utf-8') as f:
                for line in f:
                    responses.append(json.loads(line.strip()))
            resp_by_prompt = defaultdict(list)
            for r in responses:
                resp_by_prompt[r['prompt']].append(r)

        for d in to_label:
            prompt = d["prompt"]
            golden_answer = d["golden_answer"]
            if golden_answer is None:
                msgs = d["raw_prompt"]
                user_content = None
                for msg in msgs:
                    if msg['role'] == 'user':
                        user_content = msg['content']
                        break
                golden_answer = gt_lookup.get(user_content, None)
                if golden_answer is not None:
                    d["golden_answer"] = golden_answer

            if golden_answer is not None:
                group = resp_by_prompt.get(prompt, [])
                if group:
                    dt = detect_data_type(golden_answer)
                    n_correct = sum(1 for r in group
                                    if verify_response(r['generated_text'], golden_answer, dt))
                    d["accuracy"] = n_correct / len(group)
                else:
                    d["accuracy"] = 0.0
            else:
                d["accuracy"] = 0.0

        labeled_flag = True
    else:
        # Mark for deferred human annotation
        labeled_flag = "to_label"

    # Step 8: Apply train_budget (keep all labeled; trim unlabeled if over budget)
    all_labeled = probe_data + to_label
    if args.train_budget is not None:
        total = len(all_labeled) + len(not_labeled)
        if total > args.train_budget:
            max_unlabeled = max(0, args.train_budget - len(all_labeled))
            not_labeled.sort(key=lambda x: uncertainty_value(x, metric, reverse), reverse=True)
            not_labeled = not_labeled[:max_unlabeled]
            print(f"Train budget: trimmed to {len(all_labeled)} labeled + {len(not_labeled)} unlabeled "
                  f"= {len(all_labeled) + len(not_labeled)} total (budget={args.train_budget})")

    # Step 9: Build output records (unlabeled + probes + to_label)
    output_records = []
    index = 0

    for d in not_labeled:
        output_records.append(make_output_record(d, False, args.data_source, args.ability, index))
        index += 1

    for d in probe_data:
        output_records.append(make_output_record(d, True, args.data_source, args.ability, index))
        index += 1

    for d in to_label:
        output_records.append(make_output_record(d, labeled_flag, args.data_source, args.ability, index))
        index += 1

    print(f"\nOutput: {len(probe_data)} probe(labeled) + {len(to_label)} to_label({labeled_flag}) "
          f"+ {len(not_labeled)} unlabeled = {len(output_records)} total")
    return output_records


def main():
    parser = argparse.ArgumentParser(
        description="Active data selection for semi-supervised RL training.")

    # I/O
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to scored.jsonl from score_data.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output parquet path for training")

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["labeled", "unlabeled"],
                        help="'labeled' (data has accuracy) or 'unlabeled' (requires probing)")

    # Data metadata
    parser.add_argument("--data_source", type=str, default="dapo-math",
                        help="Data source tag for output parquet (default: dapo-math)")
    parser.add_argument("--ability", type=str, default="math",
                        help="Ability tag for output parquet (default: math)")

    # Sorting
    parser.add_argument("--sort_by", type=str, default="pivot_num",
                        choices=list(SORT_REVERSE.keys()),
                        help="Uncertainty metric for sorting/selection (default: pivot_num)")

    # Accuracy thresholds
    parser.add_argument("--acc_l", type=float, default=0.2,
                        help="Lower accuracy bound (default: 0.2)")
    parser.add_argument("--acc_h", type=float, default=0.8,
                        help="Upper accuracy bound (default: 0.8)")
    parser.add_argument("--acc_a", type=float, default=0.4,
                        help="Annotation threshold for tau_a detection (unlabeled, default: 0.4)")

    # Budget
    parser.add_argument("--train_budget", type=int, default=None,
                        help="Max total training samples")
    parser.add_argument("--label_budget", type=int, default=None,
                        help="Max samples to label, including probes (unlabeled mode)")

    # Probing (unlabeled mode)
    parser.add_argument("--probe_k", type=int, default=150,
                        help="Number of uniform probe samples (default: 150)")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Sliding window size for threshold detection (default: 20)")

    # Annotation
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate annotation using original parquet ground truth")
    parser.add_argument("--simulate_parquet_path", type=str, default=None,
                        help="Original train.parquet for simulation")
    parser.add_argument("--responses_path", type=str, default=None,
                        help="responses.jsonl for simulation accuracy computation")
    parser.add_argument("--probe_annotation_path", type=str, default=None,
                        help="Pre-annotated probe results (human annotation mode)")

    args = parser.parse_args()

    # Validate args
    if args.simulate:
        if args.simulate_parquet_path is None:
            parser.error("--simulate requires --simulate_parquet_path")
        if args.responses_path is None:
            parser.error("--simulate requires --responses_path")

    # Load data
    print(f"Loading scored data from {args.input_path}...")
    data_list = load_scored_data(args.input_path)
    print(f"Loaded {len(data_list)} samples")

    # Validate sort_by metric
    metric = args.sort_by
    if metric == "accuracy" and args.mode == "unlabeled":
        parser.error("--sort_by accuracy is only available in labeled mode "
                     "(unlabeled data has accuracy=None).")
    n_none = sum(1 for d in data_list if d.get(metric) is None)
    if n_none > 0:
        parser.error(f"--sort_by {metric}: {n_none}/{len(data_list)} samples have {metric}=None. "
                     f"This metric is not available for all data (e.g. consistency requires math data).")

    # Dispatch
    if args.mode == "labeled":
        output_records = handle_labeled_mode(data_list, args)
    else:
        output_records = handle_unlabeled_mode(data_list, args)

    # Save output parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    df = pd.DataFrame(output_records)
    df.to_parquet(args.output_path)
    print(f"\nSaved {len(output_records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
