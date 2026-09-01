# Selection Pipeline

Active data selection for semi-supervised RLVR training.

```
train.parquet ──▶ 01_generate_response.sh ──▶ 02_score_data.sh ──▶ 03_active_select.sh
                   responses.jsonl              scored.jsonl         selected_train.parquet
```

## Quick Start

Configure `ROOT`, `MODEL_PATH`, `DATASET`, `BASE_MODEL` in each shell script, then run sequentially:

```bash
bash 01_generate_response.sh   # Generate rollout responses (vLLM)
bash 02_score_data.sh          # Attention probing + uncertainty scoring
bash 03_active_select.sh       # Active selection → training parquet
```

## Step 3: Active Selection

The core of this pipeline. Reads `scored.jsonl` and outputs a training parquet. Controlled by `MODE` and `SORT_BY` in `03_active_select.sh`.

### Two Modes

#### Labeled Mode (`--mode labeled`)

For data where ground truth is available and accuracy has been computed (i.e., Step 1 ran with `--labeled`).

1. Filter samples to `acc_l <= accuracy <= acc_h`
2. Sort by `--sort_by` metric (most uncertain first)
3. Take top `--train_budget` samples (or all if not set)

All output samples have `labeled=True`.

```bash
python3 active_select.py --mode labeled \
  --sort_by pivot_num --acc_l 0.2 --acc_h 0.7 --train_budget 4000
```

#### Unlabeled Mode (`--mode unlabeled`)

For data without ground truth. Uses probe-based annotation to discover accuracy thresholds.

1. Sort all samples by `--sort_by` metric (uncertainty ascending)
2. Uniformly sample `--probe_k` probes across the sorted list
3. **Annotate probes** (see Annotation below) to obtain their accuracy
4. Sliding window on probes → find tau_1 (`acc_h` threshold) and tau_2 (`acc_l` threshold)
5. Filter samples to [tau_1, tau_2] uncertainty range (probes excluded)
6. Select label set from filtered samples:
   - If `--label_budget` is set: take top N most uncertain (budget includes probes)
   - Otherwise: use `--acc_a` sliding window to find tau_a, label samples above tau_a
7. **Annotate selected label set** (same mechanism as step 3)
8. Apply `--train_budget` if set (keep all labeled, trim unlabeled)
9. Output: probes (`labeled=True`) + label set + remaining unlabeled (`labeled=False`)

```bash
python3 active_select.py --mode unlabeled \
  --sort_by pivot_num --acc_l 0.2 --acc_h 0.7 --acc_a 0.3 \
  --probe_k 200 --window_size 20 \
  --simulate --simulate_parquet_path train.parquet --responses_path responses.jsonl
```

### Annotation (Unlabeled Mode)

In unlabeled mode, probes and selected samples need accuracy annotation. Three options:

| Flag | Behavior |
|------|----------|
| `--simulate` | **Simulation**: Re-compute accuracy from original parquet ground truth + responses.jsonl. Requires `--simulate_parquet_path` and `--responses_path`. Output samples get `labeled=True`. |
| `--probe_annotation_path` | **Human (pre-annotated)**: Load accuracy from a previously annotated JSONL file. Used on the second run after human annotation. |
| _(neither)_ | **Human (deferred)**: Export probe samples to `*_probe.jsonl` and exit. User annotates offline, then re-runs with `--probe_annotation_path`. Selected label set gets `labeled="to_label"`. |

Typical human annotation workflow:
```
# Run 1: export probes
python3 active_select.py --mode unlabeled ...
# → writes selected_train_probe.jsonl, then exits

# Human annotates the probe file (adds "accuracy" field)

# Run 2: use annotated probes
python3 active_select.py --mode unlabeled ... --probe_annotation_path selected_train_probe.jsonl
# → outputs parquet with labeled="to_label" for samples needing annotation
```

### Parameters

| Parameter | Default | Mode | Description |
|-----------|---------|------|-------------|
| `--mode` | — | both | `labeled` or `unlabeled` |
| `--sort_by` | `pivot_num` | both | Uncertainty metric for sorting |
| `--acc_l` / `--acc_h` | 0.2 / 0.7 | both | Accuracy filter bounds |
| `--train_budget` | - | both | Max total training samples |
| `--probe_k` | 200 | unlabeled | Number of uniform probe samples |
| `--window_size` | 20 | unlabeled | Sliding window size for threshold detection |
| `--acc_a` | 0.3 | unlabeled | Threshold for label set selection (when no `--label_budget`) |
| `--label_budget` | — | unlabeled | Fix labeled count (including probes) |
| `--simulate` | on | unlabeled | Simulate annotation from original data |

### Available `--sort_by` Metrics

| Metric | Higher value means | Notes |
|--------|--------------------|-------|
| `pivot_num` | more uncertain | |
| `entropy` | more uncertain | |
| `consistency` | more certain | math data only |
| `self_certainty` | more certain | |
| `coe_r` | more certain | |
| `coe_c` | more certain | |
| `cot_kinetics` | more certain | |
| `accuracy` | more certain | labeled mode only |
