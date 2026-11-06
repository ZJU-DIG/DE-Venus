# DE-Venus: Towards Data-Efficient LLM Post-Training

<div align="center">

**Pseudo-Label · Reward Shaping · Data Filtering**

<!-- [![](https://img.shields.io/badge/Built_on-verl-blue)](https://github.com/volcengine/verl)
[![](https://img.shields.io/github/license/yangshenzhi.ysz/DE-Venus)](https://github.com/yangshenzhi.ysz/DE-Venus/blob/main/LICENSE) -->

</div>

---

## Overview

Training LLMs with RL (e.g., GRPO/PPO) requires per-sample ground-truth rewards, which limits scalability when labeled data is scarce. **DE-Venus** closes this gap by injecting pseudo-supervision for **unlabeled** samples directly inside the training loop or filter out unreliable samples.

This repository provides **7 reproducible trainer implementations** and a **data selection pipeline** across three paradigms, all built as drop-in extensions of [verl](https://github.com/volcengine/verl)'s `RayPPOTrainer`:

| Paradigm | Core Idea | Methods |
|:--------:|-----------|---------|
| **Semi/Unsupervised RL** | Train with unlabeled or partially labeled data by replacing ground-truth rewards with intrinsic signals (entropy, KL divergence, voting consensus, etc.) | TTRL, Intuitor, EMRL, Co-rewarding, TraPO, GeoMin |
| **Noisy-Label RL** | Assume all data has labels but some are wrong; relabel or filter during training | NL |
| **Data Selection** | Actively selects the most annotation-worthy and training-worthy samples before RL training| PivotTrace |

Each trainer inherits from `RayPPOTrainer` and overrides only the data-processing hooks where pseudo-supervision is injected, keeping the PPO dataflow (rollout → reward → advantage → actor/critic update) intact.

---

## Algorithms

### Semi/Unsupervised RL

| Method | Unlabeled Reward Signal | Key Mechanism | Paper |
|:------:|:-----------------------:|:-------------|:-----:|
| **TTRL** | Majority-vote correctness | Test-Time Reinforcement Learning — uses majority voting as self-consistency reward for unlabeled data | [TTRL (2025)](https://arxiv.org/abs/2504.16084) |
| **Intuitor** | KL(U ‖ π_θ) | Intrinsic reward from KL divergence between uniform distribution and current policy; higher KL → more uniform policy → more to learn | [Intuitor (2025)](https://arxiv.org/abs/2505.19590) |
| **EMRL** | Entropy (tok-entropy / seq-entropy) | Uses token-level or sequence-level entropy of the policy as reward for unlabeled data | [EMRL (2025)](https://arxiv.org/abs/2505.15134) |
| **Co-rewarding** | Cross majority voting | Rewrites each question into a semantically equivalent variant; cross-votes between original and rewritten rollouts to produce pseudo-labels without any ground-truth | [Co-rewarding (2025)](https://arxiv.org/abs/2508.00410) |
| **TraPO** | Cosine similarity of pass-rate trajectories | Generates pseudo-labels via majority voting, then filters unreliable ones by comparing pass-rate trajectory similarity against labeled reference pool | [TraPO (2025)](https://arxiv.org/abs/2512.13106) |
| **GeoMin** | vMF distributional affinity + GMM filtering | Fits per-layer von Mises-Fisher distributions on hidden states (correct vs. incorrect); uses distributional affinity for pseudo-label confidence scoring, and GMM-based reliability filtering | [GeoMin (2026)](WIP) |

### Noisy-Label RL

| Method | Key Mechanism | Paper |
|:------:|:-------------|:-----:|
| **NL** | Detects and relabels noisy labels via trajectory cosine similarity; supports weak/strong noise settings with separate pseudo-label generation strategies | [NL (2025)](WIP) |

### Data Selection

| Method | Key Mechanism | Paper |
|:------:|:-------------|:-----:|
| **PivotTrace** | Supports two modes: **labeled filtering** — distill high-quality samples from annotated datasets for *less-is-more* training; **unlabeled selection** — smart-pick-in-the-dark on raw, unannotated data to identify which samples are worth labeling and which are worth training on | [PivotTrace (2026)](WIP) |

The scoring step (`score_data.py`) computes a suite of plug-and-play uncertainty metrics that users can freely combine for selection:

| Metric | Higher value means | Description |
|--------|--------------------|-------------|
| `pivot_num` | more uncertain | Number of attention pivot peaks from global attention heads |
| `entropy` | more uncertain | Mean token-level entropy of output distribution |
| `self_certainty` | more certain | Mean KL(uniform \|\| p) across output tokens |
| `consistency` | more certain | Answer agreement across rollouts (math data) |
| `coe_r` | more certain | Cumulative per-layer hidden-state magnitude/angle change ratio ([COE (2025)](https://arxiv.org/abs/2410.13640)) |
| `coe_c` | more certain | Resultant length of per-layer polar (magnitude, angle) vectors ([COE (2025)](https://arxiv.org/abs/2410.13640)) |
| `cot_kinetics` | more certain | Normalized speed + curvature of hidden states across layers ([CoT-Kinetics (2025)](https://arxiv.org/abs/2505.13408)) |

See `Research/Data_Selection/README.md` for detailed usage.

---

## Architecture

```
Weakly_Supervised/
├── config/
│   └── wsl.yaml                          # Hydra config (trainer_type + algorithm selection)
├── scripts/                              # Launch script per algorithm
│   ├── semi_run_ttrl.sh
│   ├── semi_run_intuitor.sh
│   ├── semi_run_emrl.sh
│   ├── semi_run_coreward.sh              # Co-rewarding: cross majority voting (unsupervised)
│   ├── semi_run_trapo.sh
│   ├── nl_run_olr_v2.sh
│   ├── semi_run_geomin_s1.sh             # GeoMin Stage 1: supervised warm-up with advantage reweighting
│   └── semi_run_geomin_s2.sh             # GeoMin Stage 2: semi-supervised with pseudo-label filtering
├── Research/
│   ├── Semi_RL/                          # Semi-supervised trainers
│   │   ├── ray_trainer_ttrl.py
│   │   ├── ray_trainer_intuitor.py
│   │   ├── ray_trainer_emrl.py
│   │   ├── ray_trainer_coreward.py       # Co-rewarding: cross majority voting between ori/aug rollouts
│   │   ├── ray_trainer_trapo.py
│   │   └── ray_trainer_geomin.py         # GeoMin: vMF distribution modeling + GMM filtering
│   ├── Noisy_Label/                      # Noisy-label trainer
│   │   └── nl_ray_trainer_olr_v2.py
│   ├── Data_Selection/                   # Active data selection pipeline (pre-training)
│   │   ├── 01_generate_response.sh       # Step 1: vLLM rollout generation
│   │   ├── 02_score_data.sh              # Step 2: attention probing + uncertainty scoring
│   │   ├── 03_active_select.sh           # Step 3: active selection → training parquet
│   │   ├── generate_response.py
│   │   ├── score_data.py
│   │   ├── active_select.py
│   │   ├── attn_probing.py
│   │   └── verify_utils.py
│   └── OPD/                              # Additional variant(TODO)
│       └── ray_trainer_semiopd.py
├── utils/
│   ├── training_utils.py                 # validate_and_log, log_train_metrics, compute_semi_metrics
│   ├── data_utils.py                     # get_labeled_unlabeled_indices, uid2idx_to_flat_list, extract_response_strings
│   ├── Semi_RL/
│   │   ├── major_voting.py               # Majority voting (extract → find majority → verify)
│   │   ├── ul_reward.py                  # Unlabeled reward: compute_klu_seq_reward, compute_entropy_seq_reward, apply_ul_seq_reward
│   │   ├── vmf_modeling.py               # VMFEstimator: per-class vMF fitting, kappa estimation, contrast logits
│   │   └── rewrite_questions.py          # Co-rewarding: question rewriting tool (vLLM + Qwen3-32B)
│   └── Noisy_Label/
│       └── __init__.py
└── wsl_main.py                           # Entry point: Hydra → WeakTrainerInterface → trainer.fit()
```

---

## Quick Start

### Prerequisites

- Python 3.12
- [verl](https://github.com/volcengine/verl) installed (verl==0.7.0 is recommended, follow their guide)
- `math_verify` package (for majority voting): `pip install math-verify`

### Launch Training

Each algorithm has a launch script in `scripts/`. Example for TTRL:

```bash
cd Weakly_Supervised
bash scripts/run_ttrl.sh
```



### Configuration

The Hydra config (`config/wsl.yaml`) selects the trainer:

```yaml
trainer_type: semi    # "semi" for Semi_RL, "nl" for Noisy_Label
algorithm: trapo      # ttrl | intuitor | emrl | co_reward | trapo | geomin (for semi); nl_olr_v2 (for nl)
```

Algorithm-specific hyperparameters are passed as CLI overrides. See individual `scripts/run_*.sh` for full examples.

---

## How It Works

### Labeled / Unlabeled Data Format

Training data is a single parquet file containing both labeled and unlabeled samples. Each sample's `extra_info` dict includes a `"labeled": true/false` field. The `get_labeled_unlabeled_indices()` utility in `data_utils.py` splits the batch at runtime.

### Where Pseudo-Supervision Enters the PPO Loop

```
                     ┌─ TTRL / TraPO: majority voting → replace reward
                     │
Rollout → Reward ────┼─ Intuitor: KL(U‖π_θ) → replace reward for unlabeled rows
                     │
                     ├─ Co-rewarding: cross majority voting between ori/aug rollouts → pseudo-labels
                     │
                     └─ EMRL: entropy → replace reward for unlabeled rows

                     ┌─ TraPO: cosine similarity of pass-rate trajectories → filter out unreliable samples
Data Processing ─────┤
                     ├─ GeoMin: vMF confidence + GMM → filter unreliable pseudo-labels
                     │
                     └─ NL: trajectory similarity → relabel / split noisy samples

Advantage ───────────── GeoMin: boundary reweighting (Eq.9) for ambiguous samples
```

### Unlabeled Reward Pipeline (Semi_RL)

For Intuitor and EMRL, the unlabeled reward computation follows a shared two-step pattern:

1. **Compute `seq_reward`** — Algorithm-specific: `compute_klu_seq_reward()` (Intuitor) or `compute_entropy_seq_reward()` (EMRL), both in `utils/Semi_RL/ul_reward.py`
2. **Write back to reward_tensor** — Common: `apply_ul_seq_reward()` clears existing reward for unlabeled rows, then writes `seq_reward` at the last valid token position

Each trainer's `_apply_entropy_reward_for_ul()` is a thin wrapper (~6 lines) calling these utilities.

---

## Adding a New Algorithm

1. Create a new trainer in `Research/Semi_RL/` (or `Research/Noisy_Label/`), inheriting `RayPPOTrainer`
2. Override `_validate()` to call `validate_and_log(self.config, self.global_steps, metric_dict)`
3. Override the hooks where your algorithm injects pseudo-supervision (e.g., reward replacement, data filtering)
4. Use `log_train_metrics(self.config, self.global_steps, metrics)` for training log writes
5. If your algorithm computes unlabeled rewards, add a `compute_*_seq_reward()` function in `utils/Semi_RL/ul_reward.py` and use `apply_ul_seq_reward()` for write-back
6. Register in `WeakTrainerInterface` (in `wsl_main.py`)
7. Add a launch script in `scripts/`

---

## Citation

If you find this work useful, please cite:

```bibtex
TODO
```
