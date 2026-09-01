# DE-Venus: Towards Data-Efficient LLM Post-Training

<div align="center">

**Semi-Supervised RL · Noisy-Label RL · Unsupervised RL**

Built on [verl](https://github.com/volcengine/verl) · GRPO by default · Apache-2.0

</div>

---

## Overview

**Weakly_Supervised** is organized around **three research pillars**. All trainers subclass verl's `RayPPOTrainer`, use **`algorithm.adv_estimator=grpo`**, and only customize the GRPO stages where labels are missing, wrong, or unused (rollout → reward → filter → advantage → update).

| Pillar | Problem setting | Status | Entry |
|:------:|-----------------|--------|-------|
| [**Semi-Supervised RL**](#1-semi-supervised-rl) | Mix of labeled + unlabeled; pseudo-reward / filter / iterative promote | **Stable** — TraPO v2 pipeline documented | `trainer_type=semi` |
| [**Noisy-Label RL**](#2-noisy-label-rl-under-active-development) | All samples have labels, but some GT are wrong | **Under active development** | `trainer_type=nl` |
| [**Unsupervised RL**](#3-unsupervised-rl-under-active-development) | No trusted labels in the training loop | **Under active development** | `trainer_type=unsupervised` |

<!-- > **Why `RayPPOTrainer` / `verl.trainer.ppo`?**  
> verl bundles GAE/PPO, **GRPO**, REMAX, etc. behind one trainer API. Our code runs **GRPO**, not classic PPO with a critic. -->

```
                    ┌─────────────────────────────────────┐
                    │     Weakly_Supervised (GRPO)        │
                    └─────────────────────────────────────┘
           ┌────────────────┬────────────────┬────────────────┐
           ▼                ▼                ▼
    Semi-Supervised    Noisy-Label      Unsupervised
    labeled + unlabeled  wrong GT        no labels in-loop
    TTRL, TraPO v2…      NL, nl_olr_v2   certainty, TTRL…
```

---

## 1. Semi-Supervised RL

**Goal:** Train on a parquet where only part of the data has reliable `ground_truth`; assign or filter supervision for the rest inside GRPO.

### Algorithms

| Method | Signal for unlabeled data | Mechanism | Paper |
|:------:|:-------------------------:|:----------|:-----:|
| **TTRL** | Majority-vote correctness | Self-consistency reward from test-time voting | [TTRL (2025)](https://arxiv.org/abs/2504.16084) |
| **Intuitor** | KL(U ‖ π_θ) | Intrinsic reward from policy vs uniform | [Intuitor (2025)](https://arxiv.org/abs/2505.19590) |
| **EMRL** | Token / sequence entropy | Entropy as learning signal | [EMRL (2025)](https://arxiv.org/abs/2505.21360) |
| **Co-rewarding** | Cross majority voting | Rewrite each question into a semantically equivalent variant; cross-vote between original and rewritten rollouts to produce pseudo-labels without any ground-truth | [Co-rewarding (2025)](https://arxiv.org/abs/2508.00410) |
| **TraPO (v1)** | Pass-rate trajectory cosine | Majority pseudo-labels + filter vs labeled reference pool | [TraPO (2025)](https://arxiv.org/abs/2512.13106) |
| **TraPO v2** | Step-wise pass-rate probes | Warmup (labeled only) → full probe → promote → optional val early-stop loop | Same + [pipeline](#trapo-v2-full-pipeline) |
| **GeoMin** | vMF distributional affinity + GMM filtering | Fits per-layer von Mises-Fisher distributions on hidden states; uses distributional affinity for pseudo-label confidence scoring and GMM-based reliability filtering | [GeoMin (2026)](WIP) |

| `wsl_algorithm` | Trainer | Launch script |
|---------------|---------|---------------|
| `ttrl` | `ray_trainer_ttrl.py` | `scripts/semi_run_ttrl.sh` |
| `intuitor` | `ray_trainer_intuitor.py` | `scripts/semi_run_intuitor.sh` |
| `emrl` | `ray_trainer_emrl.py` | `scripts/semi_run_emrl.sh` |
| `co_reward` | `ray_trainer_coreward.py` | `scripts/semi_run_coreward.sh` |
| `trapo` | `ray_trainer_trapo.py` | `scripts/semi_run_trapo.sh` |
| `trapo_v2` | `ray_trainer_trapo_v2.py` | **`scripts/trapo_v2.sh`** (`pipeline` / `loop` / `train` / `promote` / `probe` / `final`) |
| `geomin` | `ray_trainer_geomin.py` | `scripts/semi_run_geomin_s1.sh`, `scripts/semi_run_geomin_s2.sh` |

### TraPO v1 vs TraPO v2

| | **TraPO v1** (`trapo`) | **TraPO v2** (`trapo_v2`) |
|--|------------------------|---------------------------|
| Training | Single long run with cosine filter after warmup | Default: **stop after warmup** per round for probe artifacts |
| Dataset | Fixed parquet | **Iterative**: promote unlabeled → grow labeled set, keep remaining unlabeled |
| Init each round | — | Pipeline reloads **`BASE_MODEL`** (not previous iter ckpt) |
| Promotion | — | `final_passrate ≥ PASSRATE_MIN` **and** `slope > SLOPE_MIN` on current majority |
| Stop criteria | — | Val **`average`** plateau, no new promotions, or `MAX_ITER` |

### TraPO v2 full pipeline

**Detailed guide:** [trapo_v2_readme.md](readme/Semi_RL/trapo_v2_readme.md) (EN) · [cn_trapo_v2_readme.md](readme/Semi_RL/cn_trapo_v2_readme.md) (中文) — data format, warmup/probe/promote, env vars, outputs.

End-to-end iterative semi-supervision (recommended entry point for TraPO v2):

```bash
bash Weakly_Supervised/scripts/trapo_v2.sh pipeline
# RUN_FINAL_LONG_TRAIN=0  # skip optional final long GRPO phase
bash Weakly_Supervised/scripts/trapo_v2.sh help
```

Env: `PASSRATE_MIN`, `SLOPE_MIN`, `VAL_PATIENCE`, `ITER_ROOT`, `TRAPO_WARMUP_STEPS`, etc. (`trapo_v2_lib.sh`).

Each iter round uses `trainer.resume_mode=disable` so weights reload from `BASE_MODEL` at step 0 (not `iter_k/ckpt`). To resume a crashed job in the **same** iter, set `TRAPO_RESUME_CKPT=1`.

In `loop` / `pipeline`, `TRAPO_WARMUP_STEPS` and `TRAPO_PROBE_INTERVAL` scale each round by labeled count ratio: `round(old * n_labeled_new / n_labeled_old)` (iter 0 uses env baselines). Disable with `TRAPO_SCALE_WARMUP_BY_LABELED=0`.

**Per-iteration outputs** under `ITER_ROOT/`:

```
loop_state.json, final_train.parquet
iter_k/log/probe_dumps/all_probes_summary.jsonl
iter_k/log/valid/global_step_*.log   # metric: average
iter_k/train_next.parquet, iter_k/ckpt/
```

**Probe → promote tooling:** `utils/analysis/` (`build_stage2_dataset.py`, `stage2_selection.py`, `probe_analysis.sh`).

### Semi-supervised quick start

```bash
export ROOT=$(pwd)
export SAVE_ROOT=./outputs
export BASE_MODEL=/path/to/Qwen3-8B-Base
export BASE_PARQUET=$ROOT/data/example/semi-supervised-small-fixed.parquet

bash Weakly_Supervised/scripts/trapo_v2.sh probe      # single warmup+probe round
bash Weakly_Supervised/scripts/trapo_v2.sh pipeline   # full iterative loop
```

```bash
python3 -m Weakly_Supervised.wsl_main \
    trainer_type=semi \
    wsl_algorithm=trapo_v2 \
    algorithm.adv_estimator=grpo \
    data.train_files=...
```

### Semi-supervised data format

- `extra_info.labeled`: `true` / `false`
- `reward_model.ground_truth`: used for labeled rows and voting
- After TraPO v2 promotion: `labeled=true`, pseudo-GT in `reward_model`, fields like `pseudo_slope`, `promotion_passrate_min`

Split at runtime: `utils/data_utils.py` → `get_labeled_unlabeled_indices()`.

---

## 2. Noisy-Label RL *(under active development)*

> **Status:** Core trainers and launch scripts exist, but APIs, docs, and reproducible benchmarks are still being stabilized. Expect breaking changes. Contributions and issue reports welcome.

**Goal:** Every training sample has a label, but a fraction of `ground_truth` values are wrong (weak supervision from noisy annotation). Detect, relabel, or down-weight unreliable samples during GRPO using **trajectory / pass-rate similarity**—analogous in spirit to TraPO's filtering, but applied to label noise rather than missing labels.

### Planned / in-repo components

| Component | Path | Notes |
|-----------|------|--------|
| **NL-OLR v2** | `Research/Noisy_Label/nl_ray_trainer_olr_v2.py` | Thin wrapper over **`RayPPOTrainerTraPOv2`** (same warmup/probe/stop as TraPO v2) |
| **Relabel** | `utils/analysis/build_noisy_label_dataset.py` | Majority must beat **initial GT** passrate slope by `SLOPE_MARGIN` (GT absent in rollouts → slope 0); then rewrite if ≠ current label |
| **Launch** | `scripts/nl_run_olr_v2.sh`, `olr_v2_lib.sh` | Default data `data/example/noisy_label.parquet` |

### How it differs from TraPO v2 (semi-supervised)

| | TraPO v2 | NL-OLR v2 |
|--|----------|-----------|
| Row types | **Labeled** + **unlabeled** (`labeled=false` pool) | **All rows `labeled=true`** (no unlabeled pool) |
| Offline step | **Promote** unlabeled → labeled (`build_stage2_dataset`) | **Relabel** wrong GT (`build_noisy_label_dataset`) |
| Training | `RayPPOTrainerTraPOv2` + semi `train_mode` | Same trainer class; `trainer_type=nl`, `train_mode=weak` |
| Data | `semi-supervised*.parquet` | `noisy_label*.parquet` — loop sets `labeled=true` on every row if missing |

### Experimental launch *(API not finalized)*

```bash
python3 -m Weakly_Supervised.wsl_main \
    trainer_type=nl \
    wsl_algorithm=nl_olr_v2 \
    algorithm.adv_estimator=grpo \
    data.train_files=$ROOT/data/example/noisy_label.parquet \
    ...
```

```bash
bash Weakly_Supervised/scripts/nl_run_olr_v2.sh pipeline      # iterative warmup/probe/relabel (TraPO v2-style)
bash Weakly_Supervised/scripts/nl_run_olr_v2.sh loop          # iteration only
```

**Roadmap (WIP):** unified README examples, parity with TraPO v2-style probe dumps, standard noisy-label benchmarks, and clearer Hydra config blocks for weak vs strong noise regimes.

---

## 3. Unsupervised RL *(under active development)*

> **Status:** Trainer and reward utilities are implemented for experimentation; production-ready configs, evaluation protocols, and documentation are still in progress.

**Goal:** GRPO when **no ground-truth labels** are used in the training loop. Rewards come from **intrinsic or self-supervised signals** (certainty, ensemble agreement, self-verify, TTRL-style voting on generated pseudo-GT).

### Planned / in-repo components

| Component | Path | Notes |
|-----------|------|--------|
| **Trainer** | `Research/Unsupervised_RL/ray_trainer_unsupervised.py` | Overrides `fit()` for reward shaping branches |
| **Utilities** | `utils/Unsupervised_RL/ttrl_utils.py` | TTRL GT application, ensemble / certainty helpers |
| **Launch** | `scripts/unsupervised_run.sh` | `trainer_type=unsupervised` (no `wsl_algorithm` branch) |

### Reward modes *(Hydra, subject to change)*

Configured via `unsupervised_reward` and/or `ttrl` blocks in Hydra, for example:

| Mode | Config hint | Idea |
|------|-------------|------|
| **TTRL-style voting** | `ttrl.enable=true` | Multiple votes per prompt → majority pseudo-GT → GRPO reward |
| **Ensemble** | `unsupervised_reward.type=ensemble` | Ensemble-based pseudo labels from rollouts |
| **Certainty** | `unsupervised_reward.type=certainty` | Confidence / certainty-shaped rewards |

See `ray_trainer_unsupervised.py` and `ttrl_utils.py` for current flags.

### Experimental launch *(API not finalized)*

```bash
python3 -m Weakly_Supervised.wsl_main \
    trainer_type=unsupervised \
    algorithm.adv_estimator=grpo \
    data.train_files=$ROOT/data/example/semi-supervised.parquet \
    ...
```

```bash
bash Weakly_Supervised/scripts/unsupervised_run.sh
```

**Roadmap (WIP):** dedicated unsupervised parquet conventions, documented Hydra yaml snippets per reward type, val metrics without leaking GT, and alignment with the semi-supervised probe/analysis toolchain where applicable.

---

## Repository layout

```
Weakly_Supervised/
├── config/wsl.yaml
├── wsl_main.py                         # WeakTrainerInterface → trainer by type / algorithm
├── scripts/
│   ├── trapo_v2.sh, trapo_v2_lib.sh, nl_run_olr_v2.sh, olr_v2_lib.sh, dept_root.sh
│   └── unsupervised_run.sh             # WIP: unsupervised
├── Research/
│   ├── Semi_RL/                        # ★ Primary: ttrl, intuitor, emrl, trapo, trapo_v2, geomin, co_reward
│   ├── Noisy_Label/                    # WIP: nl_ray_trainer*.py
│   ├── Unsupervised_RL/                # WIP: ray_trainer_unsupervised.py
│   └── OPD/                            # WIP
└── utils/
    ├── training_utils.py, data_utils.py
    ├── Semi_RL/
    ├── Unsupervised_RL/                # ttrl_utils (WIP pillar)
    └── analysis/                       # TraPO v2 probes & promote (semi)
```

---

## Shared configuration

Hydra: `config/wsl.yaml` (inherits verl `ppo_trainer`)

```yaml
trainer_type: semi          # semi | nl | unsupervised
wsl_algorithm: trapo_v2     # semi / nl only; see wsl_main.py
# Scripts set: algorithm.adv_estimator=grpo
```

**GRPO defaults in scripts:** `algorithm.adv_estimator=grpo`, `actor_rollout_ref.rollout.n=8` (group rollouts), typically `use_kl_loss=False`.

**TraPO v2 knobs (semi):** `trapo_stop_after_warmup`, `trapo_probe_warmup_steps`, `trapo_probe_interval`, `trapo_probe_scope`, `trapo_warmup_labeled_only_train`.

---

## How supervision enters the GRPO loop (all pillars)

```
Rollout (n responses / prompt)
    → Reward  ──┬─ Semi: TTRL / TraPO voting, Intuitor KL, EMRL entropy
                ├─ Noisy-label (WIP): relabel / filter suspected wrong GT
                └─ Unsupervised (WIP): certainty, ensemble, TTRL pseudo-GT

    → [Optional data filter, e.g. TraPO cosine / NL trajectory gate]

    → GRPO group-relative advantages → policy update
```

Intuitor / EMRL (semi): `utils/Semi_RL/ul_reward.py` → `apply_ul_seq_reward()`.

---

## Validation logging

`utils/training_utils.py` → `{log_dir}/valid/global_step_{N}.log`, including **`average`** (mean over val datasets). Used by the TraPO v2 iterative loop for early stopping.

---

## Adding a new method

1. Implement `Research/<Pillar>/ray_trainer_<name>.py` subclassing verl `RayPPOTrainer`.
2. Wire `_validate()` / `log_train_metrics()` via `training_utils`.
3. Register in `WeakTrainerInterface` (`wsl_main.py`) under the correct `trainer_type` / `wsl_algorithm`.
4. Add `scripts/<pillar>_run_<name>.sh` with `algorithm.adv_estimator=grpo`.

**Semi iterative promote:** edit `utils/analysis/stage2_selection.py` + `build_stage2_dataset.py`.

---

## Citation

Please cite the papers for the methods you use (TTRL, Intuitor, EMRL, TraPO, etc.) and [verl](https://github.com/volcengine/verl).

---

## License

Apache-2.0, consistent with verl. See repository `LICENSE`.
