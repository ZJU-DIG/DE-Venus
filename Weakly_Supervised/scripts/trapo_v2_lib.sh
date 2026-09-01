# TraPO v2 shared shell helpers. Source from trapo_v2.sh or legacy wrappers.
# Requires: SCRIPT_DIR set to Weakly_Supervised/scripts before trapo_v2_init.

trapo_v2_init() {
  local script_dir="${1:-${SCRIPT_DIR:-}}"
  if [[ -z "$script_dir" ]]; then
    echo "trapo_v2_init: missing script_dir" >&2
    return 1
  fi
  SCRIPT_DIR="$(cd "$script_dir" && pwd)"
  # shellcheck source=dept_root.sh
  source "$SCRIPT_DIR/dept_root.sh"
  dept_root_init "$SCRIPT_DIR"
  trapo_v2_apply_defaults
}

trapo_v2_apply_defaults() {
  ROOT=${ROOT:-/ossfs/workspace/aml0/484999/code}
  SAVE_ROOT=${SAVE_ROOT:-/ossfs/workspace/aitech_aidata/chuwei}
  MODEL_NAME=${MODEL_NAME:-Qwen3-8B-Base}
  EXP_NAME=${EXP_NAME:-semi_supervised_trapo_v2_iter}
  BASE_MODEL=${BASE_MODEL:-$SAVE_ROOT/models/Qwen/$MODEL_NAME}
  BASE_PARQUET=${BASE_PARQUET:-$DEPT_ROOT/data/dapo-math-14k/semi-supervised-zero-fixed.parquet}
  ITER_ROOT=${ITER_ROOT:-$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/${EXP_NAME}_loop}
  DATA_DIR=${DATA_DIR:-$DEPT_ROOT/data/dapo-math-14k}
  VAL_FILES=${VAL_FILES:-$DATA_DIR/valid.parquet}
  MODEL_PATH=${MODEL_PATH:-$BASE_MODEL}

  PASSRATE_MIN=${PASSRATE_MIN:-0.5}
  SLOPE_MIN=${SLOPE_MIN:-0.05}
  HIST_LEN=${HIST_LEN:-8}
  MAX_ITER=${MAX_ITER:-20}
  VAL_PATIENCE=${VAL_PATIENCE:-2}
  VAL_MIN_DELTA=${VAL_MIN_DELTA:-0.0}

  TRAPO_WARMUP_STEPS=${TRAPO_WARMUP_STEPS:-5}
  TRAPO_WARMUP_MIN_PROBES=${TRAPO_WARMUP_MIN_PROBES:-2}
  TRAPO_PROBE_INTERVAL=${TRAPO_PROBE_INTERVAL:-1}
  TRAPO_SCALE_WARMUP_BY_LABELED=${TRAPO_SCALE_WARMUP_BY_LABELED:-1}
  PROBE_DUMP_SUBDIR=${PROBE_DUMP_SUBDIR:-probe_dumps}

  FINAL_EXP_NAME=${FINAL_EXP_NAME:-semi_supervised_trapo_v2_final}
  FINAL_EPOCHS=${FINAL_EPOCHS:-10}
  FINAL_TEST_FREQ=${FINAL_TEST_FREQ:-8}
  FINAL_SAVE_FREQ=${FINAL_SAVE_FREQ:-50}
  RUN_FINAL_LONG_TRAIN=${RUN_FINAL_LONG_TRAIN:-1}

  PROBE_EXP_NAME=${PROBE_EXP_NAME:-semi_supervised_trapo_v2_probe_artifacts}
}

trapo_v2_promote() {
  local source_parquet=$1
  local probe_dir=$2
  local output_parquet=$3
  local promotion_iter=${4:-}

  local extra=()
  if [[ -n "$promotion_iter" ]]; then
    extra+=(--promotion-iteration "$promotion_iter")
  fi

  python3 -m Weakly_Supervised.utils.analysis.build_stage2_dataset \
    "$source_parquet" \
    "$probe_dir" \
    -o "$output_parquet" \
    --passrate-min "$PASSRATE_MIN" \
    --slope-min "$SLOPE_MIN" \
    --hist-len "$HIST_LEN" \
    "${extra[@]}"
}

# Promote for loop iter k (env: ITER, TRAIN_PARQUET, optional FINAL_PARQUET).
trapo_v2_promote_iter() {
  local iter=${ITER:?set ITER=0,1,...}
  TRAIN_PARQUET=${TRAIN_PARQUET:-$BASE_PARQUET}
  FINAL_PARQUET=${FINAL_PARQUET:-$ITER_ROOT/final_train.parquet}
  local probe_dir=$ITER_ROOT/iter_${iter}/log/probe_dumps
  local next_parquet=$ITER_ROOT/iter_${iter}/train_next.parquet

  if [[ ! -f "$probe_dir/all_probes_summary.jsonl" ]]; then
    echo "ERROR: missing $probe_dir/all_probes_summary.jsonl" >&2
    return 1
  fi

  echo "Promote iter=$iter"
  echo "  source=$TRAIN_PARQUET"
  echo "  probe=$probe_dir"
  echo "  output=$next_parquet"

  trapo_v2_promote "$TRAIN_PARQUET" "$probe_dir" "$next_parquet" "$iter"
  cp -f "$next_parquet" "$FINAL_PARQUET"
  local meta=$ITER_ROOT/iter_${iter}/train_next.advance_meta.json
  python3 -c "import json; print(json.dumps(json.load(open('$meta')), indent=2))"
  echo "Wrote $next_parquet and $FINAL_PARQUET"
}

trapo_v2_hydra_common() {
  # shellcheck disable=SC2086
  python3 -m Weakly_Supervised.wsl_main \
    wsl_algorithm=trapo_v2 \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.use_reward_loop=False \
    +algorithm.wsl="trapo_v2" \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    +trainer.train_mode="semi-supervised" \
    +trainer.topk=0.1 \
    +trainer.thres=0.4 \
    +trainer.update_db=True \
    +trainer.trapo_consistency_enabled=False \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "$@"
}

# Scale warmup steps / probe interval when labeled set grows between loop rounds.
# steps_k = round(steps_{k-1} * n_labeled_k / n_labeled_{k-1}); iter 0 uses env defaults.
trapo_v2_apply_scaled_probe_schedule() {
  local n_labeled_new=$1
  if [[ "${TRAPO_SCALE_WARMUP_BY_LABELED:-1}" != "1" ]]; then
    return 0
  fi

  if [[ -z "${N_LABELED_PREV:-}" ]]; then
    export N_LABELED_PREV=$n_labeled_new
    export TRAPO_WARMUP_STEPS="${TRAPO_WARMUP_STEPS_BASE:-$TRAPO_WARMUP_STEPS}"
    export TRAPO_PROBE_INTERVAL="${TRAPO_PROBE_INTERVAL_BASE:-$TRAPO_PROBE_INTERVAL}"
    echo "probe schedule iter baseline: n_labeled=$n_labeled_new warmup_steps=$TRAPO_WARMUP_STEPS probe_interval=$TRAPO_PROBE_INTERVAL"
    return 0
  fi

  local n_labeled_old=$N_LABELED_PREV
  local scaled
  scaled=$(python3 -c "
old_l = int('${n_labeled_old}')
new_l = int('${n_labeled_new}')
warmup_old = int('${TRAPO_WARMUP_STEPS}')
interval_old = int('${TRAPO_PROBE_INTERVAL}')
ratio = (new_l / old_l) if old_l > 0 else 1.0
w = max(1, round(warmup_old * ratio))
p = max(1, round(interval_old * ratio))
print(f'{w} {p} {ratio}')
")
  TRAPO_WARMUP_STEPS=$(echo "$scaled" | awk '{print $1}')
  TRAPO_PROBE_INTERVAL=$(echo "$scaled" | awk '{print $2}')
  local ratio
  ratio=$(echo "$scaled" | awk '{print $3}')
  export TRAPO_WARMUP_STEPS TRAPO_PROBE_INTERVAL N_LABELED_PREV=$n_labeled_new
  echo "probe schedule scaled: n_labeled $n_labeled_old -> $n_labeled_new ratio=$ratio warmup_steps=$TRAPO_WARMUP_STEPS probe_interval=$TRAPO_PROBE_INTERVAL"
}

# Optional one-shot scale for ``trapo_v2.sh train`` when TRAPO_N_LABELED_OLD/NEW are set.
trapo_v2_maybe_scale_probe_schedule_manual() {
  if [[ "${TRAPO_SCALE_WARMUP_BY_LABELED:-1}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${TRAPO_N_LABELED_OLD:-}" && -n "${TRAPO_N_LABELED_NEW:-}" ]]; then
    export N_LABELED_PREV=$TRAPO_N_LABELED_OLD
    export TRAPO_WARMUP_STEPS_BASE=${TRAPO_WARMUP_STEPS_BASE:-$TRAPO_WARMUP_STEPS}
    export TRAPO_PROBE_INTERVAL_BASE=${TRAPO_PROBE_INTERVAL_BASE:-$TRAPO_PROBE_INTERVAL}
    trapo_v2_apply_scaled_probe_schedule "$TRAPO_N_LABELED_NEW"
  fi
}

# resume_mode: each iter must reload BASE_MODEL at step 0, not iter_k/ckpt (auto resume).
trapo_v2_resume_mode_arg() {
  if [[ "${TRAPO_RESUME_CKPT:-0}" == "1" ]]; then
    echo "trainer.resume_mode=auto"
  else
    echo "trainer.resume_mode=disable"
  fi
}

# One loop round: labeled warmup + full probe, stop after warmup.
trapo_v2_run_train() {
  local iter=${ITER:-0}
  TRAIN_PARQUET=${TRAIN_PARQUET:-$BASE_PARQUET}
  # Always bind to iter_k (do not use ${CKPT_DIR:-...}: a stale export from iter 0 sticks in loop shell).
  CKPT_DIR=${TRAPO_CKPT_DIR:-$ITER_ROOT/iter_${iter}/ckpt}
  LOG_DIR=${TRAPO_LOG_DIR:-$ITER_ROOT/iter_${iter}/log}
  MODEL_PATH=${BASE_MODEL:-$MODEL_PATH}
  echo "trapo_v2_run_train: ITER=$iter CKPT_DIR=$CKPT_DIR LOG_DIR=$LOG_DIR"
  trapo_v2_maybe_scale_probe_schedule_manual
  local resume_arg
  resume_arg=$(trapo_v2_resume_mode_arg)

  trapo_v2_hydra_common \
    "$resume_arg" \
    trainer.project_name="${EXP_NAME}_iter_${iter}" \
    trainer.experiment_name="${EXP_NAME}_iter_${iter}" \
    trainer.default_local_dir="$CKPT_DIR" \
    +trainer.log_dir="$LOG_DIR" \
    +trainer.trapo_stop_after_warmup=True \
    +trainer.trapo_save_on_warmup_stop=True \
    +trainer.trapo_warmup_labeled_only_train=True \
    +trainer.trapo_probe_only_during_warmup=True \
    +trainer.trapo_probe_interval="$TRAPO_PROBE_INTERVAL" \
    +trainer.trapo_probe_scope=full \
    +trainer.trapo_probe_warmup_steps="$TRAPO_WARMUP_STEPS" \
    +trainer.trapo_probe_warmup_min_probes="$TRAPO_WARMUP_MIN_PROBES" \
    +trainer.trapo_probe_save_artifacts=True \
    +trainer.trapo_probe_dump_subdir="$PROBE_DUMP_SUBDIR" \
    trainer.save_freq=50 \
    trainer.test_freq=8 \
    trainer.total_epochs=10 \
    "$@"
}

# Single-shot warmup+probe (non-iter paths under SAVE_ROOT).
trapo_v2_run_probe() {
  TRAIN_PARQUET=${TRAIN_PARQUET:-$DEPT_ROOT/data/dapo-math-14k/semi-supervised-tiny-fixed.parquet}
  CKPT_DIR=${CKPT_DIR:-$SAVE_ROOT/DE-Venus/ckpt/$MODEL_NAME/$PROBE_EXP_NAME}
  LOG_DIR=${LOG_DIR:-$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/$PROBE_EXP_NAME}
  MODEL_PATH=${MODEL_PATH:-$SAVE_ROOT/models/Qwen/$MODEL_NAME}

  local resume_arg
  resume_arg=$(trapo_v2_resume_mode_arg)

  trapo_v2_hydra_common \
    "$resume_arg" \
    trainer.project_name="$PROBE_EXP_NAME" \
    trainer.experiment_name="$PROBE_EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    +trainer.log_dir="$LOG_DIR" \
    +trainer.trapo_stop_after_warmup=True \
    +trainer.trapo_save_on_warmup_stop=True \
    +trainer.trapo_warmup_labeled_only_train=True \
    +trainer.trapo_probe_only_during_warmup=True \
    +trainer.trapo_probe_interval=8 \
    +trainer.trapo_probe_scope=full \
    +trainer.trapo_probe_warmup_steps="$TRAPO_WARMUP_STEPS" \
    +trainer.trapo_probe_warmup_min_probes="$TRAPO_WARMUP_MIN_PROBES" \
    +trainer.trapo_probe_save_artifacts=True \
    +trainer.trapo_history_len=6 \
    +trainer.trapo_consistency_min=0.5 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    "$@"
}

# Long RL on expanded parquet (probes off).
trapo_v2_run_final() {
  TRAIN_PARQUET=${TRAIN_PARQUET:-$ITER_ROOT/final_train.parquet}
  INIT_MODEL=${INIT_MODEL:-$BASE_MODEL}
  CKPT_DIR=${CKPT_DIR:-$SAVE_ROOT/DE-Venus/ckpt/$MODEL_NAME/$FINAL_EXP_NAME}
  LOG_DIR=${LOG_DIR:-$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/$FINAL_EXP_NAME}
  MODEL_PATH=$INIT_MODEL

  local resume_arg
  resume_arg=$(trapo_v2_resume_mode_arg)

  trapo_v2_hydra_common \
    "$resume_arg" \
    trainer.project_name="$FINAL_EXP_NAME" \
    trainer.experiment_name="$FINAL_EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    +trainer.log_dir="$LOG_DIR" \
    +trainer.trapo_stop_after_warmup=False \
    +trainer.trapo_warmup_labeled_only_train=True \
    +trainer.trapo_probe_only_during_warmup=False \
    +trainer.trapo_probe_interval=0 \
    +trainer.trapo_probe_save_artifacts=False \
    +trainer.trapo_probe_warmup_steps="$TRAPO_WARMUP_STEPS" \
    +trainer.trapo_probe_warmup_min_probes=0 \
    trainer.save_freq="$FINAL_SAVE_FREQ" \
    trainer.test_freq="$FINAL_TEST_FREQ" \
    trainer.total_epochs="$FINAL_EPOCHS" \
    "$@"
}

trapo_v2_print_final_counts() {
  local parquet=$1
  python3 -c "
from Weakly_Supervised.utils.analysis.build_stage2_dataset import count_labeled_unlabeled_rows
import json
c = count_labeled_unlabeled_rows('${parquet}')
print('final dataset:', json.dumps(c, indent=2))
"
}

trapo_v2_help() {
  cat <<'EOF'
TraPO v2 launcher (unified entry for iterative semi-supervised pipeline).

Usage:
  bash Weakly_Supervised/scripts/trapo_v2.sh <command> [extra hydra overrides...]

Commands:
  pipeline   Phase 1 loop + optional Phase 2 long train
  loop       Iterative warmup/probe/promote only
  train      One loop round GRPO
  promote    Offline promote from probe_dumps
  probe      Single warmup+probe round
  final      Long train on final_train.parquet

Common env:
  ROOT, SAVE_ROOT, MODEL_NAME, BASE_MODEL, BASE_PARQUET, ITER_ROOT
  PASSRATE_MIN, SLOPE_MIN, HIST_LEN, MAX_ITER, VAL_PATIENCE, VAL_MIN_DELTA
  TRAPO_WARMUP_STEPS, TRAPO_WARMUP_MIN_PROBES, TRAPO_PROBE_INTERVAL, RUN_FINAL_LONG_TRAIN
  TRAPO_SCALE_WARMUP_BY_LABELED=0  (disable auto scale in loop)
  TRAPO_N_LABELED_OLD / TRAPO_N_LABELED_NEW  (manual scale for ``train``)
  TRAPO_RESUME_CKPT=1  (same iter crash recovery only; default off = always BASE_MODEL @ step 0)
  ITER, TRAIN_PARQUET  (train / promote)
  TRAPO_CKPT_DIR / TRAPO_LOG_DIR  (optional override; do not export CKPT_DIR across loop iters)

dapo-math-14ks:
  bash Weakly_Supervised/scripts/trapo_v2.sh pipeline
  RUN_FINAL_LONG_TRAIN=0 bash Weakly_Supervised/scripts/trapo_v2.sh pipeline
  ITER=0 TRAIN_PARQUET=$BASE_PARQUET bash Weakly_Supervised/scripts/trapo_v2.sh promote
  PASSRATE_MIN=0.5 TRAPO_WARMUP_STEPS=5 TRAPO_WARMUP_MIN_PROBES=2 bash Weakly_Supervised/scripts/trapo_v2.sh train

Promote without iter env: ``trapo_v2.sh promote`` uses SOURCE_PARQUET / PROBE_DIR / OUTPUT_PARQUET defaults.
EOF
}
