# OLR v2 shared helpers (TraPO v2-style iterative noisy-label relabel loop).

nl_olr_v2_init() {
  local script_dir="${1:-${SCRIPT_DIR:-}}"
  if [[ -z "$script_dir" ]]; then
    echo "nl_olr_v2_init: missing script_dir" >&2
    return 1
  fi
  SCRIPT_DIR="$(cd "$script_dir" && pwd)"
  # shellcheck source=dept_root.sh
  source "$SCRIPT_DIR/dept_root.sh"
  dept_root_init "$SCRIPT_DIR"
  nl_olr_v2_apply_defaults
}

nl_olr_v2_apply_defaults() {
  ROOT=${ROOT:-/ossfs/workspace/aml0/484999/code}
  SAVE_ROOT=${SAVE_ROOT:-/ossfs/workspace/aitech_aidata/chuwei}
  MODEL_NAME=${MODEL_NAME:-Qwen3-8B-Base}
  EXP_NAME=${EXP_NAME:-nl_olr_v2_iter}
  BASE_MODEL=${BASE_MODEL:-$SAVE_ROOT/models/Qwen/$MODEL_NAME}
  BASE_PARQUET=${BASE_PARQUET:-$DEPT_ROOT/data/dapo-math-14k/noisy_label_zero.parquet}
  ITER_ROOT=${ITER_ROOT:-$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/${EXP_NAME}_loop}
  DATA_DIR=${DATA_DIR:-$DEPT_ROOT/data/dapo-math-14k}
  VAL_FILES=${VAL_FILES:-$DEPT_ROOT/data/deepmath-hard/valid.parquet}
  MODEL_PATH=${MODEL_PATH:-$BASE_MODEL}

  PASSRATE_MIN=${PASSRATE_MIN:-0.35}
  SLOPE_MIN=${SLOPE_MIN:-0.025}
  SLOPE_MARGIN=${SLOPE_MARGIN:-0.025}
  PASSRATE_MARGIN=${PASSRATE_MARGIN:-}
  HIST_LEN=${HIST_LEN:-8}
  MAX_ITER=${MAX_ITER:-20}
  VAL_PATIENCE=${VAL_PATIENCE:-2}
  VAL_MIN_DELTA=${VAL_MIN_DELTA:-0.0}

  NL_OLR_WARMUP_STEPS=${NL_OLR_WARMUP_STEPS:-5}
  NL_OLR_WARMUP_MIN_PROBES=${NL_OLR_WARMUP_MIN_PROBES:-3}
  NL_OLR_PROBE_INTERVAL=${NL_OLR_PROBE_INTERVAL:-1}
  NL_OLR_SCALE_WARMUP_BY_SIZE=${NL_OLR_SCALE_WARMUP_BY_SIZE:-1}
  PROBE_DUMP_SUBDIR=${PROBE_DUMP_SUBDIR:-probe_dumps}

  FINAL_EXP_NAME=${FINAL_EXP_NAME:-nl_olr_v2_final}
  FINAL_EPOCHS=${FINAL_EPOCHS:-10}
  FINAL_TEST_FREQ=${FINAL_TEST_FREQ:-8}
  FINAL_SAVE_FREQ=${FINAL_SAVE_FREQ:-50}
  RUN_FINAL_LONG_TRAIN=${RUN_FINAL_LONG_TRAIN:-1}
}

nl_olr_v2_relabel() {
  local source_parquet=$1
  local probe_dir=$2
  local output_parquet=$3
  local relabel_iter=${4:-}
  local extra=()
  if [[ -n "$relabel_iter" ]]; then
    extra+=(--relabel-iteration "$relabel_iter")
  fi
  if [[ -n "${PASSRATE_MARGIN:-}" ]]; then
    extra+=(--passrate-margin "$PASSRATE_MARGIN")
  fi
  python3 -m Weakly_Supervised.utils.analysis.build_noisy_label_dataset \
    "$source_parquet" \
    "$probe_dir" \
    -o "$output_parquet" \
    --passrate-min "$PASSRATE_MIN" \
    --slope-min "$SLOPE_MIN" \
    --slope-margin "$SLOPE_MARGIN" \
    --hist-len "$HIST_LEN" \
    "${extra[@]}"
}

nl_olr_v2_relabel_iter() {
  local iter=${ITER:?set ITER=0,1,...}
  TRAIN_PARQUET=${TRAIN_PARQUET:-$BASE_PARQUET}
  FINAL_PARQUET=${FINAL_PARQUET:-$ITER_ROOT/final_train.parquet}
  local probe_dir=$ITER_ROOT/iter_${iter}/log/probe_dumps
  local next_parquet=$ITER_ROOT/iter_${iter}/train_next.parquet

  if [[ ! -f "$probe_dir/all_probes_summary.jsonl" ]]; then
    echo "ERROR: missing $probe_dir/all_probes_summary.jsonl" >&2
    return 1
  fi

  echo "Relabel iter=$iter"
  echo "  source=$TRAIN_PARQUET"
  echo "  probe=$probe_dir"
  echo "  output=$next_parquet"

  nl_olr_v2_relabel "$TRAIN_PARQUET" "$probe_dir" "$next_parquet" "$iter"
  cp -f "$next_parquet" "$FINAL_PARQUET"
  local meta=$ITER_ROOT/iter_${iter}/train_next.advance_meta.json
  python3 -c "import json; print(json.dumps(json.load(open('$meta')), indent=2))"
  echo "Wrote $next_parquet and $FINAL_PARQUET"
}

nl_olr_v2_resume_mode_arg() {
  if [[ "${NL_OLR_RESUME_CKPT:-0}" == "1" ]]; then
    echo "trainer.resume_mode=auto"
  else
    echo "trainer.resume_mode=disable"
  fi
}

nl_olr_v2_apply_scaled_probe_schedule() {
  local n_rows_new=$1
  if [[ "${NL_OLR_SCALE_WARMUP_BY_SIZE:-1}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${N_ROWS_PREV:-}" ]]; then
    export N_ROWS_PREV=$n_rows_new
    export NL_OLR_WARMUP_STEPS="${NL_OLR_WARMUP_STEPS_BASE:-$NL_OLR_WARMUP_STEPS}"
    export NL_OLR_PROBE_INTERVAL="${NL_OLR_PROBE_INTERVAL_BASE:-$NL_OLR_PROBE_INTERVAL}"
    echo "probe schedule baseline: n_rows=$n_rows_new warmup=$NL_OLR_WARMUP_STEPS interval=$NL_OLR_PROBE_INTERVAL"
    return 0
  fi
  local n_old=$N_ROWS_PREV
  local scaled
  scaled=$(python3 -c "
old_n = int('${n_old}')
new_n = int('${n_rows_new}')
w = int('${NL_OLR_WARMUP_STEPS}')
p = int('${NL_OLR_PROBE_INTERVAL}')
ratio = (new_n / old_n) if old_n > 0 else 1.0
print(f'{max(1, round(w * ratio))} {max(1, round(p * ratio))} {ratio}')
")
  NL_OLR_WARMUP_STEPS=$(echo "$scaled" | awk '{print $1}')
  NL_OLR_PROBE_INTERVAL=$(echo "$scaled" | awk '{print $2}')
  export NL_OLR_WARMUP_STEPS NL_OLR_PROBE_INTERVAL N_ROWS_PREV=$n_rows_new
  echo "probe schedule scaled: n_rows $n_old -> $n_rows_new ratio=$(echo "$scaled" | awk '{print $3}') warmup=$NL_OLR_WARMUP_STEPS interval=$NL_OLR_PROBE_INTERVAL"
}

nl_olr_v2_hydra_common() {
  python3 -m Weakly_Supervised.wsl_main \
    trainer_type=nl \
    wsl_algorithm=nl_olr_v2 \
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
    +algorithm.wsl="nl_olr_v2" \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    +trainer.train_mode=weak \
    +trainer.slope_tres="${SLOPE_TRES:-0.05}" \
    +trainer.use_trapo=True \
    +trainer.p_1_to_0="${P_1_TO_0:-0.0}" \
    +trainer.p_0_to_1="${P_0_TO_1:-0.0}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "$@"
}

nl_olr_v2_run_train() {
  local iter=${ITER:-0}
  TRAIN_PARQUET=${TRAIN_PARQUET:-$BASE_PARQUET}
  CKPT_DIR=${NL_OLR_CKPT_DIR:-$ITER_ROOT/iter_${iter}/ckpt}
  LOG_DIR=${NL_OLR_LOG_DIR:-$ITER_ROOT/iter_${iter}/log}
  MODEL_PATH=${BASE_MODEL:-$MODEL_PATH}
  echo "nl_olr_v2_run_train: ITER=$iter CKPT_DIR=$CKPT_DIR LOG_DIR=$LOG_DIR"
  local resume_arg
  resume_arg=$(nl_olr_v2_resume_mode_arg)
  # Same TraPO v2 trainer knobs; NL differs only in offline relabel (not promote).
  nl_olr_v2_hydra_common \
    "$resume_arg" \
    trainer.project_name="${EXP_NAME}_iter_${iter}" \
    trainer.experiment_name="${EXP_NAME}_iter_${iter}" \
    trainer.default_local_dir="$CKPT_DIR" \
    +trainer.log_dir="$LOG_DIR" \
    +trainer.trapo_stop_after_warmup=True \
    +trainer.trapo_save_on_warmup_stop=True \
    +trainer.trapo_warmup_labeled_only_train=True \
    +trainer.trapo_probe_only_during_warmup=True \
    +trainer.trapo_probe_interval="$NL_OLR_PROBE_INTERVAL" \
    +trainer.trapo_probe_scope=full \
    +trainer.trapo_probe_warmup_steps="$NL_OLR_WARMUP_STEPS" \
    +trainer.trapo_probe_warmup_min_probes="$NL_OLR_WARMUP_MIN_PROBES" \
    +trainer.trapo_probe_save_artifacts=True \
    +trainer.trapo_probe_dump_subdir="$PROBE_DUMP_SUBDIR" \
    trainer.save_freq=50 \
    trainer.test_freq=8 \
    trainer.total_epochs=10 \
    "$@"
}

nl_olr_v2_run_final() {
  TRAIN_PARQUET=${TRAIN_PARQUET:-$ITER_ROOT/final_train.parquet}
  INIT_MODEL=${INIT_MODEL:-$BASE_MODEL}
  CKPT_DIR=${CKPT_DIR:-$SAVE_ROOT/DE-Venus/ckpt/$MODEL_NAME/$FINAL_EXP_NAME}
  LOG_DIR=${LOG_DIR:-$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/$FINAL_EXP_NAME}
  MODEL_PATH=$INIT_MODEL
  local resume_arg
  resume_arg=$(nl_olr_v2_resume_mode_arg)
  nl_olr_v2_hydra_common \
    "$resume_arg" \
    trainer.project_name="$FINAL_EXP_NAME" \
    trainer.experiment_name="$FINAL_EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    +trainer.log_dir="$LOG_DIR" \
    +trainer.trapo_stop_after_warmup=False \
    +trainer.trapo_probe_interval=0 \
    +trainer.trapo_probe_save_artifacts=False \
    trainer.save_freq="$FINAL_SAVE_FREQ" \
    trainer.test_freq="$FINAL_TEST_FREQ" \
    trainer.total_epochs="$FINAL_EPOCHS" \
    "$@"
}

nl_olr_v2_print_final_counts() {
  python3 -c "
from Weakly_Supervised.utils.analysis.build_stage2_dataset import count_labeled_unlabeled_rows
import json
c = count_labeled_unlabeled_rows('${1}')
print('final dataset:', json.dumps(c, indent=2))
"
}

nl_olr_v2_help() {
  cat <<'EOF'
NL-OLR v2 launcher (TraPO v2 trainer + offline relabel loop).

Training: RayPPOTrainerTraPOv2 via wsl_algorithm=nl_olr_v2; all rows must be labeled=True.
Offline: build_noisy_label_dataset (relabel GT when probe majority != current GT).

Usage:
  bash Weakly_Supervised/scripts/nl_run_olr_v2.sh <command> [hydra overrides...]

Commands:
  pipeline   loop + optional final long train
  loop       iterative warmup/probe/relabel
  train      one round
  relabel    offline refine parquet from probe_dumps
  final      long train on final_train.parquet

Data default: data/dapo-math-14k/noisy_label.parquet (set extra_info.labeled=true on every row)
EOF
}
