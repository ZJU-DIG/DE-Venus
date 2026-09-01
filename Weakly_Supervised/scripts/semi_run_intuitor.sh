set -x

ROOT=..
SAVE_ROOT=..

MODEL_NAME=Qwen3-8B-Base
TRAIN_MODE=semi-supervised
EXP_NAME=${TRAIN_MODE}_self_certainty

DATA_DIR=$ROOT/DE-Venus/data/deepmath-hard
CKPT_DIR=$SAVE_ROOT/DE-Venus/ckpt/$MODEL_NAME/$EXP_NAME
LOG_DIR=$SAVE_ROOT/DE-Venus/log/$MODEL_NAME/$EXP_NAME
MODEL_PATH=$ROOT/models/$MODEL_NAME
REWARD_DIR=$ROOT/DE-Venus/Weakly_Supervised/utils/Semi_RL/reward_score
REWARD_PATH=$REWARD_DIR/math_verify/__init__.py

python3 -m Weakly_Supervised.wsl_main \
    wsl_algorithm=intuitor \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/$TRAIN_MODE.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
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
    reward_model.use_reward_loop=True \
    reward_model.reward_manager="remote" \
    reward_model.num_workers=2 \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$CKPT_DIR \
    +trainer.log_dir=$LOG_DIR \
    +trainer.train_mode=$TRAIN_MODE \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 $@
