# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty,
    compute_response_mask,
    compute_advantage,
)
from ...utils.Semi_RL.major_voting import extract_pred_answers, find_majority, verify_answer
from ...utils.training_utils import validate_and_log, log_train_metrics


# ============================================================
# Semi-supervised co-reward utilities
# ============================================================


def compute_cross_pseudo_labels(
    data_ori: DataProto,
    data_aug: DataProto,
    tokenizer,
):
    """Replace ground_truth of unlabeled data with cross majority voting pseudo-labels.

    For unlabeled data: ori's pseudo-label comes from aug's majority vote, and vice versa.
    For labeled data: ground_truth is kept unchanged.

    This function modifies data_ori and data_aug in-place (their non_tensor_batch["reward_model"]["ground_truth"]).

    Args:
        data_ori: DataProto containing ori rollouts (N_prompts * n_rollouts).
        data_aug: DataProto containing aug rollouts (same size).
        tokenizer: tokenizer for decoding responses.

    Returns:
        metrics: dict with pseudo-label statistics.
    """
    assert len(data_ori) == len(data_aug), (
        f"ori and aug must have the same length, got {len(data_ori)} vs {len(data_aug)}"
    )

    N = len(data_ori)

    # ---- Step 1: Separate labeled and unlabeled indices ----
    unlabeled_indices = []
    labeled_count = 0
    for i in range(N):
        extra_info = data_ori[i].non_tensor_batch.get("extra_info", {})
        if extra_info and extra_info.get("labeled", True) is False:
            unlabeled_indices.append(i)
        else:
            labeled_count += 1

    # ---- Step 2: For unlabeled data, compute cross pseudo-labels ----
    # Extract answers grouped by uid for unlabeled data only
    ori_uid2indices = defaultdict(list)
    aug_uid2indices = defaultdict(list)
    for i in unlabeled_indices:
        uid = data_ori[i].non_tensor_batch["uid"]
        ori_uid2indices[uid].append(i)
        uid_aug = data_aug[i].non_tensor_batch["uid"]
        aug_uid2indices[uid_aug].append(i)

    # Extract predicted answers for unlabeled data
    ori_uid2answers = {}
    for uid, indices in ori_uid2indices.items():
        ori_uid2answers[uid] = extract_pred_answers(data_ori, indices, tokenizer)

    aug_uid2answers = {}
    for uid, indices in aug_uid2indices.items():
        aug_uid2answers[uid] = extract_pred_answers(data_aug, indices, tokenizer)

    # Cross majority voting: ori's pseudo-label from aug, aug's pseudo-label from ori
    ori_uid2pseudo = {}
    for uid, ans_list in aug_uid2answers.items():
        major = find_majority(ans_list)
        ori_uid2pseudo[uid] = major if major is not None else "None"

    aug_uid2pseudo = {}
    for uid, ans_list in ori_uid2answers.items():
        major = find_majority(ans_list)
        aug_uid2pseudo[uid] = major if major is not None else "None"

    # ---- Step 3: Replace ground_truth with pseudo-labels for unlabeled data ----
    rel_pseudo_num = 0
    rel_pseudo_correct_num = 0

    for uid, indices in ori_uid2indices.items():
        pseudo_label = ori_uid2pseudo.get(uid, "None")
        gold_answer = data_ori[indices[0]].non_tensor_batch["reward_model"]["ground_truth"]
        for idx in indices:
            data_ori[idx].non_tensor_batch["reward_model"]["ground_truth"] = pseudo_label
        if verify_answer(pseudo_label, gold_answer):
            rel_pseudo_correct_num += 1
        rel_pseudo_num += 1

    for uid, indices in aug_uid2indices.items():
        pseudo_label = aug_uid2pseudo.get(uid, "None")
        gold_answer = data_aug[indices[0]].non_tensor_batch["reward_model"]["ground_truth"]
        for idx in indices:
            data_aug[idx].non_tensor_batch["reward_model"]["ground_truth"] = pseudo_label
        if verify_answer(pseudo_label, gold_answer):
            rel_pseudo_correct_num += 1
        rel_pseudo_num += 1

    metrics = {
        "co_reward/labeled_count": labeled_count,
        "co_reward/unlabeled_count": len(unlabeled_indices),
        "co_reward/total_rollouts": N * 2,
        "semi/rel_pseudo_num": rel_pseudo_num,
        "semi/rel_pseudo_correct_num": rel_pseudo_correct_num,
        "semi/rel_pseudo_precision": rel_pseudo_correct_num / rel_pseudo_num if rel_pseudo_num > 0 else 0,
    }

    return metrics


# ============================================================
# Trainer
# ============================================================

class RayPPOTrainerCoReward(RayPPOTrainer):
    """Semi-supervised co-rewarding trainer using cross majority voting between ori and aug questions.

    For labeled data: uses ground-truth answer to compute reward via reward_fn.
    For unlabeled data: uses cross majority voting pseudo-labels
        (ori uses aug's majority vote, aug uses ori's majority vote).
    Both ori and aug contribute to the loss.

    Implementation:
    1. Each step, the dataloader yields {"ori": batch_dict, "aug": batch_dict}.
    2. Generate rollouts for ori and aug separately.
    3. For unlabeled data, replace ground_truth with cross pseudo-labels.
    4. Compute reward uniformly via reward_fn (labeled uses real GT, unlabeled uses pseudo GT).
    5. Merge ori + aug into a single batch for standard PPO update.
    """

    def _validate(self):
        metric_dict = super()._validate()
        return validate_and_log(self.config, self.global_steps, metric_dict)

    def fit(self):
        """
        The training loop of Co-rewarding PPO.
        Based on TTRL's fit() with modifications for dual-stream rollout
        and cross majority voting reward.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        n_rollouts = self.config.actor_rollout_ref.rollout.n

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # ============================================================
                # Co-rewarding: parse paired batch
                # ============================================================
                batch_ori: DataProto = DataProto.from_single_dict(batch_dict["ori"])
                batch_aug: DataProto = DataProto.from_single_dict(batch_dict["aug"])

                batch_ori.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch_aug.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # Assign shared uids so cross voting can match ori <-> aug
                shared_uids = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch_ori.batch))], dtype=object
                )
                batch_ori.non_tensor_batch["uid"] = shared_uids
                batch_aug.non_tensor_batch["uid"] = shared_uids.copy()

                gen_batch_ori = self._get_gen_batch(batch_ori)
                gen_batch_aug = self._get_gen_batch(batch_aug)

                gen_batch_ori.meta_info["global_steps"] = self.global_steps
                gen_batch_aug.meta_info["global_steps"] = self.global_steps

                gen_batch_output_ori = gen_batch_ori.repeat(repeat_times=n_rollouts, interleave=True)
                gen_batch_output_aug = gen_batch_aug.repeat(repeat_times=n_rollouts, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # ============================================================
                    # Generate rollouts for ori and aug
                    # ============================================================
                    with marked_timer("gen_ori", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output_ori = self.actor_rollout_wg.generate_sequences(gen_batch_output_ori)
                        else:
                            gen_batch_output_ori = self.async_rollout_manager.generate_sequences(gen_batch_output_ori)
                        timing_raw.update(gen_batch_output_ori.meta_info.get("timing", {}))
                        gen_batch_output_ori.meta_info.pop("timing", None)

                    with marked_timer("gen_aug", timing_raw, color="orange"):
                        if not self.async_rollout_mode:
                            gen_batch_output_aug = self.actor_rollout_wg.generate_sequences(gen_batch_output_aug)
                        else:
                            gen_batch_output_aug = self.async_rollout_manager.generate_sequences(gen_batch_output_aug)
                        timing_raw.update(gen_batch_output_aug.meta_info.get("timing", {}))
                        gen_batch_output_aug.meta_info.pop("timing", None)

                    # Repeat and union
                    batch_ori = batch_ori.repeat(repeat_times=n_rollouts, interleave=True)
                    batch_ori = batch_ori.union(gen_batch_output_ori)
                    if "response_mask" not in batch_ori.batch.keys():
                        batch_ori.batch["response_mask"] = compute_response_mask(batch_ori)

                    batch_aug = batch_aug.repeat(repeat_times=n_rollouts, interleave=True)
                    batch_aug = batch_aug.union(gen_batch_output_aug)
                    if "response_mask" not in batch_aug.batch.keys():
                        batch_aug.batch["response_mask"] = compute_response_mask(batch_aug)

                    # ============================================================
                    # Semi-supervised: cross pseudo-labels for unlabeled data
                    # ============================================================
                    if self.config.trainer.train_mode != "supervised":
                        with marked_timer("cross_pseudo", timing_raw, color="yellow"):
                            pseudo_metrics = compute_cross_pseudo_labels(
                                batch_ori, batch_aug, self.tokenizer
                            )
                            metrics.update(pseudo_metrics)

                    # ============================================================
                    # Merge ori + aug into a single batch
                    # ============================================================
                    # We need to give aug different uids so GRPO advantage
                    # normalization treats ori and aug as separate prompt groups.
                    for i in range(len(batch_aug)):
                        batch_aug[i].non_tensor_batch["uid"] = batch_aug[i].non_tensor_batch["uid"] + "_aug"

                    batch = DataProto.concat([batch_ori, batch_aug])

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Balance batch across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # ============================================================
                    # Compute old_log_probs
                    # ============================================================
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # ============================================================
                    # Reward → Advantage → Update (standard PPO flow)
                    # ============================================================
                    # Compute reward via reward_fn
                    # For labeled data: uses original ground_truth
                    # For unlabeled data: uses cross pseudo-labels (already replaced above)
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # apply KL penalty if configured
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Rollout correction (decoupled mode only)
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # checkpoint
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                # Write training information to file
                log_train_metrics(self.config, self.global_steps, metrics)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
