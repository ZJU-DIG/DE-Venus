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
Unsupervised / pseudo-reward PPO trainer for Weakly_Supervised.

Inherits verl ``RayPPOTrainer`` (same worker wiring as TTRL/TraPO). Only ``fit`` is
overridden to inject TTRL voting, certainty rewards, ensemble majority, and self-verify.

Configure via Hydra, e.g. ``trainer_type: unsupervised`` and ``unsupervised_reward`` / ``ttrl`` blocks.
"""

import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from ...utils.training_utils import log_train_metrics, validate_and_log


class RayPPOTrainerUnsupervised(RayPPOTrainer):
    """PPO trainer with optional unsupervised reward hooks (certainty, ensemble, self-verify, TTRL-style)."""

    def _validate(self):
        metric_dict = super()._validate()
        return validate_and_log(self.config, self.global_steps, metric_dict)

    def fit(self):
        """Training loop with unsupervised_reward / ttrl branches (generation + reward shaping)."""
        logger = self._setup_tracking_logger()

        self.global_steps = 0

        self._load_checkpoint()

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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        )

        profile_steps = getattr(self.config.trainer, "profile_steps", None)
        if profile_steps is None:
            profile_steps = getattr(self.config.global_profiler, "steps", None)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                do_profile = (
                    self.global_steps in profile_steps if profile_steps is not None else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                if repeat_sampling_sglang_grpo:
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                # Samples per prompt used for batch.repeat / union / advantage grouping.
                num_repeat = self.config.actor_rollout_ref.rollout.n

                with marked_timer("step", timing_raw):
                    unsupervised_reward = OmegaConf.select(self.config, "unsupervised_reward")
                    # Lazy imports: Weakly_Supervised.utils.Unsupervised_RL.ttrl_utils
                    from ...utils.Unsupervised_RL import ttrl_utils as ul_utils

                    with marked_timer("gen", timing_raw, color="red"):
                        if self.config.get("ttrl", {}).get("enable", False):
                            n_votes_per_prompt = self.config.ttrl.n_votes_per_prompt
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=n_votes_per_prompt, interleave=True
                            )
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            assert len(gen_batch_output) == len(batch) * n_votes_per_prompt

                            batch = ul_utils.apply_ttrl_gt(
                                batch, gen_batch_output, n_votes_per_prompt, self.tokenizer
                            )
                            gen_batch_output = ul_utils.select_top_k_per_prompt(
                                gen_batch_output,
                                n_votes_per_prompt,
                                self.config.ttrl.n_samples_per_prompt,
                            )

                            assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_samples_per_prompt
                            num_repeat = self.config.ttrl.n_samples_per_prompt
                        elif (
                            unsupervised_reward
                            and unsupervised_reward.get("enable", False)
                            and unsupervised_reward.get("type", None) == "ensemble"
                        ):
                            n_samples_per_prompt = self.config.actor_rollout_ref.rollout.n
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=n_samples_per_prompt, interleave=True
                            )
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            assert len(gen_batch_output) == len(batch) * n_samples_per_prompt

                            batch = ul_utils.apply_ttrl_gt(batch, gen_batch_output, n_samples_per_prompt, self.tokenizer)

                            assert len(gen_batch_output) == len(batch) * n_samples_per_prompt
                            num_repeat = n_samples_per_prompt
                        else:
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                    gen_baseline_batch
                                )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if not repeat_sampling_sglang_grpo:
                        batch = batch.repeat(repeat_times=num_repeat, interleave=True)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        need_self_certainty = (
                            unsupervised_reward
                            and unsupervised_reward.get("enable", False)
                            and unsupervised_reward.get("type", None) == "certainty"
                            and unsupervised_reward.get("estimator", None) == "self_certainty"
                        )
                        if need_self_certainty:
                            batch.meta_info["calculate_self_certainty"] = True
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        if not need_self_certainty:
                            old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        pseudo_reward_tensor = None
                        pseudo_reward_extra_infos_dict = None
                        if unsupervised_reward and unsupervised_reward.get("enable", False):
                            if unsupervised_reward.get("type", None) == "certainty":
                                if "response_mask" not in batch.batch.keys():
                                    batch.batch["response_mask"] = compute_response_mask(batch)
                                pseudo_reward_tensor, pseudo_reward_extra_infos_dict = ul_utils.compute_certainty_reward(
                                    batch, unsupervised_reward.get("estimator", None)
                                )

                            if unsupervised_reward.get("type", None) == "external":
                                if unsupervised_reward.get("estimator", None) == "self_verify":
                                    pseudo_reward_tensor, pseudo_reward_extra_infos_dict = ul_utils.apply_self_verify(
                                        batch,
                                        self.tokenizer,
                                        self.actor_rollout_wg,
                                        verify_prompt=None,
                                        async_rollout_mode=self.async_rollout_mode,
                                        async_rollout_manager=self.async_rollout_manager,
                                        max_prompt_length=self.config.actor_rollout_ref.rollout.prompt_length,
                                    )

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                        if "rollout_log_probs" in batch.batch.keys():
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        batch.batch["token_level_scores"] = reward_tensor

                        if unsupervised_reward and unsupervised_reward.get("enable", False):
                            if unsupervised_reward.get("type", None) == "certainty" and pseudo_reward_tensor is not None:
                                batch.batch["token_level_scores"] = ul_utils.as_plain_tensor(pseudo_reward_tensor)
                                if pseudo_reward_extra_infos_dict:
                                    for key, val in pseudo_reward_extra_infos_dict.items():
                                        if torch.is_tensor(val):
                                            val = val.detach().cpu().numpy()
                                        batch.non_tensor_batch[key] = np.asarray(val)
                            if (
                                unsupervised_reward.get("type", None) == "external"
                                and unsupervised_reward.get("estimator", None) == "self_verify"
                                and pseudo_reward_tensor is not None
                            ):
                                batch.batch["token_level_scores"] = ul_utils.as_plain_tensor(pseudo_reward_tensor)
                                if pseudo_reward_extra_infos_dict:
                                    batch.non_tensor_batch.update(
                                        {k: np.array(v) for k, v in pseudo_reward_extra_infos_dict.items()}
                                    )

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        token_level_rewards = ul_utils.as_plain_tensor(batch.batch["token_level_rewards"])
                        if token_level_rewards.dim() != 2:
                            token_level_rewards = ul_utils.normalize_token_level_reward_tensor(
                                token_level_rewards,
                                batch.batch["response_mask"],
                                batch.batch["responses"],
                            )
                        batch.batch["token_level_rewards"] = token_level_rewards
                        batch.batch["token_level_scores"] = token_level_rewards

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.get("ttrl", {}).get("enable", False):
                        batch = ul_utils.apply_original_gt(batch)
                        reward_tensor_original, _ = self._compute_or_extract_reward(
                            batch, reward_fn=self.reward_fn, return_dict=False
                        )
                        batch.batch["token_level_scores_original"] = reward_tensor_original
                        ttrl_metrics = ul_utils.compute_ttrl_metrics(batch, self.config.ttrl.n_samples_per_prompt)
                        for key, value in ttrl_metrics.items():
                            metrics.update({f"train/{key}": value})

                    if unsupervised_reward and unsupervised_reward.get("enable", False):
                        if unsupervised_reward.get("type", None) == "ensemble":
                            batch = ul_utils.apply_original_gt(batch)
                            reward_tensor_original, _ = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )
                            batch.batch["token_level_scores_original"] = reward_tensor_original
                            n_samples_per_prompt = self.config.actor_rollout_ref.rollout.n
                            ensemble_metrics = ul_utils.compute_ttrl_metrics(batch, n=n_samples_per_prompt)
                            for key, value in ensemble_metrics.items():
                                metrics.update({f"train/ensemble/{key}": value})

                        if unsupervised_reward.get("type", None) == "certainty":
                            batch.batch["token_level_scores_original"] = reward_tensor
                            if "score" not in batch.non_tensor_batch and "acc" not in batch.non_tensor_batch:
                                batch.non_tensor_batch["score"] = (
                                    reward_tensor.sum(dim=-1).detach().cpu().numpy()
                                )
                            certainty_metrics = ul_utils.compute_certainty_metrics(batch, num_repeat)
                            for key, value in certainty_metrics.items():
                                metrics.update({f"train/certainty/{key}": value})

                        if (
                            unsupervised_reward.get("type", None) == "external"
                            and unsupervised_reward.get("estimator", None) == "self_verify"
                        ):
                            batch.batch["token_level_scores_original"] = reward_tensor
                            self_verify_metrics = ul_utils.compute_self_verify_metrics(batch)
                            for key, value in self_verify_metrics.items():
                                metrics.update({f"train/self_verify/{key}": value})

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

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

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                log_train_metrics(self.config, self.global_steps, metrics)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

    def _setup_tracking_logger(self):
        from verl.utils.tracking import Tracking

        return Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
