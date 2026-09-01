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
from collections import defaultdict, deque
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
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty, 
    compute_response_mask, 
    compute_advantage
)
from ...utils.Semi_RL.major_voting import extract_pred_answers_mv as extract_pred_answers
from ...utils.Semi_RL.major_voting import verify_answer_mv as verify_answer
from ...utils.Semi_RL.major_voting import find_majority_mv as find_majority
from ...utils.Semi_RL.vmf_modeling import VMFEstimator
from ...utils.training_utils import validate_and_log, log_train_metrics, compute_semi_metrics
from ...utils.data_utils import get_labeled_unlabeled_indices
from sklearn.mixture import GaussianMixture


class RayPPOTrainerOurs(RayPPOTrainer):
    """PPO trainer with vMF-based semi-supervised learning and advantage reweighting."""

    # ── lifecycle: validate / checkpoint ────────────────────────────────

    def _validate(self):
        metric_dict = super()._validate()
        return validate_and_log(self.config, self.global_steps, metric_dict)

    def _save_checkpoint(self):
        super()._save_checkpoint()

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        # save vMF estimators (one per layer) if they exist
        if hasattr(self, "vmf_estimators") and self.vmf_estimators:
            vmf_state = []
            for est in self.vmf_estimators:
                vmf_state.append({
                    "class_num": est.class_num,
                    "feature_num": est.feature_num,
                    "Ave": est.Ave.cpu(),
                    "Amount": est.Amount.cpu(),
                    "kappa": est.kappa.cpu(),
                    "logc": est.logc.cpu(),
                })
            vmf_path = os.path.join(local_global_step_folder, "vmf_estimators.pt")
            torch.save(vmf_state, vmf_path)
            print(f"Saved {len(vmf_state)} vMF estimators to {vmf_path}")

        # save discriminability history for layer selection continuity across resumes
        if hasattr(self, "_disc_history_sum") and self._disc_history_count > 0:
            disc_state = {
                "disc_history_sum": self._disc_history_sum,
                "disc_history_count": self._disc_history_count,
            }
            disc_path = os.path.join(local_global_step_folder, "disc_history.pt")
            torch.save(disc_state, disc_path)

    def _load_checkpoint(self):
        super()._load_checkpoint()

        if self.config.trainer.resume_mode == "disable":
            return 0

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        
        # load vMF estimators if available
        vmf_path = os.path.join(global_step_folder, "vmf_estimators.pt")
        if os.path.exists(vmf_path):
            vmf_state = torch.load(vmf_path, weights_only=False)
            self.vmf_estimators = []
            for state in vmf_state:
                est = VMFEstimator(feature_num=state["feature_num"], class_num=state["class_num"])
                est.Ave = state["Ave"]
                est.Amount = state["Amount"]
                est.kappa = state["kappa"]
                est.logc = state["logc"]
                self.vmf_estimators.append(est)
            print(f"Loaded {len(self.vmf_estimators)} vMF estimators from {vmf_path}")

        # load discriminability history if available
        disc_path = os.path.join(global_step_folder, "disc_history.pt")
        if os.path.exists(disc_path):
            disc_state = torch.load(disc_path, weights_only=False)
            self._disc_history_sum = disc_state["disc_history_sum"]
            self._disc_history_count = disc_state["disc_history_count"]
            print(f"Loaded disc history ({self._disc_history_count} steps) from {disc_path}")

    # ── vMF distribution modeling & boundary disambiguation ───

    def _distribution_modeling(self, data: DataProto):
        """Fit per-layer vMF estimators (class 0=incorrect, 1=correct) on labeled data."""
        
        hidden_states = data.batch["last_token_hidden_states"]  # (bs, num_layers, hidden_size)
        num_layers = hidden_states.shape[1]
        hidden_size = hidden_states.shape[2]
        
        # initialize estimators on first call
        if not hasattr(self, "vmf_estimators") or self.vmf_estimators is None:
            self.vmf_estimators = [
                VMFEstimator(feature_num=hidden_size, class_num=2)
                for _ in range(num_layers)
            ]
        
        uid_correctness = getattr(self, "_uid_correctness_cache", None)
        if not uid_correctness:
            print("No labeled samples in this batch, skipping vMF update.")
            return
        
        correct_indices = []
        incorrect_indices = []
        for corr, incorr in uid_correctness.values():
            correct_indices.extend(corr)
            incorrect_indices.extend(incorr)
        
        all_indices = incorrect_indices + correct_indices
        if len(all_indices) == 0:
            return
        
        # labels: 0 = incorrect, 1 = correct
        labels = torch.cat([
            torch.zeros(len(incorrect_indices), dtype=torch.long),
            torch.ones(len(correct_indices), dtype=torch.long),
        ])
        
        idx_tensor = torch.tensor(all_indices, dtype=torch.long)
        all_hidden = hidden_states[idx_tensor].float()
        all_hidden_norm = torch.nn.functional.normalize(all_hidden, p=2, dim=-1)
        
        for layer_idx in range(num_layers):
            features = all_hidden_norm[:, layer_idx, :]
            est = self.vmf_estimators[layer_idx]
            est.update_CV(features.detach(), labels)
            est.update_kappa()

    def _cache_all_contrast_logits(self, data: DataProto):
        """Compute ρ(z,c) for all samples × all layers; store in self._contrast_logits_cache."""
        if not hasattr(self, "vmf_estimators") or self.vmf_estimators is None:
            self._contrast_logits_cache = None
            return
        if "last_token_hidden_states" not in data.batch:
            self._contrast_logits_cache = None
            return

        hidden_states = data.batch["last_token_hidden_states"]  # (bs, num_layers, hidden_size)
        bs = hidden_states.shape[0]
        num_layers = hidden_states.shape[1]
        feature_num = hidden_states.shape[2]

        all_cl = torch.zeros(bs, num_layers, 2)

        for layer_idx in range(num_layers):
            est = self.vmf_estimators[layer_idx]
            Ave_norm = torch.nn.functional.normalize(est.Ave.detach(), dim=1)  # (2, H)
            kappa = est.kappa.detach()   # (2,)
            logc = est.logc.detach()     # (2,)

            h = hidden_states[:, layer_idx, :].float()                        # (bs, H)
            h_norm = torch.nn.functional.normalize(h, p=2, dim=-1)
            temp = kappa.reshape(-1, 1) * Ave_norm                            # (2, H)
            temp = temp.unsqueeze(0) + h_norm.unsqueeze(1)                    # (bs, 2, H)
            kappa_new = torch.linalg.norm(temp, dim=2)                        # (bs, 2)
            with torch.no_grad():
                cl = VMFEstimator.compute_log_ratio(kappa_new, feature_num, logc)  # (bs, 2)
            all_cl[:, layer_idx, :] = cl.cpu()

        self._contrast_logits_cache = all_cl  # (bs, num_layers, 2)
    
    def _cache_uid_correctness(self, data: DataProto):
        """Cache per-uid correctness partitions and response statistics for downstream use."""
        uid2idx_l, _ = get_labeled_unlabeled_indices(data)

        uid_correctness = {}  # uid -> (correct_indices, incorrect_indices)
        eligible_uids = set()

        for uid, indices in uid2idx_l.items():
            golden_answer = data[indices[0]].non_tensor_batch["reward_model"]["ground_truth"]
            correct_indices = []
            incorrect_indices = []
            for idx in indices:
                ans = self.pred_answers[idx]
                if ans is not None and verify_answer(ans, golden_answer):
                    correct_indices.append(idx)
                else:
                    incorrect_indices.append(idx)
            uid_correctness[uid] = (correct_indices, incorrect_indices)

            # track response-length asymmetry per uid
            if correct_indices and incorrect_indices:
                corr_lens = [data.batch["attention_mask"][i][data.batch["prompts"][i].shape[-1]:].sum().item() for i in correct_indices]
                incorr_lens = [data.batch["attention_mask"][i][data.batch["prompts"][i].shape[-1]:].sum().item() for i in incorrect_indices]
                if sum(corr_lens) / len(corr_lens) > sum(incorr_lens) / len(incorr_lens):
                    eligible_uids.add(uid)

        self._uid_correctness_cache = uid_correctness
        self._length_eligible_uids = eligible_uids

    def _log_vmf_contrast_stats(self, data: DataProto):
        """Log contrast stats, update per-layer discriminability, and select top-K layers."""
        if not hasattr(self, "vmf_estimators") or self.vmf_estimators is None:
            return
        if "last_token_hidden_states" not in data.batch:
            return

        cache = getattr(self, "_contrast_logits_cache", None)
        if cache is None:
            return

        num_layers = cache.shape[1]

        uid_correctness = getattr(self, "_uid_correctness_cache", None)
        if not uid_correctness:
            return

        correct_indices = []
        incorrect_indices = []
        for corr, incorr in uid_correctness.values():
            correct_indices.extend(corr)
            incorrect_indices.extend(incorr)

        if not correct_indices or not incorrect_indices:
            return

        corr_idx = torch.tensor(correct_indices, dtype=torch.long)
        incorr_idx = torch.tensor(incorrect_indices, dtype=torch.long)

        # per-layer discriminability
        step_disc = []
        for layer_idx in range(num_layers):
            cl_corr = cache[corr_idx, layer_idx, :]
            cl_incorr = cache[incorr_idx, layer_idx, :]

            disc = ((cl_corr[:, 1].mean().item() - cl_incorr[:, 1].mean().item())
                    + (cl_incorr[:, 0].mean().item() - cl_corr[:, 0].mean().item()))
            step_disc.append(disc)

        # running average
        if not hasattr(self, "_disc_history_sum"):
            self._disc_history_sum = [0.0] * num_layers
            self._disc_history_count = 0
        self._disc_history_count += 1
        for i in range(num_layers):
            self._disc_history_sum[i] += step_disc[i]

        avg_disc = [self._disc_history_sum[i] / self._disc_history_count for i in range(num_layers)]

        # select top-K% layers
        top_layer_ratio = self.config.trainer.get("vmf_top_layer_ratio", 0.1)
        k = max(1, int(num_layers * top_layer_ratio))
        sorted_layer_indices = sorted(range(num_layers), key=lambda i: avg_disc[i], reverse=True)
        self._top_disc_layers = sorted(sorted_layer_indices[:k])

        # write summary to log
        log_dir = self.config.trainer.get("log_dir", None)
        if not log_dir:
            return

        log_path = os.path.join(log_dir, "vmf_contrast_stats.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== step {self.global_steps} ===\n")
            f.write(f"n_correct={len(correct_indices)}, n_incorrect={len(incorrect_indices)}\n")
            f.write(f"top_layers (k={k}): {self._top_disc_layers}\n")
            f.write(f"disc_avg range: "
                    f"min={min(avg_disc):.4f} (layer {avg_disc.index(min(avg_disc))}), "
                    f"max={max(avg_disc):.4f} (layer {avg_disc.index(max(avg_disc))})\n\n")

    def _reweight_adv_by_div(self, data: DataProto):
        """Amplify advantages for boundary-ambiguous samples."""
        
        if "advantages" not in data.batch:
            return data, {}

        cache = getattr(self, "_contrast_logits_cache", None)
        has_vmf = cache is not None

        uid_correctness = getattr(self, "_uid_correctness_cache", None)
        if uid_correctness is None:
            return data, {}

        eligible_uids = getattr(self, "_length_eligible_uids", None)

        reweight_ratio = self.config.trainer.get("reweight_ratio", 3.0)
        boosted_uids = 0
        skipped_uids = 0

        for uid, (correct_indices, incorrect_indices) in uid_correctness.items():
            if not correct_indices or not incorrect_indices:
                skipped_uids += 1
                continue

            # geometric inversion check (sum affinity across layers, then max/min)
            vmf_misaligned = False
            if has_vmf:
                corr_idx = torch.tensor(correct_indices, dtype=torch.long)
                incorr_idx = torch.tensor(incorrect_indices, dtype=torch.long)

                corr_aff_to_corr_vmf = cache[corr_idx, :, 1].sum(dim=1)
                incorr_aff_to_corr_vmf = cache[incorr_idx, :, 1].sum(dim=1)
                corr_aff_to_incorr_vmf = cache[corr_idx, :, 0].sum(dim=1)
                incorr_aff_to_incorr_vmf = cache[incorr_idx, :, 0].sum(dim=1)

                if (incorr_aff_to_corr_vmf.max().item() > corr_aff_to_corr_vmf.min().item()
                        or corr_aff_to_incorr_vmf.max().item() > incorr_aff_to_incorr_vmf.min().item()):
                    vmf_misaligned = True

            # check eligibility from pre-computed structural signals
            uid_eligible = eligible_uids is None or uid in eligible_uids

            if vmf_misaligned and uid_eligible:
                all_indices = correct_indices + incorrect_indices
                for idx in all_indices:
                    data.batch["advantages"][idx] *= reweight_ratio
                boosted_uids += 1
            else:
                skipped_uids += 1

        reweight_metrics = {
            "vmf_reweight/total_uids": len(uid_correctness),
            "vmf_reweight/boosted_uids": boosted_uids,
            "vmf_reweight/skipped_uids": skipped_uids,
        }

        return data, reweight_metrics

    # ── unlabeled sample mining for semi-supervised RLVR ─────────────────────────

    def _compute_vmf_conf(self, uid_major_map, uid_minor_map, hidden_states):
        """Compute per-uid confidence scores over top discriminative layers."""
        if not uid_major_map:
            return {}
        
        uid_order = []
        all_indices = []
        all_signs = []       # +1 for majority, -1 for minority
        uid_sizes = []
        
        for uid, major_indices in uid_major_map.items():
            uid_order.append(uid)
            minor_indices = uid_minor_map.get(uid, [])
            
            all_indices.extend(major_indices)
            all_signs.extend([1.0] * len(major_indices))
            
            all_indices.extend(minor_indices)
            all_signs.extend([-1.0] * len(minor_indices))
            
            uid_sizes.append(len(major_indices) + len(minor_indices))
        
        total_rollouts = len(all_indices)
        idx_tensor = torch.tensor(all_indices, dtype=torch.long)
        sign_tensor = torch.tensor(all_signs, dtype=torch.float64)
        
        num_layers = hidden_states.shape[1]
        feature_num = hidden_states.shape[2]
        
        conf_layers = getattr(self, "_top_disc_layers", None)
        if conf_layers is None:
            conf_layers = list(range(num_layers))
        
        cache = getattr(self, "_contrast_logits_cache", None)
        
        rollout_conf_sum = torch.zeros(total_rollouts, dtype=torch.float64)
        
        for layer_idx in conf_layers:
            if cache is not None:
                cl = cache[idx_tensor, layer_idx, :]  # (total_rollouts, 2)
                conf = (cl[:, 1] - cl[:, 0]).double()
            else:
                # fallback: compute on the fly
                est = self.vmf_estimators[layer_idx]
                Ave_norm = torch.nn.functional.normalize(est.Ave.detach(), dim=1)
                kappa = est.kappa.detach()
                logc = est.logc.detach()
                
                group_hidden = hidden_states[idx_tensor, layer_idx, :].float()
                group_hidden_norm = torch.nn.functional.normalize(group_hidden, p=2, dim=-1)
                
                temp = kappa.reshape(-1, 1) * Ave_norm
                temp = temp.unsqueeze(0) + group_hidden_norm.unsqueeze(1)
                kappa_new = torch.linalg.norm(temp, dim=2)
                
                with torch.no_grad():
                    contrast_logits = VMFEstimator.compute_log_ratio(kappa_new, feature_num, logc)
                
                conf = (contrast_logits[:, 1] - contrast_logits[:, 0]).cpu().double()
            
            rollout_conf_sum += conf
        
        signed_conf = rollout_conf_sum * sign_tensor
        
        uid_scores = {}
        offset = 0
        for uid, size in zip(uid_order, uid_sizes):
            uid_scores[uid] = signed_conf[offset:offset + size].mean().item()
            offset += size
        
        return uid_scores
    
    def _find_unreliable_uids(self, fit_scores: list, all_conf_scores: list, uid_info: dict):
        """Filter unreliable pseudo-labels via 2-component GMM (threshold=0.5)."""
        if not hasattr(self, "_fit_score_history"):
            self._fit_score_history = deque(maxlen=5)
        
        if fit_scores:
            self._fit_score_history.append([s for _, s in fit_scores])
        
        historical_scores = []
        for batch_scores in self._fit_score_history:
            historical_scores.extend(batch_scores)
        
        all_scores_array = np.array([s for _, s in all_conf_scores]).reshape(-1, 1)

        if len(historical_scores) < 4:
            return  # not enough data to fit GMM
        
        fit_scores_array = np.array(historical_scores).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=100, reg_covar=1e-3)
        gmm.fit(fit_scores_array)

        means = gmm.means_.flatten()
        reliable_component = int(np.argmax(means))

        probs = gmm.predict_proba(all_scores_array)
        reliable_probs = probs[:, reliable_component]

        for i, (uid, score) in enumerate(all_conf_scores):
            if reliable_probs[i] < 0.5:
                uid_info[uid]["reliable"] = False

    def _process_unlabeled_data(self, data: DataProto):
        """Majority voting → vMF confidence → GMM filtering → pseudo-label assignment."""
        uid2idx = defaultdict(list)
        rel_pseudo_correct_num = 0
        unrel_pseudo_correct_num = 0
        rel_pseudo_num = 0
        unrel_pseudo_num = 0

        for i in range(len(data)):
            data_item = data[i]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            assert extra_info and extra_info["labeled"] in (True, False)

            if extra_info["labeled"] is False:
                uid = data_item.non_tensor_batch.get("uid")
                uid2idx[uid].append(i)

        vmf_ready = (
            hasattr(self, "vmf_estimators") 
            and self.vmf_estimators is not None 
            and self.vmf_estimators[0].Amount.min() >= 2
            and "last_token_hidden_states" in data.batch.keys()
        )
        
        uid_info = {}
        uid_major_map = {}
        uid_minor_map = {}
        hidden_states = data.batch["last_token_hidden_states"]
        
        for uid, indices in uid2idx.items():
            pred_answers = [self.pred_answers[idx] for idx in indices]
            major_answer = find_majority(pred_answers)
            gold_answer = data[indices[0]].non_tensor_batch["reward_model"]["ground_truth"]
            if major_answer is None:
                uid_info[uid] = {
                    "major_answer": None,
                    "gold_answer": gold_answer,
                    "reliable": False,
                }
                continue
        
            major_rollout_indices = []
            minor_rollout_indices = []
            for idx, ans in zip(indices, pred_answers):
                if verify_answer(ans, major_answer):
                    major_rollout_indices.append(idx)
                else:
                    minor_rollout_indices.append(idx)
            
            if len(major_rollout_indices) > 0:
                uid_major_map[uid] = major_rollout_indices
                uid_minor_map[uid] = minor_rollout_indices
            
            uid_info[uid] = {
                "major_answer": major_answer,
                "gold_answer": gold_answer,
                "reliable": True,
            }
        
        if vmf_ready:
            all_conf_scores = []
            if uid_major_map:
                uid_scores = self._compute_vmf_conf(uid_major_map, uid_minor_map, hidden_states)
                for uid, score in uid_scores.items():
                    all_conf_scores.append((uid, score))

            fit_scores = [(uid, s) for uid, s in all_conf_scores if uid_minor_map.get(uid, [])]
            self._find_unreliable_uids(fit_scores, all_conf_scores, uid_info)

        for uid, indices in uid2idx.items():
            info = uid_info[uid]
            major_answer = info["major_answer"]
            gold_answer = info["gold_answer"]
            reliable_major = info["reliable"]

            if reliable_major:
                for idx in indices:
                    data[idx].non_tensor_batch["reward_model"]["original_ground_truth"] = gold_answer
                    data[idx].non_tensor_batch["reward_model"]["ground_truth"] = major_answer
                
                rel_pseudo_num += 1
                if verify_answer(major_answer, gold_answer):
                    rel_pseudo_correct_num += 1
            else:
                for idx in indices:
                    data[idx].non_tensor_batch["reward_model"]["original_ground_truth"] = gold_answer
                    data[idx].non_tensor_batch["reward_model"]["ground_truth"] = "None"

                unrel_pseudo_num += 1
                if major_answer is not None and verify_answer(major_answer, gold_answer):
                    unrel_pseudo_correct_num += 1

        semi_metrics = compute_semi_metrics(
            rel_pseudo_num, rel_pseudo_correct_num,
            unrel_pseudo_correct_num, unrel_pseudo_num=unrel_pseudo_num,
        )

        return data, semi_metrics

    # ── training loop ──────────────────────────────────────────────────

    def fit(self):
        """Main PPO training loop with vMF distribution modeling and semi-supervised learning."""
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
        # currently, we only support validation using the reward_function.
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

        self.dist_modeling = self.config.trainer.get("dist_modeling", False)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            # reset vMF estimators at the beginning of each epoch
            if self.dist_modeling and hasattr(self, "vmf_estimators"):
                for est in self.vmf_estimators:
                    est.reset()
                print(f"[Epoch {epoch}] Reset vMF estimators for fresh fitting.")

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
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
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
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            if self.dist_modeling:
                                batch.meta_info["extract_hidden_states"] = True
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
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # pre-extract all predicted answers once for the entire batch, cache in self
                    all_indices = list(range(len(batch)))
                    self.pred_answers = extract_pred_answers(batch, all_indices, self.tokenizer)
                    self._cache_uid_correctness(batch)

                    # model vMF distributions for each layer using labeled data
                    if self.dist_modeling:
                        with marked_timer("dist_model", timing_raw, color="cyan"):
                            self._distribution_modeling(batch)
                            self._cache_all_contrast_logits(batch)
                            self._log_vmf_contrast_stats(batch)
                    
                    # generate pseudo-labels for unlabeled data and filter unreliable ones
                    semi_metrics = {}
                    if self.config.trainer.train_mode != "supervised":
                        with marked_timer("process_ul", timing_raw, color="green"):
                            batch, semi_metrics = self._process_unlabeled_data(batch)

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
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
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

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    
                    # reweight advantages by divergence
                    reweight_metrics = {}
                    if self.config.trainer.get("adv_reweight", False):
                        with marked_timer("adv_reweight", timing_raw, color="magenta"):
                            batch, reweight_metrics = self._reweight_adv_by_div(batch)

                    # clear per-step caches
                    self._contrast_logits_cache = None
                    self._uid_correctness_cache = None
                    self._length_eligible_uids = None

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

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

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
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
                # training metrics for semi-supervised / unsupervised training
                if semi_metrics:
                    metrics.update(semi_metrics)
                # reweight metrics
                if reweight_metrics:
                    metrics.update(reweight_metrics)
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
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

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
