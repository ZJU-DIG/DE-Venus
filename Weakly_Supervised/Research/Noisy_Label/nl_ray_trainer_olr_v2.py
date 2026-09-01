# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Noisy-label OLR v2: reuse TraPO v2 training (warmup + full probe + stop after warmup).

Unlike semi-supervised TraPO v2, every row is treated as **labeled** (``extra_info.labeled=True``).
There is no unlabeled pool to promote offline; ``nl_run_olr_v2.sh relabel`` rewrites
``reward_model.ground_truth`` when probe majority differs from the current label.
"""

from __future__ import annotations

from ..Semi_RL.ray_trainer_trapo_v2 import RayPPOTrainerTraPOv2


class RayNLTrainerOLRv2(RayPPOTrainerTraPOv2):
    """NL-OLR v2 trainer = :class:`RayPPOTrainerTraPOv2` on an all-labeled noisy-label parquet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_labeled = len(self._labeled_indices) if self._labeled_indices else 0
        print(
            "[NL-OLR v2] Using TraPO v2 training loop. "
            f"Labeled pool size={n_labeled} (expect all rows labeled=True). "
            "Offline step: build_noisy_label_dataset (relabel), not stage2 promote."
        )
