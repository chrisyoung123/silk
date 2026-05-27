# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DISK 风格策略梯度：从检测 logits 采样稀疏关键点，用匹配/几何一致性作 reward。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from silk.matching.mnn import mutual_nearest_neighbor


def sample_keypoint_indices(
    logits: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
    replace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从扁平 logits (B, N) 上无放回采样关键点索引。

    Returns
    -------
    indices : (B, K)
    log_probs : (B, K)  每个采样点的 log π(a)
    """
    B, N = logits.shape
    K = min(num_samples, N)
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    indices = torch.stack(
        [
            torch.multinomial(probs[b], K, replacement=replace)
            for b in range(B)
        ],
        dim=0,
    )
    log_probs = torch.log(probs.gather(1, indices).clamp(min=1e-8))
    return indices, log_probs


def gather_at_indices(flat_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """flat_tensor (B, N, C) or (B, N), indices (B, K) -> (B, K, C) or (B, K)"""
    if flat_tensor.dim() == 2:
        return flat_tensor.gather(1, indices)
    return flat_tensor.gather(1, indices.unsqueeze(-1).expand(-1, -1, flat_tensor.shape[-1]))


def matching_reward_from_correspondences(
    sampled_idx_0: torch.Tensor,
    sampled_idx_1: torch.Tensor,
    corr_forward: torch.Tensor,
    corr_backward: torch.Tensor,
) -> torch.Tensor:
    """
    几何真值 reward：采样点在两图上互一致对应则计 1，否则 0。

    sampled_idx_* : (B, K) 在各自半批（img0/img1）descriptor 栅格上的索引
    corr_forward : (B, N)  img0 格点 -> img1 索引
    """
    B, K = sampled_idx_0.shape
    device = sampled_idx_0.device
    rewards = []

    for b in range(B):
        cf = corr_forward[b]
        j_pred = cf[sampled_idx_0[b]]
        hit_fwd = (j_pred >= 0) & (corr_backward[b, j_pred.clamp(min=0)] == sampled_idx_0[b])

        cb = corr_backward[b]
        i_pred = cb[sampled_idx_1[b]]
        hit_bwd = (i_pred >= 0) & (corr_forward[b, i_pred.clamp(min=0)] == sampled_idx_1[b])

        r = 0.5 * (hit_fwd.float().mean() + hit_bwd.float().mean())
        rewards.append(r)

    return torch.stack(rewards, dim=0)


def descriptor_matching_reward(
    desc_0: torch.Tensor,
    desc_1: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """无 GT 时的匹配率 reward（MNN 命中率），desc (B, K, C)。"""
    B, K, _ = desc_0.shape
    rewards = []
    for b in range(B):
        d0 = F.normalize(desc_0[b], dim=-1)
        d1 = F.normalize(desc_1[b], dim=-1)
        matches = mutual_nearest_neighbor(d0, d1)
        if matches.numel() == 0:
            rewards.append(torch.zeros((), device=desc_0.device))
            continue
        rewards.append(matches.shape[0] / float(K))
    return torch.stack(rewards)


class PolicyGradientKeypointLoss(nn.Module):
    def __init__(
        self,
        num_keypoints: int = 256,
        temperature: float = 1.0,
        baseline_momentum: float = 0.9,
        reward_mode: str = "geometry",
        entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.temperature = temperature
        self.baseline_momentum = baseline_momentum
        assert reward_mode in {"geometry", "matching", "hybrid"}
        self.reward_mode = reward_mode
        self.entropy_weight = entropy_weight
        self.register_buffer("_baseline", torch.zeros(1), persistent=True)

    def forward(
        self,
        logits_0: torch.Tensor,
        logits_1: torch.Tensor,
        descriptors_0: torch.Tensor,
        descriptors_1: torch.Tensor,
        corr_forward: Optional[torch.Tensor] = None,
        corr_backward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        logits_* : (B, N)  已按 img0/img1 拆好的半批
        descriptors_* : (B, N, C)
        """
        idx0, log_p0 = sample_keypoint_indices(
            logits_0, self.num_keypoints, self.temperature
        )
        idx1, log_p1 = sample_keypoint_indices(
            logits_1, self.num_keypoints, self.temperature
        )

        d0 = gather_at_indices(descriptors_0, idx0)
        d1 = gather_at_indices(descriptors_1, idx1)

        if self.reward_mode == "geometry" and corr_forward is not None:
            reward = matching_reward_from_correspondences(
                idx0, idx1, corr_forward, corr_backward
            )
        elif self.reward_mode == "matching":
            reward = descriptor_matching_reward(d0, d1)
        else:
            r_geo = matching_reward_from_correspondences(
                idx0, idx1, corr_forward, corr_backward
            )
            r_mat = descriptor_matching_reward(d0, d1)
            reward = 0.5 * (r_geo + r_mat)

        baseline = self._baseline.item()
        advantage = reward - baseline
        with torch.no_grad():
            self._baseline.mul_(self.baseline_momentum).add_(
                reward.mean() * (1.0 - self.baseline_momentum)
            )

        log_prob = log_p0.sum(dim=-1) + log_p1.sum(dim=-1)
        pg_loss = -(advantage.detach() * log_prob).mean()

        entropy = 0.0
        if self.entropy_weight > 0:
            p0 = F.softmax(logits_0 / self.temperature, dim=-1)
            entropy = -(p0 * torch.log(p0.clamp(min=1e-8))).sum(dim=-1).mean()
            pg_loss = pg_loss - self.entropy_weight * entropy

        metrics = {
            "pg_loss": pg_loss.detach(),
            "reward": reward.mean().detach(),
            "advantage": advantage.mean().detach(),
            "baseline": torch.tensor(baseline, device=reward.device),
        }
        if isinstance(entropy, torch.Tensor):
            metrics["entropy"] = entropy.detach()

        return pg_loss, metrics
