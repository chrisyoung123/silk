# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SiLK 训练阶段二：在阶段一 checkpoint 上冻结 descriptor，用位姿真值构造对应关系，
对 detector 做 DISK 风格策略梯度微调。

数据需包含 image0/image1 及下列之一：
  - T_0to1 + K0 + K1
  - homography
  - 平面位姿 + 外参（由 JsonPlanarPosePairs / MASt3RManifestPosePairs 提供）
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch

from silk.config.core import ensure_is_instance
from silk.cv.homography import HomographicSampler
from silk.flow import AutoForward, Flow
from silk.geometry.pose_correspondence import compute_pair_correspondences
from silk.losses.policy_gradient import PolicyGradientKeypointLoss
from silk.models.silk import SiLKBase
from silk.transforms.abstract import NamedContext, Transform


class _NoOpInfoNCELoss(torch.nn.Module):
    """占位，满足 SiLKBase 构造；阶段二不使用 JAX loss。"""

    def forward(self, *args, **kwargs):
        z = torch.tensor(0.0, requires_grad=True)
        return z, z, z, z


class SiLKStage2PolicyGradient(SiLKBase):
    """阶段二 Lightning 模块：仅优化 detector（logits 头）。"""

    def __init__(
        self,
        model,
        optimizer_spec=None,
        image_aug_transform: Optional[Transform] = None,
        pg_loss: Optional[PolicyGradientKeypointLoss] = None,
        freeze_backbone: bool = True,
        freeze_descriptor: bool = True,
        freeze_contextualizer: bool = True,
        default_depth: float = 5.0,
        train_detector_only: bool = True,
        **kwargs,
    ):
        # 阶段二不用 JAX InfoNCE；传入占位 loss
        super().__init__(
            model=model,
            loss=_NoOpInfoNCELoss(),
            optimizer_spec=optimizer_spec,
            image_aug_transform=image_aug_transform,
            **kwargs,
        )
        self._pg_loss = pg_loss or PolicyGradientKeypointLoss()
        self._freeze_backbone = freeze_backbone
        self._freeze_descriptor = freeze_descriptor
        self._freeze_contextualizer = freeze_contextualizer
        self._default_depth = default_depth
        self._train_detector_only = train_detector_only

        self.flow.define_transition("checked_batch", self._check_pose_batch, "batch")
        self.flow.define_transition(
            ("images", "image_shape", "pose_ctx"),
            self._unpack_pose_batch,
            "checked_batch",
        )
        self.flow.define_transition(
            "stacked_images",
            self._stack_pair_images,
            "images",
        )
        self._init_stage2_forward_flow("stacked_images", "pose_ctx", "image_shape")

    def on_fit_start(self) -> None:
        if self._train_detector_only:
            self._apply_stage2_freeze()

    def _apply_stage2_freeze(self) -> None:
        for name, param in self.named_parameters():
            trainable = any(
                key in name
                for key in (
                    "logits",
                    "detH",
                    "detector",
                    "DetectorHead",
                )
            )
            param.requires_grad = trainable

    def _check_pose_batch(self, batch):
        ensure_is_instance(batch, NamedContext)
        batch.ensure_exists("image0", "image1")

        def to_device(el):
            if isinstance(el, torch.Tensor):
                return el.to(self.device)
            return el

        return batch.map(to_device)

    def _unpack_pose_batch(self, batch: NamedContext):
        img0 = batch["image0"].to(self.device)
        img1 = batch["image1"].to(self.device)
        if img0.dim() == 3:
            img0 = img0.unsqueeze(0)
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        assert img0.shape[0] == img1.shape[0]
        shape = img0.shape

        pose_ctx = {
            "T_0to1": batch.get("T_0to1"),
            "K0": batch.get("K0"),
            "K1": batch.get("K1"),
            "homography": batch.get("homography"),
            "depth0": batch.get("depth0"),
        }
        for k, v in pose_ctx.items():
            if isinstance(v, torch.Tensor):
                pose_ctx[k] = v.to(self.device)

        images = (img0, img1)
        return images, shape, pose_ctx

    @staticmethod
    def _stack_pair_images(images):
        img0, img1 = images
        stacked = torch.stack((img0, img1), dim=1)
        b, two, c, h, w = stacked.shape
        return stacked.view(b * two, c, h, w)

    def _init_stage2_forward_flow(
        self,
        images_input_name: str,
        pose_ctx_name: str,
        image_shape_name: str,
    ):
        self.flow.define_transition(
            "augmented_images",
            self._aug_images,
            images_input_name,
            "use_image_aug",
        )
        self.flow.define_transition(
            "gray_images",
            self._grayify,
            "augmented_images",
        )
        self.flow.define_transition(
            ("descriptors", "logits"),
            self._model.forward_flow,
            outputs=Flow.Constant(("normalized_descriptors", "logits")),
            images="gray_images",
        )
        self.flow.define_transition(
            "descriptors_shape",
            lambda x: x.shape,
            "descriptors",
        )
        self.flow.define_transition(
            ("logits_0", "logits_1", "desc_0", "desc_1", "corr_forward", "corr_backward"),
            self._pose_pg_tensors,
            "descriptors",
            "logits",
            pose_ctx_name,
            image_shape_name,
        )
        self.flow.define_transition(
            ("pg_loss", "metrics"),
            self._compute_pg_loss,
            "logits_0",
            "logits_1",
            "desc_0",
            "desc_1",
            "corr_forward",
            "corr_backward",
        )
        AutoForward.__init__(self, self.flow, "pg_loss")
        self._stage2_fn = self.flow.with_outputs(("pg_loss", "metrics"))

    def _pose_pg_tensors(
        self,
        descriptors,
        logits,
        pose_ctx: Dict[str, Any],
        image_shape,
    ):
        desc_flat = SiLKBase._img_to_flat(descriptors)
        logits_flat = SiLKBase._img_to_flat(logits).squeeze(-1)

        desc_0 = desc_flat[0::2]
        desc_1 = desc_flat[1::2]
        logits_0 = logits_flat[0::2]
        logits_1 = logits_flat[1::2]

        corr_fwd, corr_bwd = self._correspondences_from_pose(
            descriptors,
            pose_ctx,
            image_shape,
        )

        return logits_0, logits_1, desc_0, desc_1, corr_fwd, corr_bwd

    def _correspondences_from_pose(
        self,
        descriptors: torch.Tensor,
        pose_ctx: Dict[str, Any],
        image_shape,
    ):
        desc_view0 = descriptors[0::2]
        B = desc_view0.shape[0]
        descriptors_height = desc_view0.shape[2]
        descriptors_width = desc_view0.shape[3]
        img_h = int(image_shape[-2])
        img_w = int(image_shape[-1])
        device = desc_view0.device
        cell_size = 1.0

        positions = HomographicSampler._create_meshgrid(
            descriptors_height,
            descriptors_width,
            device=device,
            normalized=False,
        )
        positions = positions.expand(B, -1, -1, -1).reshape(B, -1, 2)

        coord_mapping = self._model.coordinate_mapping_composer.get(
            "images",
            "raw_descriptors",
        )
        positions_xy = coord_mapping.reverse(positions)

        T_0to1 = pose_ctx.get("T_0to1")
        K0 = pose_ctx.get("K0")
        K1 = pose_ctx.get("K1")
        H_mat = pose_ctx.get("homography")
        depth0 = pose_ctx.get("depth0")

        if H_mat is not None:
            corr_forward, corr_backward = compute_pair_correspondences(
                positions_xy,
                descriptors_width,
                descriptors_height,
                homography=H_mat,
                image_shape=(img_h, img_w),
                coord_mapping=coord_mapping,
            )
            return corr_forward, corr_backward

        if T_0to1 is None:
            raise RuntimeError("阶段二需要 T_0to1 或 homography 位姿真值")

        if K0 is None or K1 is None:
            K0 = K1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)

        if K0.dim() == 2:
            K0 = K0.unsqueeze(0).expand(B, -1, -1)
        if K1.dim() == 2:
            K1 = K1.unsqueeze(0).expand(B, -1, -1)
        if T_0to1.dim() == 2:
            T_0to1 = T_0to1.unsqueeze(0).expand(B, -1, -1)

        corr_forward, corr_backward = compute_pair_correspondences(
            positions_xy,
            descriptors_width,
            descriptors_height,
            T_0to1=T_0to1,
            K0=K0,
            K1=K1,
            depth0=depth0,
            default_depth=self._default_depth,
            image_shape=(img_h, img_w),
            coord_mapping=coord_mapping,
        )
        return corr_forward, corr_backward

    def _compute_pg_loss(
        self,
        logits_0,
        logits_1,
        desc_0,
        desc_1,
        corr_forward,
        corr_backward,
    ):
        loss, metrics = self._pg_loss(
            logits_0,
            logits_1,
            desc_0,
            desc_1,
            corr_forward,
            corr_backward,
        )
        return loss, metrics

    def _total_loss(self, mode, batch, use_image_aug: bool):
        pg_loss, metrics = self._stage2_fn(batch, use_image_aug)
        self.log(f"{mode}.pg.loss", pg_loss)
        for key, val in metrics.items():
            if isinstance(val, torch.Tensor):
                self.log(f"{mode}.pg.{key}", val.float())
        return pg_loss

    def training_step(self, batch, batch_idx):
        return self._total_loss("train", batch, use_image_aug=True)

    def validation_step(self, batch, batch_idx):
        return self._total_loss("val", batch, use_image_aug=False)
