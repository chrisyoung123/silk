"""
天空/语义分割 ONNX 推理（仅依赖 numpy、opencv、onnxruntime）。

前处理可选：
- mmseg: SegDataPreProcessor（0–255 像素 + mean/std，BGR→RGB）
- imagenet: Resize + BGR→RGB + (x/255 - mean) / std（PyTorch 常用 ImageNet 系数），float32 NCHW
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

OnnxProviders = Sequence[str]
NormStyle = Literal["mmseg", "imagenet"]
OutputStyle = Literal["argmax", "minmax_u8"]

# SegDataPreProcessor（与训练配置一致）
_SEG_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_SEG_STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)
_SEG_MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_SEG_STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# PyTorch / torchvision 常用 ImageNet 归一化（RGB 顺序，先 /255）
_IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def providers_for_device(device: str) -> List[str]:
    """onnxruntime provider 列表；cuda 不可用时回退 CPU。"""
    d = device.lower().strip()
    if d in ("cpu",):
        return ["CPUExecutionProvider"]
    if d in ("cuda", "gpu"):
        try:
            ort = importlib.import_module("onnxruntime")
            available = set(ort.get_available_providers())
        except ImportError:
            available = set()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(
            "警告：请求 sky ONNX 使用 CUDA，但 onnxruntime 无 CUDAExecutionProvider；"
            "请安装 onnxruntime-gpu。回退 CPU。",
            file=sys.stderr,
        )
        return ["CPUExecutionProvider"]
    raise ValueError(f"未知 sky 分割 device: {device!r}（支持 cpu / cuda）")


def default_sky_seg_device() -> str:
    return os.environ.get("SKY_SEG_DEVICE", "cuda").strip().lower() or "cuda"


def resolve_sky_seg_providers(device: Optional[str] = None) -> List[str]:
    dev = (device or default_sky_seg_device()).strip().lower() or "cuda"
    return providers_for_device(dev)


def _resolve_hw_from_onnx_input_shape(in_shape: object) -> Tuple[int, int]:
    """从 ONNX 输入 shape 解析 (H, W)。支持 NCHW / NHWC。"""
    if len(in_shape) != 4:
        raise RuntimeError(f"当前仅支持 4D ONNX 输入，实际为 {in_shape}")
    if isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
        return int(in_shape[2]), int(in_shape[3])
    if isinstance(in_shape[1], int) and isinstance(in_shape[2], int):
        return int(in_shape[1]), int(in_shape[2])
    raise RuntimeError(f"无法从 ONNX 输入 shape 解析 H,W: {in_shape}")


def preprocess_seg_image_bgr(
    image_bgr: np.ndarray,
    size_hw: Tuple[int, int],
    *,
    bgr_to_rgb: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    SegDataPreProcessor 风格：resize → (可选 BGR→RGB) → (x - mean) / std。
    mean/std 在 [0, 255] 像素尺度上应用。返回 NCHW float32。
    """
    h_in, w_in = int(size_hw[0]), int(size_hw[1])
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道图，当前 shape={img.shape}")

    if img.shape[0] != h_in or img.shape[1] != w_in:
        img = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    x = img.astype(np.float32)
    if bgr_to_rgb:
        x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
        m = _SEG_MEAN_RGB if mean is None else mean
        s = _SEG_STD_RGB if std is None else std
    else:
        m = _SEG_MEAN_BGR if mean is None else mean
        s = _SEG_STD_BGR if std is None else std

    x = (x - m.reshape(1, 1, 3)) / s.reshape(1, 1, 3)
    return np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)


def preprocess_imagenet_bgr(
    image_bgr: np.ndarray,
    size_hw: Tuple[int, int],
) -> np.ndarray:
    """
    与常见 ONNX 示例一致：Resize → BGR→RGB → float32 → (x/255 - mean) / std → NCHW。
    mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]（RGB）。
    """
    h_in, w_in = int(size_hw[0]), int(size_hw[1])
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"需要 BGR 三通道图，当前 shape={img.shape}")

    if img.shape[0] != h_in or img.shape[1] != w_in:
        img = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    x = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (x / 255.0 - _IMAGENET_MEAN_RGB) / _IMAGENET_STD_RGB
    return np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)


def _postprocess_minmax_u8(raw: np.ndarray, h0: int, w0: int) -> np.ndarray:
    """
    与参考脚本一致：squeeze → 对全体元素做 min-max → *255 → uint8。
    若为 (C,H,W) 或 (H,W,C) 多通道，再压成单通道显著图（通道 max），最后 resize 到原图。
    """
    x = np.asarray(raw, dtype=np.float32)
    x = np.squeeze(x)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        u8 = np.zeros(x.shape, dtype=np.uint8)
    else:
        u8 = ((x - mn) / (mx - mn) * 255.0).astype(np.uint8)

    if u8.ndim == 3:
        # 常见 CHW（C 小）或 HWC（C 小）
        c0, c1, c2 = u8.shape[0], u8.shape[1], u8.shape[2]
        if c0 <= 8 and c0 <= c2 and c0 <= c1:
            u8 = np.transpose(u8, (1, 2, 0))
        sal = np.max(u8, axis=-1)
    elif u8.ndim == 2:
        sal = u8
    else:
        raise RuntimeError(f"minmax_u8 无法处理 squeeze 后 shape={u8.shape}")

    hi, wi = int(sal.shape[0]), int(sal.shape[1])
    if (hi, wi) != (h0, w0):
        sal = cv2.resize(sal, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return sal


class SkySegOnnxSession:
    """对单一路径 ONNX 建一次 Session，可对多帧反复 predict。"""

    def __init__(
        self,
        onnx_path: Path,
        *,
        providers: Optional[OnnxProviders] = None,
        norm_style: NormStyle = "mmseg",
        bgr_to_rgb: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        output_style: OutputStyle = "argmax",
    ) -> None:
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要 onnxruntime：pip install onnxruntime") from e

        path = Path(onnx_path).expanduser().resolve()
        prov: List[str] = (
            list(providers)
            if providers is not None
            else resolve_sky_seg_providers()
        )
        self._session = ort.InferenceSession(str(path), providers=prov)
        self._providers = list(self._session.get_providers())
        inp = self._session.get_inputs()[0]
        self._in_name = inp.name
        self._in_shape = inp.shape
        self._norm_style: NormStyle = norm_style if norm_style in ("mmseg", "imagenet") else "mmseg"
        self._output_style: OutputStyle = output_style if output_style in ("argmax", "minmax_u8") else "argmax"
        self._bgr_to_rgb = bool(bgr_to_rgb)
        if mean is not None and std is not None:
            mean_a = np.asarray(mean, dtype=np.float32)
            std_a = np.asarray(std, dtype=np.float32)
            self._mean = mean_a if bgr_to_rgb else mean_a[[2, 1, 0]]
            self._std = std_a if bgr_to_rgb else std_a[[2, 1, 0]]
        else:
            self._mean = _SEG_MEAN_RGB if bgr_to_rgb else _SEG_MEAN_BGR
            self._std = _SEG_STD_RGB if bgr_to_rgb else _SEG_STD_BGR

    @property
    def providers(self) -> List[str]:
        return list(self._providers)

    @property
    def input_name(self) -> str:
        return self._in_name

    @property
    def input_shape(self) -> object:
        return self._in_shape

    @property
    def norm_style(self) -> NormStyle:
        return self._norm_style

    @property
    def output_style(self) -> OutputStyle:
        return self._output_style

    @property
    def output_infos(self) -> List[Tuple[str, object, str]]:
        """各输出的 (name, shape, type_string)。"""
        return [(o.name, o.shape, o.type) for o in self._session.get_outputs()]

    def _build_input_tensor(self, image_bgr: np.ndarray, h_in: int, w_in: int) -> np.ndarray:
        in_shape = self._in_shape
        if len(in_shape) != 4:
            raise RuntimeError(f"当前仅支持 4D ONNX 输入，实际为 {in_shape}")

        n_ch = None
        if isinstance(in_shape[1], int):
            n_ch = int(in_shape[1])
            layout = "NCHW"
        elif isinstance(in_shape[3], int):
            n_ch = int(in_shape[3])
            layout = "NHWC"
        else:
            layout = "NCHW"
            n_ch = 3

        if n_ch == 3:
            if self._norm_style == "imagenet":
                x = preprocess_imagenet_bgr(image_bgr, (h_in, w_in))
            else:
                x = preprocess_seg_image_bgr(
                    image_bgr,
                    (h_in, w_in),
                    bgr_to_rgb=self._bgr_to_rgb,
                    mean=self._mean,
                    std=self._std,
                )
            if layout == "NHWC":
                x = np.transpose(x[0], (1, 2, 0))[None, :, :, :]
            return x

        if n_ch == 1:
            img = image_bgr
            if img.shape[0] != h_in or img.shape[1] != w_in:
                img = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = (gray - float(self._mean[0])) / float(self._std[0])
            if layout == "NHWC":
                return gray[None, :, :, None]
            return gray[None, None, :, :]

        raise RuntimeError(f"不支持的输入通道数: {n_ch}")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        默认 (argmax + mmseg/imagenet): 返回与原图同尺寸的类别图 (H, W)，int32。

        output_style=minmax_u8: 返回与原图同尺寸的 uint8 显著图 [0,255]（非类别 id）。
        """
        h0, w0 = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        h_in, w_in = _resolve_hw_from_onnx_input_shape(self._in_shape)
        x = self._build_input_tensor(image_bgr, h_in, w_in)

        y = self._session.run(None, {self._in_name: x})[0]
        arr = np.asarray(y)

        if self._output_style == "minmax_u8":
            return _postprocess_minmax_u8(arr, h0, w0)

        if arr.ndim == 4:
            if arr.shape[1] > 1:
                cls = np.argmax(arr[0], axis=0).astype(np.int32)
            elif arr.shape[-1] > 1:
                cls = np.argmax(arr[0], axis=-1).astype(np.int32)
            else:
                cls = np.squeeze(arr[0]).astype(np.int32)
        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                cls = arr[0].astype(np.int32)
            elif arr.shape[-1] > 1:
                cls = np.argmax(arr, axis=-1).astype(np.int32)
            else:
                cls = np.squeeze(arr).astype(np.int32)
        elif arr.ndim == 2:
            cls = arr.astype(np.int32)
        else:
            raise RuntimeError(f"无法解析分割输出 shape={arr.shape}")

        if cls.shape != (h0, w0):
            cls = cv2.resize(cls.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(
                np.int32
            )
        return cls


def run_segmentation_class_map(
    image_bgr: np.ndarray,
    onnx_path: Union[Path, str],
    *,
    norm_style: NormStyle = "mmseg",
) -> np.ndarray:
    """运行 ONNX 分割并返回与原图同尺寸的类别图 (H, W, int32)。每次调用新建 Session。"""
    return SkySegOnnxSession(Path(onnx_path), norm_style=norm_style).predict(image_bgr)
