from __future__ import annotations

import numpy as np
from PIL import Image


def _resize_float(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    up = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(up, dtype=np.float32) / 255.0


def refine_coastline_from_draft(
    draft_mask: np.ndarray,
    *,
    seed: int,
    sea_level: float,
    coarse_scale: int,
    draft_weight: float = 0.70,
) -> np.ndarray:
    """
    输入草稿二值图（陆地=255, 海洋=0），输出细化后的二值图。
    """
    if draft_mask.ndim != 2:
        raise ValueError("draft_mask 必须是二维数组")
    if not (0.0 < sea_level < 1.0):
        raise ValueError("sea_level 必须在 (0,1) 之间")
    if coarse_scale < 1:
        raise ValueError("coarse_scale 必须 >= 1")

    height, width = draft_mask.shape
    draft_float = (draft_mask > 127).astype(np.float32)

    sh = max(8, height // coarse_scale)
    sw = max(8, width // coarse_scale)
    rng = np.random.default_rng(seed)
    coarse_noise = rng.random((sh, sw), dtype=np.float32)
    noise = _resize_float(coarse_noise, width, height)

    # 对草稿做轻微平滑，避免锯齿导致阈值结果碎裂。
    smooth_draft = _resize_float(_resize_float(draft_float, sw, sh), width, height)

    merged = draft_weight * smooth_draft + (1.0 - draft_weight) * noise
    refined = merged > sea_level
    return (refined.astype(np.uint8) * 255)
