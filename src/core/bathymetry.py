from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def _resize_noise(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    sh = max(8, height // 16)
    sw = max(8, width // 16)
    coarse = rng.random((sh, sw), dtype=np.float32)
    img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
    up = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(up, dtype=np.float32) / 255.0


def build_bathymetry(refined_mask: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """
    生成海底地形矩阵:
    - 陆地固定为 0
    - 海域为 0~1（离岸越远越深）
    """
    if refined_mask.ndim != 2:
        raise ValueError("refined_mask 必须是二维数组")

    h, w = refined_mask.shape
    land = refined_mask > 127
    sea = ~land

    dist = np.full((h, w), -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()

    # 以“贴岸海格”作为 BFS 起点。
    for y in range(h):
        for x in range(w):
            if not sea[y, x]:
                continue
            near_land = (
                (y > 0 and land[y - 1, x])
                or (y < h - 1 and land[y + 1, x])
                or (x > 0 and land[y, x - 1])
                or (x < w - 1 and land[y, x + 1])
            )
            if near_land:
                dist[y, x] = 0
                q.append((y, x))

    # 特殊情况：全海或全陆，直接返回零矩阵。
    if not q:
        return np.zeros((h, w), dtype=np.float32)

    while q:
        y, x = q.popleft()
        base = dist[y, x] + 1
        if y > 0 and sea[y - 1, x] and dist[y - 1, x] == -1:
            dist[y - 1, x] = base
            q.append((y - 1, x))
        if y < h - 1 and sea[y + 1, x] and dist[y + 1, x] == -1:
            dist[y + 1, x] = base
            q.append((y + 1, x))
        if x > 0 and sea[y, x - 1] and dist[y, x - 1] == -1:
            dist[y, x - 1] = base
            q.append((y, x - 1))
        if x < w - 1 and sea[y, x + 1] and dist[y, x + 1] == -1:
            dist[y, x + 1] = base
            q.append((y, x + 1))

    depth = np.zeros((h, w), dtype=np.float32)
    sea_dist = dist[sea]
    max_dist = float(sea_dist.max()) if sea_dist.size else 0.0
    if max_dist > 0:
        depth[sea] = sea_dist.astype(np.float32) / max_dist

    rng = np.random.default_rng(seed)
    noise = _resize_noise(h, w, rng)
    depth = np.clip(depth * 0.85 + noise * 0.15, 0.0, 1.0)
    depth[land] = 0.0
    return depth
