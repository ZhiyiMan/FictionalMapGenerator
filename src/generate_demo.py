"""
最小示例：生成一张陆海二值图并保存为 PNG。

说明：这里的宽高是「工作分辨率」演示。最终导出分辨率可在作品完成后，
根据比例尺与画面内容再自动放大/裁切（后续步骤实现）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from core.coastline import refine_coastline_from_draft
from core.scale import compute_scale_info


def land_mask_from_noise(
    height: int,
    width: int,
    *,
    seed: int = 0,
    sea_level: float = 0.48,
    coarse_scale: int = 8,
) -> np.ndarray:
    """
    用低分辨率随机噪声经平滑放大 + 阈值，得到陆海二值图（无需额外依赖）。

    返回 uint8 数组：陆地=255，海洋=0。
    """
    if height < 4 or width < 4:
        raise ValueError("height 和 width 至少为 4")
    if not (0.0 < sea_level < 1.0):
        raise ValueError("sea_level 必须在 (0, 1) 之间")
    if coarse_scale < 1:
        raise ValueError("coarse_scale 必须 >= 1")

    # demo 模式没有手绘草稿时，先用随机草稿占位，再复用核心细化算法。
    rng = np.random.default_rng(seed)
    draft = (rng.random((height, width)) > 0.5).astype(np.uint8) * 255
    return refine_coastline_from_draft(
        draft,
        seed=seed,
        sea_level=sea_level,
        coarse_scale=coarse_scale,
    )


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成演示用陆海二值 PNG")
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=512,
        metavar="PX",
        help="图像宽度（像素），默认 512",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=512,
        metavar="PX",
        help="图像高度（像素），默认 512",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="随机种子，相同种子可复现同一张图，默认 0",
    )
    parser.add_argument(
        "--sea-level",
        type=float,
        default=0.48,
        metavar="V",
        help="海平面阈值，范围 (0,1)，越大陆地越少，默认 0.48",
    )
    parser.add_argument(
        "--coarse-scale",
        type=int,
        default=8,
        metavar="N",
        help="粗尺度因子，越大越偏大块大陆，默认 8",
    )
    parser.add_argument(
        "--world-width-km",
        type=float,
        default=4000.0,
        metavar="KM",
        help="世界宽度（公里），默认 4000",
    )
    parser.add_argument(
        "--world-height-km",
        type=float,
        default=4000.0,
        metavar="KM",
        help="世界高度（公里），默认 4000",
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="输出 PNG 的完整路径（可含任意目录与文件名）",
    )
    out_group.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        metavar="FILENAME",
        help="仅指定文件名，文件会保存在 output/ 目录下",
    )
    args = parser.parse_args()

    if args.out is not None:
        out_path = args.out
    elif args.name is not None:
        out_path = Path("output") / args.name
    else:
        out_path = Path("output") / "demo_land_mask.png"

    mask = land_mask_from_noise(
        args.height,
        args.width,
        seed=args.seed,
        sea_level=args.sea_level,
        coarse_scale=args.coarse_scale,
    )
    scale = compute_scale_info(
        world_width_km=args.world_width_km,
        world_height_km=args.world_height_km,
        image_width_px=args.width,
        image_height_px=args.height,
    )
    save_png(mask, out_path)
    print(f"已写入: {out_path.resolve()}")
    print(
        "当前比例尺: "
        f"X轴 1 px = {scale.km_per_px_x:.3f} km, "
        f"Y轴 1 px = {scale.km_per_px_y:.3f} km; "
        f"100 km 对应约 {scale.px_per_100km_x:.1f}px × {scale.px_per_100km_y:.1f}px"
    )


if __name__ == "__main__":
    main()