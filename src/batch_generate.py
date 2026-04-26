from __future__ import annotations

import argparse
from pathlib import Path

from generate_demo import land_mask_from_noise, save_png


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成陆海二值 PNG")
    parser.add_argument("-W", "--width", type=int, default=512, help="图像宽度（像素）")
    parser.add_argument("-H", "--height", type=int, default=512, help="图像高度（像素）")
    parser.add_argument(
        "--sea-levels",
        type=str,
        default="0.45,0.50,0.55",
        help="海平面列表，逗号分隔，如 0.45,0.50,0.55",
    )
    parser.add_argument(
        "--coarse-scales",
        type=str,
        default="6,8,10",
        help="粗尺度列表，逗号分隔，如 6,8,10",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="随机种子列表，逗号分隔，如 0,1,2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output") / "batch",
        help="批量输出目录",
    )
    args = parser.parse_args()

    sea_levels = parse_float_list(args.sea_levels)
    coarse_scales = parse_int_list(args.coarse_scales)
    seeds = parse_int_list(args.seeds)

    if not sea_levels or not coarse_scales or not seeds:
        raise ValueError("sea-levels、coarse-scales、seeds 都不能为空")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for seed in seeds:
        for sea_level in sea_levels:
            for coarse_scale in coarse_scales:
                mask = land_mask_from_noise(
                    args.height,
                    args.width,
                    seed=seed,
                    sea_level=sea_level,
                    coarse_scale=coarse_scale,
                )
                name = (
                    f"map_w{args.width}_h{args.height}"
                    f"_seed{seed}"
                    f"_sl{sea_level:.2f}"
                    f"_cs{coarse_scale}.png"
                )
                save_png(mask, args.out_dir / name)
                total += 1

    print(f"批量生成完成: {total} 张")
    print(f"输出目录: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
