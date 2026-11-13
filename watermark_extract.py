#!/usr/bin/env python3
"""Show a full-sized FFT spectrum or reconstruct a photo with inverse FFT."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def enable_heif_if_needed(path, parser):
    if path.suffix.lower() not in {".heic", ".heif", ".heics", ".heifs", ".hif"}:
        return
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        parser.exit(1, "读取 HEIC/HEIF 需要安装：python3 -m pip install pillow-heif\n")
    register_heif_opener(thumbnails=False)


def spectrum_image(pixels):
    """Visualize the full spectrum, emphasizing text-shaped high-frequency details."""
    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    h, w = spectrum.shape

    low, high = np.percentile(spectrum, (1, 99))
    scaled = np.uint8(np.clip(
        (spectrum - low) / max(high - low, 1e-6) * 255, 0, 255
    ))

    radius = max(5, round(min(w, h) * 0.012))
    background = np.asarray(
        Image.fromarray(scaled).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    )
    detail = scaled.astype(np.float32) - background
    low, high = np.percentile(detail, (5, 99.8))
    image = np.uint8(np.clip(
        (detail - low) / max(high - low, 1e-6) * 255, 0, 255
    ))
    return Image.fromarray(image, "L")


def fft_mode(image_path, output, save_npz=False):
    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        pixels = np.asarray(oriented.convert("RGB"), dtype=np.float32)
        if save_npz:
            has_alpha = (
                "A" in oriented.getbands() or "transparency" in oriented.info
            )
            alpha = (
                np.asarray(oriented.convert("RGBA").getchannel("A"), dtype=np.uint8)
                if has_alpha else np.empty((0, 0), dtype=np.uint8)
            )

    spectrum_image(pixels).save(output, format="PNG")

    if not save_npz:
        return None

    # Save complex coefficients (including phase) only for explicit --mode fft.
    freq = np.fft.rfft2(pixels, axes=(0, 1)).astype(np.complex64)
    data_path = output.with_suffix(".npz")
    np.savez_compressed(data_path, freq=freq, width=pixels.shape[1], alpha=alpha)
    return data_path


def ifft_mode(data_path, output):
    with np.load(data_path, allow_pickle=False) as data:
        freq = data["freq"]
        alpha = data["alpha"]
        width = int(data["width"])

    if freq.ndim != 3 or freq.shape[2] != 3:
        raise ValueError("复数频谱数据不是三通道图像")
    if not np.issubdtype(freq.dtype, np.complexfloating):
        raise ValueError("频谱数据必须包含复数系数及其相位")

    if width < 1 or freq.shape[1] != width // 2 + 1:
        raise ValueError("频谱的宽度信息不匹配")

    pixels = np.fft.irfft2(freq, s=(freq.shape[0], width), axes=(0, 1))
    pixels = np.uint8(np.clip(np.rint(pixels), 0, 255))
    result = Image.fromarray(pixels, "RGB")

    if alpha.size:
        if alpha.shape != pixels.shape[:2]:
            raise ValueError("透明通道尺寸与图像不一致")
        result = result.convert("RGBA")
        result.putalpha(Image.fromarray(alpha, "L"))

    result.save(output, format="PNG")


def main():
    parser = Parser(
        description="FFT 提取完整尺寸的频谱水印图；IFFT 从复数频谱还原普通照片。"
    )
    parser.add_argument(
        "image", type=Path,
        help="FFT：输入照片；IFFT：输入 FFT 生成的 .npz 文件",
    )
    parser.add_argument(
        "-m", "--mode", choices=("fft", "ifft"), default=None,
        help="省略时仅生成频谱 PNG；显式 fft 同时生成 NPZ；ifft 从 NPZ 恢复照片",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="输出 PNG 路径，默认自动命名",
    )
    args = parser.parse_args()
    explicit_fft = args.mode == "fft"
    if args.mode is None:
        args.mode = "fft"

    if not args.image.is_file():
        parser.error(f"输入文件不存在：{args.image}")
    if args.mode == "ifft" and args.image.suffix.lower() != ".npz":
        parser.error("IFFT 需要 FFT 模式生成的 .npz 文件；频谱 PNG 缺少相位信息")

    suffix = "_spectrum.png" if args.mode == "fft" else "_restored.png"
    output = args.output or args.image.with_name(args.image.stem + suffix)
    if output.suffix.lower() != ".png":
        parser.error("输出文件必须为 .png")
    if output.resolve() == args.image.resolve():
        parser.error("不能覆盖输入文件")

    if args.mode == "fft":
        if explicit_fft:
            data_path = output.with_suffix(".npz")
            if data_path.resolve() == args.image.resolve():
                parser.error("频谱数据文件不能覆盖输入文件")
        enable_heif_if_needed(args.image, parser)

    try:
        if args.mode == "fft":
            data_path = fft_mode(args.image, output, save_npz=explicit_fft)
            print(f"频谱图：{output}")
            if data_path is not None:
                print(f"复数频谱：{data_path}")
        else:
            ifft_mode(args.image, output)
            print(f"恢复照片：{output}")
    except (OSError, ValueError, MemoryError, KeyError) as exc:
        parser.exit(1, f"处理失败：{exc}\n")


if __name__ == "__main__":
    main()
