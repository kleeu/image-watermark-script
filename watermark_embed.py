#!/usr/bin/env python3
"""Add a dense diagonal visible watermark and a readable FFT-spectrum watermark."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def chinese_font(path=None):
    """
    Prefer plain sans / hei-style Chinese fonts.
    Do not default to Songti / serif fonts.
    """
    candidates = [path] if path else [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",

        # Windows
        "C:/Windows/Fonts/msyh.ttc",     # Microsoft YaHei
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",   # SimHei

        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)

    raise ValueError("找不到合适的中文黑体字体，请用 --font 指定 .ttf/.ttc 文件")


def make_text_stamp(text, font_path, font_size, opacity, fill_color):
    """
    Create one visible watermark tile with:
    - main text
    - subtle stroke
    - subtle shadow
    """
    font = ImageFont.truetype(font_path, font_size)

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=max(1, font_size // 22))

    stroke = max(1, font_size // 22)
    shadow_offset = max(1, font_size // 18)
    pad = max(8, font_size // 4)

    w = (bbox[2] - bbox[0]) + 2 * pad + shadow_offset
    h = (bbox[3] - bbox[1]) + 2 * pad + shadow_offset
    stamp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)

    x = pad - bbox[0]
    y = pad - bbox[1]

    # Shadow
    draw.text(
        (x + shadow_offset, y + shadow_offset),
        text,
        font=font,
        fill=(0, 0, 0, round(255 * opacity * 0.32)),
    )

    # Stroke + main text
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(*fill_color, round(255 * opacity)),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, round(255 * opacity * 0.45)),
    )

    return stamp


def visible_watermark(rgb, text, font_path, opacity):
    """
    Dense diagonal visible watermark.
    """
    if opacity == 0:
        return rgb

    w, h = rgb.size

    # Slightly smaller text so more copies fit in the image.
    font_size = max(16, min(w, h) // 28)

    # Shrink long captions so each tile is not too wide.
    while font_size > 12:
        font = ImageFont.truetype(font_path, font_size)
        bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
        if bb[2] - bb[0] <= w * 0.50:
            break
        font_size -= 2

    # Decide text color from average brightness.
    small = rgb.convert("L").resize((32, 32))
    mean_luma = np.asarray(small).mean()

    # Slightly brighter or darker text, but not too strong.
    if mean_luma < 128:
        fill_color = (245, 245, 245)
    else:
        fill_color = (35, 35, 35)

    stamp = make_text_stamp(text, font_path, font_size, opacity, fill_color)
    stamp = stamp.rotate(30, expand=True, resample=Image.Resampling.BICUBIC)

    sx = max(1, int(stamp.width * 0.78))
    sy = max(1, int(stamp.height * 0.91))

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Start far outside bounds to ensure full coverage.
    row_index = 0
    for y in range(-2 * sy, h + 2 * sy, sy):
        shift = sx // 2 if row_index % 2 else 0
        for x in range(-2 * sx, w + 2 * sx, sx):
            layer.alpha_composite(stamp, (x + shift, y))
        row_index += 1

    return Image.alpha_composite(rgb.convert("RGBA"), layer).convert("RGB")


def spectrum_text_mask(size, text, font_path):
    """Draw one normal-reading text line in upper FFT space."""
    w, h = size
    mask_image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_image)

    max_w, max_h = int(w * 0.82), int(h * 0.14)
    size_px = max(12, int(min(w, h) * 0.12))

    while size_px >= 12:
        font = ImageFont.truetype(font_path, size_px)
        bb = draw.textbbox((0, 0), text, font=font)
        if (bb[2] - bb[0] <= max_w) and (bb[3] - bb[1] <= max_h):
            break
        size_px -= 1

    if size_px < 12:
        raise ValueError("水印文字过长或图片过小，无法在频谱中清晰放下文字")

    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    x = (w - bw) // 2 - bb[0]
    y = round(h * 0.27) - bh // 2 - bb[1]

    draw.text((x, y), text, font=font, fill=255)
    return np.asarray(mask_image) > 127


def frequency_watermark(rgb, text, font_path, strength):
    if strength == 0:
        return rgb

    w, h = rgb.size
    if min(w, h) < 256:
        raise ValueError("频域水印需要图片短边至少为 256 像素")

    pixels = np.asarray(rgb, dtype=np.float32)
    luminance = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )

    freq = np.fft.fftshift(np.fft.fft2(luminance))
    mask = spectrum_text_mask((w, h), text, font_path)
    ys, xs = np.nonzero(mask)

    roi = np.abs(freq[int(h * 0.20):int(h * 0.34), int(w * 0.10):int(w * 0.90)])
    base = max(float(np.median(roi)), 1.0)
    delta = base * 14.0 * strength

    old = freq[ys, xs]
    phase = np.divide(
        old,
        np.abs(old),
        out=np.ones_like(old),
        where=np.abs(old) > 0,
    )
    changed = (np.abs(old) + delta) * phase
    freq[ys, xs] = changed

    mirror_y = (2 * (h // 2) - ys) % h
    mirror_x = (2 * (w // 2) - xs) % w
    freq[mirror_y, mirror_x] = np.conj(changed)

    modified = np.fft.ifft2(np.fft.ifftshift(freq)).real
    delta_y = (modified - luminance).astype(np.float32)

    output = np.empty_like(pixels, dtype=np.uint8)
    for i in range(3):
        output[:, :, i] = np.clip(
            np.rint(pixels[:, :, i] + delta_y),
            0,
            255,
        ).astype(np.uint8)

    return Image.fromarray(output, "RGB")


def enable_heif_if_needed(path, parser):
    if path.suffix.lower() not in {".heic", ".heif", ".heics", ".heifs", ".hif"}:
        return

    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        parser.exit(1, "读取 HEIC/HEIF 需要安装：python3 -m pip install pillow-heif\n")

    register_heif_opener(thumbnails=False)


def main():
    parser = Parser(description="添加高密度斜向可见水印与可在频谱中阅读的中文水印。")
    parser.add_argument(
        "image",
        type=Path,
        help="输入图片路径（JPEG、PNG、WebP、TIFF、HEIC/HEIF）",
    )
    parser.add_argument(
        "text",
        help="水印文字，例如：本照片仅供XX使用",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出 PNG 路径，默认：原文件名_watermarked.png",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.4,
        help="频域水印强度，0~1，默认 0.4",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.15,
        help="可见水印透明度，0~1，默认 0.15",
    )
    parser.add_argument(
        "--font",
        type=Path,
        help="中文字体文件路径，默认自动查找黑体字体",
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    if args.font and not args.font.is_file():
        parser.error(f"字体文件不存在：{args.font}")
    if not args.text.strip():
        parser.error("水印文字不能为空")

    for key in ("strength", "opacity"):
        value = getattr(args, key)
        if not np.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{key} 必须是 0~1 的数字")

    output = args.output or args.image.with_name(args.image.stem + "_watermarked.png")

    if output.suffix.lower() != ".png":
        parser.error("为避免有损压缩破坏频谱水印，输出文件必须为 .png")
    if output.resolve() == args.image.resolve():
        parser.error("不能覆盖原图，请指定不同的输出路径")

    enable_heif_if_needed(args.image, parser)

    try:
        font_path = chinese_font(args.font)

        with Image.open(args.image) as source:
            oriented = ImageOps.exif_transpose(source)
            has_alpha = ("A" in oriented.getbands() or "transparency" in oriented.info)
            alpha = oriented.convert("RGBA").getchannel("A") if has_alpha else None
            icc = source.info.get("icc_profile")
            rgb = oriented.convert("RGB")

        rgb = visible_watermark(rgb, args.text, font_path, args.opacity)
        rgb = frequency_watermark(rgb, args.text, font_path, args.strength)

        if alpha is not None:
            rgb = rgb.convert("RGBA")
            rgb.putalpha(alpha)

        save_options = {"icc_profile": icc} if icc else {}
        rgb.save(output, format="PNG", **save_options)

    except (OSError, ValueError, MemoryError) as exc:
        parser.exit(1, f"处理失败：{exc}\n")

    print(f"已保存：{output}")


if __name__ == "__main__":
    main()