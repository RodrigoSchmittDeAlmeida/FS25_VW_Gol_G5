#!/usr/bin/env python3
"""Generate DXT5 (BC3) DDS assets with full mipmap chains for FS25_VW_Gol_G5."""

from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
GLB_PATH = ROOT_DIR / "source_assets" / "gol_g5.glb"
PREVIEW_PATH = ROOT_DIR / "source_assets" / "gol_g5_preview.png"
MOD_DIR = ROOT_DIR / "FS25_VW_Gol_G5"
TEXTURES_DIR = MOD_DIR / "vehicles" / "vwGolG5" / "textures"

# DDS header constants
DDS_MAGIC = b"DDS "
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000

DDPF_FOURCC = 0x4

DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000


def _rgb565_to_rgb888(c565: np.ndarray) -> np.ndarray:
    """Expand uint16 RGB565 values to float32 RGB888 (shape ..., 3)."""
    r5 = (c565 >> 11) & 0x1F
    g6 = (c565 >> 5) & 0x3F
    b5 = c565 & 0x1F
    r = (r5 << 3) | (r5 >> 2)
    g = (g6 << 2) | (g6 >> 4)
    b = (b5 << 3) | (b5 >> 2)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _rgb888_to_rgb565(rgb: np.ndarray) -> np.ndarray:
    """Quantize float32/uint8 RGB (shape ..., 3) to uint16 RGB565."""
    rgb_c = np.clip(rgb, 0.0, 255.0)
    r5 = np.rint(rgb_c[..., 0] * (31.0 / 255.0)).astype(np.uint16)
    g6 = np.rint(rgb_c[..., 1] * (63.0 / 255.0)).astype(np.uint16)
    b5 = np.rint(rgb_c[..., 2] * (31.0 / 255.0)).astype(np.uint16)
    return (r5 << 11) | (g6 << 5) | b5


def encode_dxt5_mip(rgba: np.ndarray) -> bytes:
    """Encode an (H, W, 4) uint8 RGBA image into raw DXT5 (BC3) blocks."""
    h, w, _ = rgba.shape
    bw = max(1, (w + 3) // 4)
    bh = max(1, (h + 3) // 4)
    pad_w = bw * 4 - w
    pad_h = bh * 4 - h
    if pad_w > 0 or pad_h > 0:
        rgba = np.pad(rgba, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    # Reshape into (N, 16, 4) blocks where N = bh * bw
    blocks = rgba.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4)
    n_blocks = blocks.shape[0]

    # --- Alpha Block (8 bytes per block) ---
    alpha = blocks[:, :, 3]  # (N, 16) uint8
    a0 = alpha.max(axis=1)
    a1 = alpha.min(axis=1)

    a0_f = a0.astype(np.float32)
    a1_f = a1.astype(np.float32)
    # 8-level palette when a0 > a1 (shape N, 8)
    alpha_pal = np.stack(
        [
            a0_f,
            a1_f,
            (6.0 * a0_f + 1.0 * a1_f) / 7.0,
            (5.0 * a0_f + 2.0 * a1_f) / 7.0,
            (4.0 * a0_f + 3.0 * a1_f) / 7.0,
            (3.0 * a0_f + 4.0 * a1_f) / 7.0,
            (2.0 * a0_f + 5.0 * a1_f) / 7.0,
            (1.0 * a0_f + 6.0 * a1_f) / 7.0,
        ],
        axis=1,
    )
    a_diff = np.abs(alpha[:, :, None].astype(np.float32) - alpha_pal[:, None, :])
    a_idx = np.argmin(a_diff, axis=2).astype(np.uint64)  # (N, 16)
    # When a0 == a1, all indices should be 0
    same_alpha = a0 == a1
    a_idx[same_alpha] = 0

    shifts_a = (np.arange(16, dtype=np.uint64) * 3)[None, :]
    packed_a = np.bitwise_or.reduce(a_idx << shifts_a, axis=1)  # (N,) uint64 (48 bits used)

    alpha_bytes = np.empty((n_blocks, 8), dtype=np.uint8)
    alpha_bytes[:, 0] = a0
    alpha_bytes[:, 1] = a1
    for b in range(6):
        alpha_bytes[:, 2 + b] = ((packed_a >> (8 * b)) & 0xFF).astype(np.uint8)

    # --- Color Block (8 bytes per block) ---
    rgb = blocks[:, :, :3].astype(np.float32)  # (N, 16, 3)

    # Select endpoints along the dominant diagonal of the block's RGB bounding box
    rgb_min = rgb.min(axis=1)  # (N, 3)
    rgb_max = rgb.max(axis=1)  # (N, 3)

    # Refine endpoints by choosing the corner pair maximizing projection spread
    # Standard fast bbox + 1/16 inset improves RMS error on smooth gradients
    inset = (rgb_max - rgb_min) * (1.0 / 16.0)
    ep0_f = np.clip(rgb_max - inset, 0.0, 255.0)
    ep1_f = np.clip(rgb_min + inset, 0.0, 255.0)

    c0 = _rgb888_to_rgb565(ep0_f)
    c1 = _rgb888_to_rgb565(ep1_f)

    # Ensure c0 > c1 whenever c0 != c1 so all decoders use 4-color mode
    swap = c0 < c1
    if np.any(swap):
        c0_s = np.where(swap, c1, c0)
        c1_s = np.where(swap, c0, c1)
        c0, c1 = c0_s, c1_s

    p0 = _rgb565_to_rgb888(c0)
    p1 = _rgb565_to_rgb888(c1)
    p2 = (2.0 * p0 + p1) / 3.0
    p3 = (p0 + 2.0 * p1) / 3.0
    color_pal = np.stack([p0, p1, p2, p3], axis=1)  # (N, 4, 3)

    c_diff = rgb[:, :, None, :] - color_pal[:, None, :, :]
    c_dist = np.sum(c_diff * c_diff, axis=3)  # (N, 16, 4)
    c_idx = np.argmin(c_dist, axis=2).astype(np.uint32)  # (N, 16)
    c_idx[c0 == c1] = 0

    shifts_c = (np.arange(16, dtype=np.uint32) * 2)[None, :]
    packed_c = np.bitwise_or.reduce(c_idx << shifts_c, axis=1)  # (N,) uint32

    color_bytes = np.empty((n_blocks, 8), dtype=np.uint8)
    color_bytes[:, 0] = (c0 & 0xFF).astype(np.uint8)
    color_bytes[:, 1] = ((c0 >> 8) & 0xFF).astype(np.uint8)
    color_bytes[:, 2] = (c1 & 0xFF).astype(np.uint8)
    color_bytes[:, 3] = ((c1 >> 8) & 0xFF).astype(np.uint8)
    color_bytes[:, 4] = (packed_c & 0xFF).astype(np.uint8)
    color_bytes[:, 5] = ((packed_c >> 8) & 0xFF).astype(np.uint8)
    color_bytes[:, 6] = ((packed_c >> 16) & 0xFF).astype(np.uint8)
    color_bytes[:, 7] = ((packed_c >> 24) & 0xFF).astype(np.uint8)

    combined = np.concatenate([alpha_bytes, color_bytes], axis=1)  # (N, 16)
    return combined.tobytes()


def build_dds_header(width: int, height: int, mip_count: int) -> bytes:
    """Build a 128-byte DDS DXT5 header with full mipmap flags."""
    flags = (
        DDSD_CAPS
        | DDSD_HEIGHT
        | DDSD_WIDTH
        | DDSD_PIXELFORMAT
        | DDSD_MIPMAPCOUNT
        | DDSD_LINEARSIZE
    )
    linear_size = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * 16
    caps = DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    reserved1 = (0,) * 11

    header = struct.pack(
        "<4s7I11I2I4s5I5I",
        DDS_MAGIC,
        124,  # dwSize
        flags,
        height,
        width,
        linear_size,
        0,  # dwDepth
        mip_count,
        *reserved1,
        32,  # ddspf.dwSize
        DDPF_FOURCC,  # ddspf.dwFlags
        b"DXT5",  # ddspf.dwFourCC
        0, 0, 0, 0, 0,  # bitcount + masks
        caps,
        0, 0, 0, 0,  # dwCaps2..4, dwReserved2
    )
    assert len(header) == 128
    return header


def _downsample_premultiplied(img: Image.Image, mw: int, mh: int) -> Image.Image:
    """Downsample RGBA image with premultiplied alpha to avoid dark/halo fringes."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    premul = np.concatenate([arr[:, :, :3] * alpha, arr[:, :, 3:4]], axis=2)
    premul_img = Image.fromarray(np.clip(np.rint(premul), 0, 255).astype(np.uint8), mode="RGBA")
    resized = premul_img.resize((mw, mh), Image.Resampling.LANCZOS)
    r_arr = np.asarray(resized, dtype=np.float32)
    r_a = r_arr[:, :, 3:4] / 255.0
    rgb = np.where(r_a > 1e-4, r_arr[:, :, :3] / np.maximum(r_a, 1e-4), 0.0)
    out = np.concatenate([np.clip(np.rint(rgb), 0, 255), r_arr[:, :, 3:4]], axis=2).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def save_dxt5_dds(img: Image.Image, out_path: Path) -> tuple[int, int, int, int]:
    """Encode PIL Image as DXT5 DDS with complete mipmap chain down to 1x1."""
    rgba_img = img.convert("RGBA")
    w, h = rgba_img.size
    if (w & (w - 1)) != 0 or (h & (h - 1)) != 0:
        raise ValueError(f"DDS dimensions must be powers of 2, got {w}x{h}")

    mip_count = int(math.log2(max(w, h))) + 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = [build_dds_header(w, h, mip_count)]
    for level in range(mip_count):
        mw = max(1, w >> level)
        mh = max(1, h >> level)
        mip_img = rgba_img if level == 0 else _downsample_premultiplied(rgba_img, mw, mh)
        chunks.append(encode_dxt5_mip(np.asarray(mip_img, dtype=np.uint8)))

    payload = b"".join(chunks)
    out_path.write_bytes(payload)
    return w, h, mip_count, len(payload)


def extract_glb_textures() -> list[tuple[Path, int, int, int, int]]:
    """Extract the 6 embedded PNG textures from gol_g5.glb into FS25 DDS files."""
    data = GLB_PATH.read_bytes()
    magic, version, _ = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2:
        raise RuntimeError(f"Invalid GLB header in {GLB_PATH}")

    jlen, jtype = struct.unpack("<II", data[12:20])
    if jtype != 0x4E4F534A:
        raise RuntimeError("First GLB chunk is not JSON")
    gltf = json.loads(data[20 : 20 + jlen].decode("utf-8"))

    bin_header_off = 20 + jlen
    blen, btype = struct.unpack("<II", data[bin_header_off : bin_header_off + 8])
    if btype != 0x004E4942:
        raise RuntimeError("Second GLB chunk is not BIN")
    bin_chunk = data[bin_header_off + 8 : bin_header_off + 8 + blen]

    results = []
    for img_meta in gltf.get("images", []):
        name = img_meta["name"]
        bv = gltf["bufferViews"][img_meta["bufferView"]]
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        png_bytes = bin_chunk[offset : offset + length]
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        out_file = TEXTURES_DIR / f"{name}.dds"
        w, h, mips, size_b = save_dxt5_dds(pil_img, out_file)
        results.append((out_file, w, h, mips, size_b))
    return results


def _make_studio_backdrop(size: int) -> Image.Image:
    """Create a dark automotive studio gradient backdrop with subtle vignette."""
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :]
    # Vertical studio gradient (#16191F top -> #232832 horizon -> #111419 floor)
    horizon = np.exp(-((y - 0.56) ** 2) / 0.08)
    r = 18.0 + 18.0 * horizon
    g = 20.0 + 20.0 * horizon
    b = 26.0 + 24.0 * horizon
    vignette = 1.0 - 0.28 * np.clip(x**2 + (y * 2.0 - 1.0) ** 2, 0.0, 1.2)
    rgb = np.stack([r * vignette, g * vignette, b * vignette], axis=-1)
    alpha = np.full((size, size, 1), 255.0, dtype=np.float32)
    arr = np.clip(np.rint(np.concatenate([rgb, alpha], axis=-1)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _compose_car_card(preview: Image.Image, target_size: int, crop_box: tuple[int, int, int, int]) -> Image.Image:
    """Frame the preview render cleanly onto a square studio card."""
    cropped = preview.convert("RGBA").crop(crop_box)
    canvas = _make_studio_backdrop(target_size)

    # Fit cropped render horizontally with small margin
    margin = int(target_size * 0.04)
    inner_w = target_size - 2 * margin
    scale = inner_w / cropped.width
    inner_h = int(round(cropped.height * scale))
    resized = cropped.resize((inner_w, inner_h), Image.Resampling.LANCZOS)

    # Soft alpha mask on top/bottom edges for smooth blend with studio backdrop
    mask_arr = np.full((inner_h, inner_w), 255.0, dtype=np.float32)
    fade = max(8, int(inner_h * 0.10))
    ramp = np.linspace(0.0, 255.0, fade, dtype=np.float32)
    mask_arr[:fade, :] *= (ramp[:, None] / 255.0)
    mask_arr[-fade:, :] *= (ramp[::-1, None] / 255.0)
    side_fade = max(6, int(inner_w * 0.04))
    sramp = np.linspace(0.0, 255.0, side_fade, dtype=np.float32)
    mask_arr[:, :side_fade] *= (sramp[None, :] / 255.0)
    mask_arr[:, -side_fade:] *= (sramp[None, ::-1] / 255.0)
    mask_img = Image.fromarray(np.clip(np.rint(mask_arr), 0, 255).astype(np.uint8), mode="L")

    paste_x = margin
    paste_y = (target_size - inner_h) // 2
    canvas.paste(resized, (paste_x, paste_y), mask_img)
    return canvas


def generate_store_and_icon() -> list[tuple[Path, int, int, int, int]]:
    """Generate icon_golG5.dds (256x256) and store_golG5.dds (512x512) from preview."""
    preview = Image.open(PREVIEW_PATH).convert("RGBA")
    # Crop tightly around the Gol G5 silhouette in the 1280x720 preview
    store_crop = (0, 20, 1020, 710)
    icon_crop = (0, 35, 995, 700)

    store_img = _compose_car_card(preview, 512, store_crop)
    icon_img = _compose_car_card(preview, 256, icon_crop)

    results = []
    for path, im in [
        (MOD_DIR / "store_golG5.dds", store_img),
        (MOD_DIR / "icon_golG5.dds", icon_img),
    ]:
        w, h, mips, size_b = save_dxt5_dds(im, path)
        results.append((path, w, h, mips, size_b))
    return results


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/seguisb.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
    ]
    for fp in candidates:
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)
    return ImageFont.load_default()


def generate_brand_logo() -> tuple[Path, int, int, int, int]:
    """Generate 512x256 white vector-style Volkswagen brand logo on transparent RGBA."""
    # Render at 4x supersampling (2048x1024) for crisp anti-aliased edges
    ss = 4
    w, h = 512 * ss, 256 * ss
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    white = (255, 255, 255, 255)
    clear = (0, 0, 0, 0)

    cx, cy = w // 2, int(h * 0.39)
    outer_r = int(h * 0.31)
    ring_w = max(8, int(outer_r * 0.115))

    # Outer ring
    draw.ellipse((cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r), outline=white, width=ring_w)

    # Geometric VW emblem inside the circle
    inner_r = outer_r - ring_w // 2
    stroke = max(8, int(outer_r * 0.115))

    # Upper V
    v_top_l = (cx - int(inner_r * 0.44), cy - int(inner_r * 0.80))
    v_top_r = (cx + int(inner_r * 0.44), cy - int(inner_r * 0.80))
    v_bot = (cx, cy - int(inner_r * 0.04))
    draw.line([v_top_l, v_bot, v_top_r], fill=white, width=stroke, joint="curve")

    # Lower W
    w_top_l = (cx - int(inner_r * 0.72), cy - int(inner_r * 0.56))
    w_bot_l = (cx - int(inner_r * 0.33), cy + int(inner_r * 0.82))
    w_mid = (cx, cy + int(inner_r * 0.14))
    w_bot_r = (cx + int(inner_r * 0.33), cy + int(inner_r * 0.82))
    w_top_r = (cx + int(inner_r * 0.72), cy - int(inner_r * 0.56))
    draw.line([w_top_l, w_bot_l, w_mid, w_bot_r, w_top_r], fill=white, width=stroke, joint="curve")

    # Clip any stroke overhang cleanly to the outer circle radius
    circle_mask = Image.new("L", (w, h), 0)
    cm_draw = ImageDraw.Draw(circle_mask)
    cm_draw.ellipse((cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r), fill=255)
    # Apply mask only to the emblem region
    emblem_region = img.copy()
    blank = Image.new("RGBA", (w, h), clear)
    img = Image.composite(emblem_region, blank, circle_mask)
    draw = ImageDraw.Draw(img)

    # Clean wordmark "VOLKSWAGEN" centered beneath the emblem
    font = _load_font(int(40 * ss))
    wordmark = "V O L K S W A G E N"
    bbox = draw.textbbox((0, 0), wordmark, font=font)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    ty = int(h * 0.76)
    draw.text((tx, ty), wordmark, fill=white, font=font)

    final_img = _downsample_premultiplied(img, 512, 256)
    out_path = MOD_DIR / "brand_vw.dds"
    bw, bh, mips, size_b = save_dxt5_dds(final_img, out_path)
    return out_path, bw, bh, mips, size_b


def main() -> None:
    generated = []
    generated.extend(extract_glb_textures())
    generated.extend(generate_store_and_icon())
    generated.append(generate_brand_logo())

    print(f"Generated {len(generated)} DXT5 DDS assets:")
    for path, w, h, mips, size_b in generated:
        rel = path.relative_to(ROOT_DIR)
        print(f"  - {rel.as_posix()}: {w}x{h}, {mips} mipmaps, {size_b} bytes")


if __name__ == "__main__":
    main()
