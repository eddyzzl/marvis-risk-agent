#!/usr/bin/env python3
"""Generate the MARVIS mascot glow/breathing animation frames.

VD-5: the mascot's brand asset is the real logo pixels — the requirement is
"animate the real logo pixels", not a hand-drawn redraw. This script derives
every frame from ``marvis/static/brand/marvis-logo.png`` by compositing a
Gaussian-blurred glow halo (sourced from the logo's own alpha mask) behind
the untouched logo pixels, with the halo's blur radius/opacity breathing
across the sequence. No pixel of the logo artwork itself is repainted.

Usage::

    python scripts/generate_marvis_glow_animation.py

Writes numbered PNG frames plus an animated WebP loop to
``marvis/static/brand/glow/``. Safe to re-run; it always regenerates every
output file from the source logo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "marvis" / "static" / "brand" / "marvis-logo.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "marvis" / "static" / "brand" / "glow"

# One full breathe cycle: halo blur radius and peak opacity rise then fall.
# Values are (blur_radius_px, glow_opacity_0_to_1, logo_brightness_0_to_1_extra).
_BREATHE_CURVE: tuple[tuple[float, float, float], ...] = (
    (8.0, 0.40, 0.00),
    (12.0, 0.55, 0.04),
    (16.0, 0.70, 0.08),
    (19.0, 0.82, 0.11),
    (21.0, 0.90, 0.13),
    (19.0, 0.82, 0.11),
    (16.0, 0.70, 0.08),
    (12.0, 0.55, 0.04),
    (8.0, 0.40, 0.00),
    (5.0, 0.26, 0.00),
)

# Glow tint sourced from the platform's own accent token family (--accent /
# --tone-modeling in styles.css) rather than an invented color, so the halo
# reads as "MARVIS light", not a generic drop-shadow.
_GLOW_RGB = (10, 132, 255)

_CANVAS_PADDING = 64


# The mascot renders at ~102px in the UI; cap the working resolution well
# above that (retina-safe) without carrying full 512px source weight into
# every frame + the animated loop.
_MAX_LOGO_DIMENSION = 256


def _load_logo(source: Path) -> Image.Image:
    logo = Image.open(source).convert("RGBA")
    if max(logo.size) > _MAX_LOGO_DIMENSION:
        scale = _MAX_LOGO_DIMENSION / max(logo.size)
        new_size = (round(logo.width * scale), round(logo.height * scale))
        logo = logo.resize(new_size, Image.LANCZOS)
    return logo


def _padded_canvas_size(logo: Image.Image) -> tuple[int, int]:
    return (logo.width + _CANVAS_PADDING * 2, logo.height + _CANVAS_PADDING * 2)


def _glow_layer(logo: Image.Image, *, blur_radius: float, opacity: float) -> Image.Image:
    """Build a blurred glow halo from the logo's own alpha mask (real pixels
    in, real pixels out — only the mask is blurred, never redrawn)."""
    alpha = logo.getchannel("A")
    tinted = Image.new("RGBA", logo.size, (*_GLOW_RGB, 0))
    tinted.putalpha(alpha)
    blurred = tinted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if opacity < 1.0:
        r, g, b, a = blurred.split()
        a = a.point(lambda v: int(v * opacity))
        blurred = Image.merge("RGBA", (r, g, b, a))
    return blurred


def _brighten(logo: Image.Image, amount: float) -> Image.Image:
    """Lift the logo's own RGB channels toward white by `amount` (0-1),
    keeping its alpha mask untouched — a breathing highlight on the real
    artwork, not an overlay."""
    if amount <= 0:
        return logo
    r, g, b, a = logo.split()
    white = Image.new("L", logo.size, 255)

    def _lift(channel: Image.Image) -> Image.Image:
        lifted = ImageChops.add(channel, white, scale=1.0 / amount) if amount < 1 else white
        return Image.blend(channel, lifted, amount)

    return Image.merge("RGBA", (_lift(r), _lift(g), _lift(b), a))


def _compose_frame(logo: Image.Image, *, blur_radius: float, glow_opacity: float, brighten_amount: float) -> Image.Image:
    canvas_size = _padded_canvas_size(logo)
    frame = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    offset = (_CANVAS_PADDING, _CANVAS_PADDING)

    glow = _glow_layer(logo, blur_radius=blur_radius, opacity=glow_opacity)
    glow_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    glow_canvas.paste(glow, offset, glow)
    frame = Image.alpha_composite(frame, glow_canvas)

    logo_frame = _brighten(logo, brighten_amount)
    logo_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    logo_canvas.paste(logo_frame, offset, logo_frame)
    frame = Image.alpha_composite(frame, logo_canvas)
    return frame


def generate_frames(source: Path, output_dir: Path) -> list[Path]:
    logo = _load_logo(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    frames: list[Image.Image] = []
    for index, (blur_radius, glow_opacity, brighten_amount) in enumerate(_BREATHE_CURVE):
        frame = _compose_frame(
            logo,
            blur_radius=blur_radius,
            glow_opacity=glow_opacity,
            brighten_amount=brighten_amount,
        )
        frames.append(frame)
        frame_path = output_dir / f"marvis-glow-{index:02d}.png"
        frame.save(frame_path)
        frame_paths.append(frame_path)

    loop_path = output_dir / "marvis-glow-loop.webp"
    frame_duration_ms = 140
    frames[0].save(
        loop_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        lossless=False,
        quality=90,
        method=6,
    )
    frame_paths.append(loop_path)
    return frame_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source logo PNG (default: marvis/static/brand/marvis-logo.png)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write frames + loop into")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source logo not found: {args.source}")

    written = generate_frames(args.source, args.output_dir)
    for path in written:
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
