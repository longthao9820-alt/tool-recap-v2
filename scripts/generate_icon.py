"""Generate high-quality icon assets for ToolRecap V2 using Pillow.
Creates icon.png, icon-256.png, and icon.ico (multi-resolution).
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw


def create_app_icon(size: int = 512) -> Image.Image:
    # 32-bit RGBA image
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Rounded rectangle background (gradient-like modern slate blue / violet)
    margin = int(size * 0.06)
    radius = int(size * 0.22)
    bg_color = (24, 28, 42, 255)  # Dark slate background
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=bg_color,
    )

    # Subtle inner border
    border_color = (70, 85, 125, 255)
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        outline=border_color,
        width=int(size * 0.02),
    )

    # 2. Film strip / Video card accent (Cyan to Indigo accent)
    # Draw stylized film reel / video frame in top-left to center
    film_left = int(size * 0.20)
    film_top = int(size * 0.22)
    film_right = int(size * 0.80)
    film_bottom = int(size * 0.62)
    film_radius = int(size * 0.08)

    # Gradient film background
    draw.rounded_rectangle(
        [film_left, film_top, film_right, film_bottom],
        radius=film_radius,
        fill=(35, 45, 75, 255),
        outline=(59, 130, 246, 255),  # Blue-500
        width=int(size * 0.025),
    )

    # Film sprockets (top and bottom perforations)
    sprocket_w = int(size * 0.05)
    sprocket_h = int(size * 0.035)
    sprocket_r = int(size * 0.01)
    sprocket_y_top = film_top + int(size * 0.03)
    sprocket_y_bot = film_bottom - int(size * 0.03) - sprocket_h

    for i in range(5):
        sp_x = film_left + int(size * 0.08) + i * int(size * 0.11)
        draw.rounded_rectangle(
            [sp_x, sprocket_y_top, sp_x + sprocket_w, sprocket_y_top + sprocket_h],
            radius=sprocket_r,
            fill=(15, 20, 35, 255),
        )
        draw.rounded_rectangle(
            [sp_x, sprocket_y_bot, sp_x + sprocket_w, sprocket_y_bot + sprocket_h],
            radius=sprocket_r,
            fill=(15, 20, 35, 255),
        )

    # Play triangle inside film window
    play_cx = int(size * 0.48)
    play_cy = int(size * 0.42)
    play_s = int(size * 0.09)
    play_points = [
        (play_cx - play_s * 0.7, play_cy - play_s),
        (play_cx + play_s * 1.1, play_cy),
        (play_cx - play_s * 0.7, play_cy + play_s),
    ]
    draw.polygon(play_points, fill=(96, 165, 250, 255))  # Blue-400

    # 3. Audio waveform / Voice narration bars at the bottom
    wave_y_base = int(size * 0.78)
    bar_width = int(size * 0.035)
    bar_gap = int(size * 0.022)
    bar_heights = [0.06, 0.11, 0.18, 0.25, 0.16, 0.28, 0.20, 0.13, 0.07]
    total_wave_w = len(bar_heights) * bar_width + (len(bar_heights) - 1) * bar_gap
    start_x = int((size - total_wave_w) / 2)

    # Wave gradient colors: Violet to Cyan
    wave_colors = [
        (168, 85, 247, 255),  # Purple
        (192, 132, 252, 255),
        (236, 72, 153, 255),  # Pink
        (244, 114, 182, 255),
        (56, 189, 248, 255),  # Sky
        (14, 165, 233, 255),
        (59, 130, 246, 255),  # Blue
        (99, 102, 241, 255),  # Indigo
        (139, 92, 246, 255),  # Violet
    ]

    for i, (h_ratio, color) in enumerate(zip(bar_heights, wave_colors)):
        bx = start_x + i * (bar_width + bar_gap)
        h = int(size * h_ratio)
        by1 = wave_y_base - h // 2
        by2 = wave_y_base + h // 2
        draw.rounded_rectangle(
            [bx, by1, bx + bar_width, by2],
            radius=bar_width // 2,
            fill=color,
        )

    # 4. "V2" badge at bottom right
    badge_x = int(size * 0.70)
    badge_y = int(size * 0.68)
    badge_w = int(size * 0.20)
    badge_h = int(size * 0.16)
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=int(badge_h * 0.35),
        fill=(16, 185, 129, 255),  # Emerald-500
        outline=(255, 255, 255, 220),
        width=int(size * 0.015),
    )

    # Stylized "V2" text strokes on badge
    # Draw 'V'
    vx = badge_x + int(badge_w * 0.25)
    vy = badge_y + int(badge_h * 0.25)
    vw = int(badge_w * 0.25)
    vh = int(badge_h * 0.5)
    draw.line([(vx, vy), (vx + vw // 2, vy + vh)], fill=(255, 255, 255, 255), width=int(size * 0.02))
    draw.line([(vx + vw // 2, vy + vh), (vx + vw, vy)], fill=(255, 255, 255, 255), width=int(size * 0.02))

    # Draw '2'
    tx = badge_x + int(badge_w * 0.60)
    ty = badge_y + int(badge_h * 0.25)
    tw = int(badge_w * 0.22)
    th = int(badge_h * 0.5)
    # top arc / line, diagonal, bottom horizontal
    draw.line([(tx, ty), (tx + tw, ty)], fill=(255, 255, 255, 255), width=int(size * 0.02))
    draw.line([(tx + tw, ty), (tx + tw, ty + th // 2)], fill=(255, 255, 255, 255), width=int(size * 0.02))
    draw.line([(tx + tw, ty + th // 2), (tx, ty + th)], fill=(255, 255, 255, 255), width=int(size * 0.02))
    draw.line([(tx, ty + th), (tx + tw, ty + th)], fill=(255, 255, 255, 255), width=int(size * 0.02))

    return img


def generate_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_img = create_app_icon(512)

    # Save PNGs
    base_img.save(output_dir / "icon.png", "PNG")
    img_256 = base_img.resize((256, 256), Image.Resampling.LANCZOS)
    img_256.save(output_dir / "icon-256.png", "PNG")

    # Save multi-res ICO
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base_img.save(
        output_dir / "icon.ico",
        format="ICO",
        sizes=sizes,
    )
    print(f"Icon assets generated successfully in {output_dir}")


if __name__ == "__main__":
    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    generate_assets(assets_dir)
