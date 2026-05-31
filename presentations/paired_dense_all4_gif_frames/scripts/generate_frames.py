from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "frames"
W, H = 1600, 900

BG = "#f7f4ec"
PANEL = "#fffdf8"
INK = "#17212b"
MUTED = "#8f887e"
LINE = "#ddd4c6"
SOFT = "#ebe4d8"
TEAL = "#168a87"
TEAL_DARK = "#0c5d60"
BLUE = "#3f68a8"
AMBER = "#d59b2d"
GREEN = "#4c8c4b"
CORAL = "#c94f4f"

SOURCE_NAMES = ["S1", "S2", "S3", "S4"]
SOURCE_COLORS = ["#65b6b1", TEAL, BLUE, AMBER]
SOURCE_WEIGHTS = [0.174, 0.316, 0.268, 0.242]

STEPS = [
    ("setup", "setup"),
    ("dense_all4", "dense"),
    ("soft_reliability", "reliability"),
    ("geometric_pool", "pool"),
    ("clean_pass", "result"),
]

STEP_INDEX = {name: idx for idx, (_slug, name) in enumerate(STEPS)}


def active(stage: str, name: str) -> bool:
    return STEP_INDEX[stage] >= STEP_INDEX[name]


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ) if weight == "bold" else (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TINY = font(18)
F_SMALL = font(24)
F_BODY = font(32)
F_MED = font(40, "bold")
F_BIG = font(54, "bold")
F_NUM = font(68, "bold")


def rgb(color: str, opacity: float = 1.0) -> tuple[int, int, int]:
    color = color.lstrip("#")
    fg = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
    bg = tuple(int(BG.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return tuple(int(bg[i] * (1.0 - opacity) + fg[i] * opacity) for i in range(3))


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, color: str, fnt: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 2), text, fill=color, font=fnt)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 28) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, *, width: int = 6, opacity: float = 1.0) -> None:
    fill = rgb(color, opacity)
    draw.line((*start, *end), fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 18 + width
    p1 = (x1 - size * math.cos(angle - 0.45), y1 - size * math.sin(angle - 0.45))
    p2 = (x1 - size * math.cos(angle + 0.45), y1 - size * math.sin(angle + 0.45))
    draw.polygon([end, p1, p2], fill=fill)


def source_positions() -> list[tuple[int, int]]:
    return [(220, 220), (220, 360), (220, 500), (220, 640)]


def draw_source(draw: ImageDraw.ImageDraw, idx: int, stage: str) -> None:
    x, y = source_positions()[idx]
    color = SOURCE_COLORS[idx]
    selected = active(stage, "dense")
    r = 56
    draw.ellipse((x - r, y - r, x + r, y + r), fill=PANEL, outline=color, width=4)
    centered(draw, (x - r, y - r, x + r, y + r), SOURCE_NAMES[idx], INK, F_MED)
    if selected:
        draw.ellipse((x - r - 9, y - r - 9, x + r + 9, y + r + 9), outline=rgb(color, 0.7), width=4)
    if active(stage, "reliability"):
        w = SOURCE_WEIGHTS[idx]
        bar_x0, bar_x1 = 315, 430
        draw.rounded_rectangle((bar_x0, y - 14, bar_x1, y + 14), radius=14, fill="#ded8cc")
        draw.rounded_rectangle((bar_x0, y - 14, bar_x0 + int((bar_x1 - bar_x0) * w / 0.34), y + 14), radius=14, fill=color)


def draw_mixer(draw: ImageDraw.ImageDraw, stage: str) -> None:
    on = active(stage, "dense")
    outline = TEAL_DARK if on else LINE
    rounded(draw, (575, 285, 825, 575), PANEL if on else SOFT, outline, 5 if on else 2, 38)
    if active(stage, "reliability"):
        centered(draw, (600, 318, 800, 370), "reliability", TEAL_DARK, F_BODY)
        centered(draw, (600, 370, 800, 430), "weighted", TEAL_DARK, F_MED)
        centered(draw, (600, 440, 800, 515), "all4", TEAL_DARK, F_NUM)
    elif on:
        centered(draw, (600, 330, 800, 390), "dense", TEAL_DARK, F_MED)
        centered(draw, (600, 420, 800, 505), "all4", TEAL_DARK, F_NUM)
    else:
        centered(draw, (600, 355, 800, 500), "all4", MUTED, F_BIG)

    if on:
        for idx, (x, y) in enumerate(source_positions()):
            weight = SOURCE_WEIGHTS[idx] if active(stage, "reliability") else 0.25
            width = max(5, int(24 * weight))
            arrow(draw, (x + 72, y), (575, 430), SOURCE_COLORS[idx], width=width, opacity=0.88)


def draw_pool(draw: ImageDraw.ImageDraw, stage: str) -> None:
    on = active(stage, "pool")
    rounded(draw, (980, 310, 1195, 550), PANEL if on else SOFT, TEAL_DARK if on else LINE, 5 if on else 2, 34)
    if on:
        centered(draw, (1005, 340, 1170, 390), "generated", TEAL_DARK, F_BODY)
        centered(draw, (1005, 392, 1170, 438), "heads", TEAL_DARK, F_BODY)
        centered(draw, (1005, 455, 1170, 505), "geom pool", TEAL_DARK, F_BODY)
        arrow(draw, (825, 430), (980, 430), TEAL_DARK, width=8, opacity=0.9)
    else:
        centered(draw, (1005, 360, 1170, 505), "pool", MUTED, F_BIG)


def draw_target(draw: ImageDraw.ImageDraw, stage: str) -> None:
    result = active(stage, "result")
    x, y = 1380, 430
    rounded(draw, (x - 92, y - 92, x + 92, y + 92), PANEL, CORAL if active(stage, "pool") else LINE, 5 if active(stage, "pool") else 2, 34)
    centered(draw, (x - 70, y - 75, x + 70, y - 20), "T", CORAL if active(stage, "pool") else MUTED, F_BIG)
    for dx, dy in [(-35, 16), (-6, 34), (25, 11), (40, 43), (4, -3)]:
        draw.ellipse((x + dx - 7, y + dy - 7, x + dx + 7, y + dy + 7), fill=CORAL if active(stage, "pool") else MUTED)
    if active(stage, "pool"):
        arrow(draw, (1195, 430), (1288, 430), TEAL_DARK, width=8, opacity=0.9)
    if result:
        rounded(draw, (1165, 655, 1500, 760), PANEL, GREEN, 4, 26)
        draw.text((1195, 678), "BACC", fill=MUTED, font=F_BODY)
        draw.text((1300, 668), "0.8506", fill=GREEN, font=F_NUM)
        draw.text((1168, 780), "equal all4: 0.8235", fill=MUTED, font=F_SMALL)


def draw_claim(draw: ImageDraw.ImageDraw, stage: str) -> None:
    if not active(stage, "result"):
        return
    if stage == "result":
        rounded(draw, (90, 760, 845, 820), PANEL, GREEN, 3, 18)
        draw.text((120, 774), "PASS: dense generated-embedding aggregation", fill=GREEN, font=F_BODY)
        return
    rounded(draw, (90, 735, 760, 805), PANEL, TEAL, 3, 20)
    draw.text((120, 753), "all sources stay active", fill=TEAL_DARK, font=F_BODY)
    rounded(draw, (800, 735, 1510, 805), PANEL, CORAL, 3, 20)
    draw.text((830, 753), "not sparse routing or target-conditioned selection", fill=CORAL, font=F_BODY)


def draw_progress(draw: ImageDraw.ImageDraw, stage: str, frame_number: int) -> None:
    labels = ["sources", "dense", "weights", "pool", "PASS"]
    x0 = 410
    y = 845
    for idx, label in enumerate(labels):
        x = x0 + idx * 200
        on = idx <= STEP_INDEX[stage]
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=TEAL if on else LINE)
        if idx < len(labels) - 1:
            draw.line((x + 14, y, x + 186, y), fill=TEAL if on else LINE, width=3)
        if idx == STEP_INDEX[stage]:
            draw.text((x - 34, y + 18), label, fill=INK, font=F_TINY)
    draw.text((72, 832), f"{frame_number:02d}", fill=MUTED, font=F_SMALL)


def draw_frame(stage: str, frame_number: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    for x in range(80, W, 80):
        draw.line((x, 0, x, H), fill="#efe8dd", width=1)
    for y in range(80, H, 80):
        draw.line((0, y, W, y), fill="#efe8dd", width=1)
    rounded(draw, (54, 48, 1546, 825), PANEL, "#e4ddd1", 2, 34)

    for idx in range(4):
        draw_source(draw, idx, stage)
    draw_mixer(draw, stage)
    draw_pool(draw, stage)
    draw_target(draw, stage)
    draw_claim(draw, stage)
    draw_progress(draw, stage, frame_number)
    return img


def build_contact_sheet(paths: list[Path]) -> None:
    thumb_w, thumb_h = 480, 270
    margin = 32
    cols = 1
    rows = len(paths)
    sheet = Image.new("RGB", (thumb_w + margin * 2, rows * (thumb_h + 54) + margin), BG)
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(paths):
        im = Image.open(path).resize((thumb_w, thumb_h), resample=Image.Resampling.LANCZOS)
        x = margin
        y = margin + idx * (thumb_h + 54)
        sheet.paste(im, (x, y))
        draw.text((x, y + thumb_h + 10), path.name, fill=INK, font=F_TINY)
    sheet.save(ROOT / "contact_sheet.png")


def build_preview_gif(paths: list[Path]) -> None:
    frames = [Image.open(path).resize((960, 540), resample=Image.Resampling.LANCZOS) for path in paths]
    frames[0].save(
        ROOT / "paired_dense_all4_reliability_preview.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1250,
        loop=0,
    )


def main() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()
    paths: list[Path] = []
    for idx, (slug, stage) in enumerate(STEPS, start=1):
        path = FRAME_DIR / f"frame_{idx:02d}_{slug}.png"
        draw_frame(stage, idx).save(path)
        paths.append(path)
    build_contact_sheet(paths)
    build_preview_gif(paths)


if __name__ == "__main__":
    main()
