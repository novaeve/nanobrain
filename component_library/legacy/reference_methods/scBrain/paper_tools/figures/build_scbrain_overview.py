#!/usr/bin/env python3
"""Compose the scBrain overview figure from an AI-generated biological base image.

This script keeps all labels, arrows, and mathematical structure deterministic,
while using one GPT-generated raster asset to provide the biological scene and
overall visual polish.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BASE_DEFAULT = ROOT / "source_scbrain_overview_base_v1.png"
OUT_PNG = ROOT / "fig_scbrain_overview.png"
OUT_PDF = ROOT / "fig_scbrain_overview.pdf"

W, H = 3600, 1900

BG = "#FBFAF5"
TEXT = "#1D2630"
MUTED = "#5C6670"
GRID = "#D9D7CF"
TEAL = "#2A7F86"
TEAL_LIGHT = "#D8EFED"
SLATE = "#335C74"
SLATE_LIGHT = "#DCE8F1"
ORANGE = "#D56A2C"
ORANGE_LIGHT = "#F7E1D4"
GOLD = "#C69432"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size=size)


def rounded_panel(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=2, radius=32):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadowed_panel(canvas: Image.Image, box, fill, outline=None, width=2, radius=36, shadow=(10, 12), blur=18):
    shadow_layer = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    sx, sy = shadow
    shadow_box = (box[0] + sx, box[1] + sy, box[2] + sx, box[3] + sy)
    shadow_draw.rounded_rectangle(shadow_box, radius=radius, fill=(30, 38, 48, 52))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(shadow_layer)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start, end, color, width=8, head=22):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    tip = (x2, y2)
    left = (x2 - ux * head - px * head * 0.55, y2 - uy * head - py * head * 0.55)
    right = (x2 - ux * head + px * head * 0.55, y2 - uy * head + py * head * 0.55)
    draw.polygon([tip, left, right], fill=color)


def orth_arrow(draw: ImageDraw.ImageDraw, points, color, width=8, head=22):
    pts = [(int(x), int(y)) for x, y in points]
    if len(pts) < 2:
        return
    draw.line(pts, fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color, width=width, head=head)


def curved_residual(draw: ImageDraw.ImageDraw, p0, p1, color, width=8):
    x0, y0 = p0
    x1, y1 = p1
    midx = (x0 + x1) / 2
    pts = [
        (x0, y0),
        (midx, y0 - 110),
        (midx, y1 - 110),
        (x1, y1),
    ]
    draw.line(pts, fill=color, width=width, joint="curve")
    arrow(draw, pts[-2], pts[-1], color, width=width, head=20)


def draw_matrix_icon(draw: ImageDraw.ImageDraw, box, fill=SLATE):
    x0, y0, x1, y1 = box
    cols, rows = 5, 6
    pad = 12
    cell_w = (x1 - x0 - pad * 2) / cols
    cell_h = (y1 - y0 - pad * 2) / rows
    for r in range(rows):
        for c in range(cols):
            alpha = 70 + ((r + c) % 4) * 35
            color = (*ImageColor.getrgb(fill), alpha)  # type: ignore[name-defined]
            draw.rounded_rectangle(
                (
                    x0 + pad + c * cell_w + 2,
                    y0 + pad + r * cell_h + 2,
                    x0 + pad + (c + 1) * cell_w - 4,
                    y0 + pad + (r + 1) * cell_h - 4,
                ),
                radius=5,
                fill=color,
            )


def draw_hierarchy_tree(draw: ImageDraw.ImageDraw, origin_x: int, origin_y: int, scale: float = 1.0):
    s = lambda v: int(round(v * scale))
    levels = [
        [(origin_x, origin_y + s(110)), (origin_x, origin_y + s(280)), (origin_x, origin_y + s(450))],
        [
            (origin_x + s(140), origin_y + s(55)),
            (origin_x + s(140), origin_y + s(145)),
            (origin_x + s(140), origin_y + s(245)),
            (origin_x + s(140), origin_y + s(335)),
            (origin_x + s(140), origin_y + s(420)),
            (origin_x + s(140), origin_y + s(505)),
        ],
        [(origin_x + s(290), origin_y + s(30) + i * s(43)) for i in range(12)],
    ]
    for idx, (x, y) in enumerate(levels[0]):
        r = s(20)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=TEAL if idx < 2 else SLATE)
    for idx, (x, y) in enumerate(levels[1]):
        r = s(14)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=SLATE if idx % 2 == 0 else ORANGE)
    for idx, (x, y) in enumerate(levels[2]):
        r = s(9)
        fill = ORANGE if idx % 3 == 0 else SLATE
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    mapping = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    for i, js in mapping.items():
        x0, y0 = levels[0][i]
        for j in js:
            x1, y1 = levels[1][j]
            draw.line((x0 + s(20), y0, x1 - s(14), y1), fill=MUTED, width=max(s(4), 2))
    for j, (x0, y0) in enumerate(levels[1]):
        for k in [2 * j, 2 * j + 1]:
            x1, y1 = levels[2][k]
            draw.line((x0 + s(14), y0, x1 - s(9), y1), fill=GRID, width=max(s(3), 2))


def add_text(draw: ImageDraw.ImageDraw, xy, text, size, color=TEXT, bold=False, anchor="la"):
    draw.text(xy, text, fill=color, font=font(size, bold=bold), anchor=anchor)


def crop_resize_cover(image: Image.Image, size, crop_box=None) -> Image.Image:
    if crop_box is not None:
        image = image.crop(crop_box)
    tw, th = size
    sw, sh = image.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    image = image.resize((nw, nh), resample)
    left = max((nw - tw) // 2, 0)
    top = max((nh - th) // 2, 0)
    return image.crop((left, top, left + tw, top + th))


def generate_fallback_biology_panel(size):
    w, h = size
    panel = Image.new("RGBA", size, "#F7F4EE")
    draw = ImageDraw.Draw(panel)

    layer_colors = ["#E8EEF0", "#E3EFE8", "#F1E7DC", "#E9EDF6", "#F4E7E0"]
    for i, color in enumerate(layer_colors):
        top = int(h * (0.08 + i * 0.17))
        bottom = top + int(h * 0.17)
        pts = [
            (0, top + 28 * ((i % 2) * 2 - 1)),
            (w * 0.22, top - 24),
            (w * 0.45, top + 12),
            (w * 0.72, top - 18),
            (w, top + 22),
            (w, bottom + 20),
            (w * 0.75, bottom - 14),
            (w * 0.42, bottom + 18),
            (w * 0.16, bottom - 10),
            (0, bottom + 24),
        ]
        draw.polygon(pts, fill=color)

    # Stylized vessels / tissue contours
    for x in [int(w * 0.16), int(w * 0.31), int(w * 0.63), int(w * 0.81)]:
        draw.line((x, int(h * 0.12), x - 20, int(h * 0.86)), fill="#D6C7BD", width=5)

    # Cells
    cells = [
        (0.14, 0.24, 40, "#437F8E"),
        (0.27, 0.31, 48, "#CF6D3B"),
        (0.39, 0.20, 34, "#67889B"),
        (0.58, 0.28, 52, "#C5923C"),
        (0.68, 0.41, 44, "#4C7C6F"),
        (0.21, 0.50, 50, "#7F94A8"),
        (0.37, 0.56, 38, "#D57E4F"),
        (0.55, 0.61, 46, "#40758A"),
        (0.73, 0.58, 55, "#758B5F"),
        (0.84, 0.73, 42, "#C46A54"),
        (0.62, 0.80, 36, "#8B7FA8"),
        (0.29, 0.77, 50, "#476F84"),
    ]
    for cxr, cyr, r, fill in cells:
        cx, cy = int(w * cxr), int(h * cyr)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=(255, 255, 255, 120), width=3)
        draw.ellipse((cx - r * 0.34, cy - r * 0.34, cx + r * 0.34, cy + r * 0.34), fill=(255, 255, 255, 110))
        # neuronal processes
        if r > 42:
            for dx, dy in [(-1.2, -0.5), (1.1, -0.7), (-0.8, 1.0)]:
                draw.line((cx, cy, cx + int(dx * r * 1.8), cy + int(dy * r * 1.8)), fill=fill, width=4)

    # Sparse MERFISH-like molecular dots
    dot_colors = ["#4EB4C8", "#F08B52", "#F2C14E", "#6F9C88"]
    for i in range(220):
        x = int((0.08 + (i * 37 % 84) / 100) * w)
        y = int((0.08 + (i * 53 % 80) / 100) * h)
        r = 2 + (i % 3)
        c = dot_colors[i % len(dot_colors)]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c)

    # subtle vignette
    vignette = Image.new("RGBA", size, (255, 255, 255, 0))
    vd = ImageDraw.Draw(vignette)
    vd.rounded_rectangle((10, 10, w - 10, h - 10), radius=36, outline=(255, 255, 255, 100), width=6)
    panel.alpha_composite(vignette)
    return panel.convert("RGB")


def compose(base_path: Path = BASE_DEFAULT):
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    base_full = Image.open(base_path).convert("RGB") if base_path.exists() else None

    left_box = (70, 250, 1125, 1675)
    mid_box = (1185, 250, 2095, 1675)
    model_panel = (2155, 250, 3530, 1675)
    cx = lambda box: (box[0] + box[2]) // 2
    cy = lambda box: (box[1] + box[3]) // 2

    add_text(draw, (92, 92), "scBrain Overview", 54, color=TEXT, bold=True)
    add_text(draw, (94, 138), "From panel-limited cortical measurement to residual hierarchical decoding", 27, color=MUTED)

    # Left biological panel
    shadowed_panel(canvas, left_box, WHITE, outline="#E4E1D8", width=3, radius=42)
    target_w = left_box[2] - left_box[0] - 40
    target_h = left_box[3] - left_box[1] - 40
    if base_full is not None:
        left_crop = (0, 0, int(base_full.width * 0.34), base_full.height)
        base = crop_resize_cover(base_full, (target_w, target_h), crop_box=left_crop)
    else:
        base = generate_fallback_biology_panel((target_w, target_h))
    canvas.paste(base, (left_box[0] + 20, left_box[1] + 20))

    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    left_callout = (left_box[0] + 35, left_box[3] - 330, left_box[2] - 35, left_box[3] - 30)
    od.rounded_rectangle(left_callout, radius=34, fill=(251, 250, 245, 222), outline=(220, 217, 207, 255), width=2)
    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)

    add_text(draw, (left_callout[0] + 35, left_callout[1] + 58), "Biological Setting", 38, color=SLATE, bold=True)
    add_text(draw, (left_callout[0] + 35, left_callout[1] + 128), "Targeted spatial assays preserve tissue context,", 25, color=TEXT)
    add_text(draw, (left_callout[0] + 35, left_callout[1] + 170), "but coarse programs are easier to resolve than", 25, color=TEXT)
    add_text(draw, (left_callout[0] + 35, left_callout[1] + 212), "the weakest within-lineage supertype markers.", 25, color=TEXT)
    pill = (left_callout[0] + 35, left_callout[3] - 82, left_callout[0] + 390, left_callout[3] - 18)
    rounded_panel(draw, pill, fill=TEAL_LIGHT, outline=None, radius=18)
    add_text(draw, (cx(pill), cy(pill)), "Panel-limited spatial tissue", 22, color=TEAL, bold=True, anchor="mm")

    # Middle hierarchy panel
    add_text(draw, (mid_box[0], 168), "Nested Brain Taxonomy", 34, color=SLATE, bold=True)
    add_text(draw, (mid_box[0], 216), "Coarse identity constrains the plausible fine labels.", 22, color=MUTED)
    shadowed_panel(canvas, mid_box, WHITE, outline="#E4E1D8", width=3, radius=38)
    if base_full is not None:
        mid_bg = crop_resize_cover(
            base_full,
            (mid_box[2] - mid_box[0] - 40, mid_box[3] - mid_box[1] - 40),
            crop_box=(int(base_full.width * 0.30), 0, int(base_full.width * 0.66), base_full.height),
        )
        mid_bg = Image.blend(mid_bg, Image.new("RGB", mid_bg.size, WHITE), 0.87)
        canvas.paste(mid_bg, (mid_box[0] + 20, mid_box[1] + 20))
    draw = ImageDraw.Draw(canvas)
    draw_hierarchy_tree(draw, mid_box[0] + 110, mid_box[1] + 255, scale=1.3)
    mid_note = (mid_box[0] + 55, mid_box[3] - 235, mid_box[2] - 55, mid_box[3] - 42)
    rounded_panel(draw, mid_note, fill=(251, 250, 245, 232), outline="#E4E1D8", width=2, radius=30)
    add_text(draw, (cx(mid_note), mid_note[1] + 56), "Fine labels are interpreted only inside", 24, color=TEXT, anchor="mm")
    add_text(draw, (cx(mid_note), mid_note[1] + 94), "the correct parent lineage.", 24, color=TEXT, anchor="mm")
    legend_boxes = [
        (mid_box[0] + 70, mid_box[3] - 108, mid_box[0] + 235, mid_box[3] - 52),
        (mid_box[0] + 255, mid_box[3] - 108, mid_box[0] + 445, mid_box[3] - 52),
        (mid_box[0] + 465, mid_box[3] - 108, mid_box[0] + 640, mid_box[3] - 52),
    ]
    legend_specs = [("Class", SLATE_LIGHT, SLATE), ("Subclass", TEAL_LIGHT, TEAL), ("Supertype", ORANGE_LIGHT, ORANGE)]
    for box, (label, fill, color) in zip(legend_boxes, legend_specs):
        rounded_panel(draw, box, fill=fill, radius=16)
        add_text(draw, (cx(box), cy(box)), label, 22, color=color, bold=True, anchor="mm")

    # Right model panel
    add_text(draw, (model_panel[0], 168), "Residual Hierarchical Decoder", 34, color=ORANGE, bold=True)
    add_text(draw, (model_panel[0], 214), "Shallow states solve easier coarse decisions.", 22, color=MUTED)
    add_text(draw, (model_panel[0], 244), "Deeper states refine harder within-lineage boundaries.", 22, color=MUTED)

    shadowed_panel(canvas, model_panel, fill=WHITE, outline="#E4E1D8", width=3, radius=38)
    if base_full is not None:
        right_bg = crop_resize_cover(
            base_full,
            (model_panel[2] - model_panel[0] - 40, model_panel[3] - model_panel[1] - 40),
            crop_box=(int(base_full.width * 0.63), 0, base_full.width, base_full.height),
        )
        right_bg = Image.blend(right_bg, Image.new("RGB", right_bg.size, WHITE), 0.92)
        canvas.paste(right_bg, (model_panel[0] + 20, model_panel[1] + 20))
    draw = ImageDraw.Draw(canvas)
    rx0, ry0, rx1, ry1 = model_panel

    # Callouts
    rounded_panel(draw, (rx1 - 535, ry0 + 38, rx1 - 28, ry0 + 92), fill="#FFF4EC", radius=18)
    add_text(draw, (rx1 - 282, ry0 + 65), "Easy coarse decisions use shallow features", 19, color=ORANGE, bold=True, anchor="mm")
    rounded_panel(draw, (rx1 - 535, ry0 + 98, rx1 - 28, ry0 + 152), fill="#EEF6F7", radius=18)
    add_text(draw, (rx1 - 282, ry0 + 125), "Hard fine decisions rely on deeper refinement", 19, color=TEAL, bold=True, anchor="mm")

    # Input block
    input_box = (rx0 + 65, ry0 + 285, rx0 + 285, ry0 + 475)
    rounded_panel(draw, input_box, fill=SLATE_LIGHT, radius=26)
    add_text(draw, (cx(input_box), input_box[1] - 34), "Input", 26, color=SLATE, bold=True, anchor="mm")
    for rr in range(6):
        for cc in range(5):
            x0 = input_box[0] + 25 + cc * 36
            y0 = input_box[1] + 26 + rr * 24
            alpha = 130 + ((rr + cc) % 3) * 28
            color = (51, 92, 116, alpha)
            draw.rounded_rectangle((x0, y0, x0 + 30, y0 + 16), radius=4, fill=color)
    rounded_panel(draw, (input_box[0] + 18, input_box[3] - 32, input_box[2] - 18, input_box[3] - 2), fill=(255, 255, 255, 190), radius=12)
    add_text(draw, (cx(input_box), input_box[3] - 16), "Cell × gene", 18, color=TEXT, anchor="mm")
    coord_box = (rx0 + 82, ry0 + 500, rx0 + 268, ry0 + 552)
    rounded_panel(draw, coord_box, fill=TEAL_LIGHT, radius=18)
    add_text(draw, (cx(coord_box), cy(coord_box)), "Optional normalized x-y", 17, color=TEAL, anchor="mm")

    # Aligned encoder row
    enc_boxes = [
        (rx0 + 360, ry0 + 260, rx0 + 585, ry0 + 430),
        (rx0 + 625, ry0 + 260, rx0 + 850, ry0 + 430),
        (rx0 + 890, ry0 + 260, rx0 + 1115, ry0 + 430),
    ]
    enc_labels = [("Shallow encoder", "h1", SLATE), ("Mid-depth encoder", "h2", TEAL), ("Deep latent", "z", ORANGE)]
    for box, (title, short, color) in zip(enc_boxes, enc_labels):
        fill = ORANGE_LIGHT if color == ORANGE else SLATE_LIGHT if color == SLATE else TEAL_LIGHT
        rounded_panel(draw, box, fill=fill, radius=26)
        add_text(draw, (cx(box), box[1] + 40), title, 24, color=color, bold=True, anchor="mm")
        add_text(draw, (cx(box), box[1] + 103), short, 52, color=color, bold=True, anchor="mm")
    arrow(draw, (input_box[2], cy(input_box)), (enc_boxes[0][0], cy(enc_boxes[0])), SLATE, width=8)
    arrow(draw, (enc_boxes[0][2], cy(enc_boxes[0])), (enc_boxes[1][0], cy(enc_boxes[1])), TEAL, width=8)
    arrow(draw, (enc_boxes[1][2], cy(enc_boxes[1])), (enc_boxes[2][0], cy(enc_boxes[2])), ORANGE, width=8)

    # Aligned decoder row
    dec_boxes = [
        (enc_boxes[0][0], ry0 + 835, enc_boxes[0][2], ry0 + 965),
        (enc_boxes[1][0], ry0 + 835, enc_boxes[1][2], ry0 + 965),
        (enc_boxes[2][0], ry0 + 835, enc_boxes[2][2], ry0 + 965),
    ]
    head_boxes = [
        (cx(dec_boxes[0]) - 100, ry0 + 1115, cx(dec_boxes[0]) + 100, ry0 + 1245),
        (cx(dec_boxes[1]) - 100, ry0 + 1115, cx(dec_boxes[1]) + 100, ry0 + 1245),
        (cx(dec_boxes[2]) - 100, ry0 + 1115, cx(dec_boxes[2]) + 100, ry0 + 1245),
    ]
    names = [("Coarse decoder", "Class", SLATE), ("Refinement decoder", "Subclass", TEAL), ("Fine decoder", "Supertype", ORANGE)]
    for db, hb, (dname, hname, color) in zip(dec_boxes, head_boxes, names):
        fill = SLATE_LIGHT if color == SLATE else TEAL_LIGHT if color == TEAL else ORANGE_LIGHT
        rounded_panel(draw, db, fill=fill, radius=24)
        rounded_panel(draw, hb, fill=WHITE, outline=color, width=3, radius=24)
        add_text(draw, (cx(db), db[1] + 40), dname, 22, color=color, bold=True, anchor="mm")
        add_text(draw, (cx(hb), cy(hb)), hname, 24, color=color, bold=True, anchor="mm")

    # Orthogonal encoder-to-decoder routes
    for eb, db, color in zip(enc_boxes, dec_boxes, [SLATE, TEAL, ORANGE]):
        arrow(draw, (cx(eb), eb[3]), (cx(db), db[1]), color, width=8)

    # Residual flow across the decoder row, kept horizontal and clean.
    residual_y = dec_boxes[0][1] - 42
    orth_arrow(draw, [(dec_boxes[0][2] - 8, residual_y), (dec_boxes[1][0] + 8, residual_y)], ORANGE, width=7, head=18)
    orth_arrow(draw, [(dec_boxes[1][2] - 8, residual_y), (dec_boxes[2][0] + 8, residual_y)], ORANGE, width=7, head=18)
    orth_arrow(draw, [(cx(dec_boxes[0]), residual_y), (cx(dec_boxes[0]), dec_boxes[0][1] - 2)], ORANGE, width=6, head=16)
    orth_arrow(draw, [(cx(dec_boxes[1]), residual_y), (cx(dec_boxes[1]), dec_boxes[1][1] - 2)], ORANGE, width=6, head=16)
    orth_arrow(draw, [(cx(dec_boxes[2]), residual_y), (cx(dec_boxes[2]), dec_boxes[2][1] - 2)], ORANGE, width=6, head=16)
    add_text(draw, ((dec_boxes[0][0] + dec_boxes[2][2]) / 2, residual_y - 26), "Residual feature flow", 22, color=ORANGE, bold=True, anchor="mm")

    # Decoder outputs drop vertically for a cleaner readout.
    for db, hb, color in zip(dec_boxes, head_boxes, [SLATE, TEAL, ORANGE]):
        arrow(draw, (cx(db), db[3]), (cx(hb), hb[1]), color, width=8)

    # Reconstruction head
    recon_box = (rx0 + 70, ry1 - 210, rx0 + 660, ry1 - 42)
    rounded_panel(draw, recon_box, fill=WHITE, outline=GOLD, width=3, radius=26)
    add_text(draw, (cx(recon_box), recon_box[1] + 44), "Auxiliary reconstruction head", 23, color=GOLD, bold=True, anchor="mm")
    add_text(draw, (cx(recon_box), recon_box[1] + 88), "Preserves panel-wide molecular structure", 20, color=MUTED, anchor="mm")
    add_text(draw, (cx(recon_box), recon_box[1] + 128), "and regularizes the shared embedding.", 20, color=MUTED, anchor="mm")
    orth_arrow(
        draw,
        [
            (cx(enc_boxes[2]), enc_boxes[2][3]),
            (cx(enc_boxes[2]), recon_box[1] - 58),
            (recon_box[2] - 16, recon_box[1] - 58),
            (recon_box[2] - 16, recon_box[1] + 54),
        ],
        GOLD,
        width=7,
        head=18,
    )

    final = canvas.convert("RGB")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUT_PNG, format="PNG")

    plt.figure(figsize=(14.5, 7.65))
    plt.imshow(final)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    from PIL import ImageColor  # delayed import for clean dependency surface

    compose()
