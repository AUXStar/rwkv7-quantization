"""轻量绘图工具（PIL+numpy，无matplotlib依赖）。

支持：折线图（可log y）、热力图（可log color）。
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def line_chart(xs_list, ys_list, labels, title, xlabel, ylabel, out,
               log_y=False, log_x=False, w=1100, h=700):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = _get_font(14)
    font_t = _get_font(18)

    margin_l, margin_b, margin_t, margin_r = 90, 60, 60, 30
    pw, ph = w - margin_l - margin_r, h - margin_t - margin_b

    # data range
    all_x = [x for xs in xs_list for x in xs]
    all_y = [y for ys in ys_list for y in ys]
    if not all_x or not all_y:
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymax = ymin + 1
    if log_y:
        ymin = max(ymin, 1e-12)
        ymax = max(ymax, ymin * 10)

    def tx(x):
        if log_x:
            v = math.log10(max(x, 1)) - math.log10(max(xmin, 1))
            rng = math.log10(max(xmax, 1)) - math.log10(max(xmin, 1))
            return margin_l + (v / rng if rng > 0 else 0) * pw
        return margin_l + (x - xmin) / (xmax - xmin) * pw

    def ty(y):
        if log_y:
            v = math.log10(max(y, 1e-12)) - math.log10(ymin)
            rng = math.log10(ymax) - math.log10(ymin)
            return margin_t + ph - (v / rng if rng > 0 else 0) * ph
        return margin_t + ph - (y - ymin) / (ymax - ymin) * ph

    # grid + ticks
    for i in range(6):
        yy = ymin + (ymax - ymin) * i / 5
        gy = ty(yy)
        d.line([margin_l, gy, w - margin_r, gy], fill="lightgray", width=1)
        lab = f"{yy:.2e}" if (log_y or abs(yy) < 0.01 or abs(yy) > 1e5) else f"{yy:.4g}"
        d.text((8, gy - 8), lab, fill="black", font=font)
    for i in range(6):
        xx = xmin + (xmax - xmin) * i / 5
        gx = tx(xx)
        d.line([gx, margin_t, gx, h - margin_b], fill="lightgray", width=1)
        lab = f"{xx:.3g}"
        d.text((gx - 20, h - margin_b + 5), lab, fill="black", font=font)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for k, (xs, ys) in enumerate(zip(xs_list, ys_list)):
        pts = [(tx(x), ty(y)) for x, y in zip(xs, ys) if y is not None]
        if len(pts) >= 2:
            d.line(pts, fill=colors[k % len(colors)], width=2)
        elif pts:
            d.ellipse([pts[0][0]-3, pts[0][1]-3, pts[0][0]+3, pts[0][1]+3],
                      fill=colors[k % len(colors)])
        if labels:
            lx, ly = pts[-1] if pts else (margin_l, margin_t)
            d.text((lx + 5, ly - 12), labels[k], fill=colors[k % len(colors)], font=font)

    d.text((w // 2, 15), title, fill="black", font=font_t)
    d.text((w // 2 - 40, h - 25), xlabel, fill="black", font=font)
    # y label rotated
    img_draw = ImageDraw.Draw(img)
    txt = Image.new("RGBA", (200, 30), (255, 255, 255, 0))
    td = ImageDraw.Draw(txt)
    td.text((0, 0), ylabel, fill="black", font=font)
    txt = txt.rotate(90, expand=True)
    img.paste(txt, (15, h // 2 - 40), txt)
    img.save(out)


def heatmap(mat, row_labels, col_labels, title, out, log_scale=True,
            w=1000, h=640):
    """mat: numpy array [nrows, ncols]."""
    nrows, ncols = mat.shape
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = _get_font(13)
    font_t = _get_font(18)

    margin_l, margin_b, margin_t, margin_r = 130, 50, 60, 120
    pw, ph = w - margin_l - margin_r, h - margin_t - margin_b
    cw, ch = pw / ncols, ph / nrows

    if log_scale:
        vals = np.log10(np.maximum(mat, 1e-12))
        vmin, vmax = vals.min(), vals.max()
    else:
        vals = mat
        vmin, vmax = mat.min(), mat.max()
    vrange = vmax - vmin if vmax > vmin else 1.0

    import colorsys
    for i in range(nrows):
        for j in range(ncols):
            v = (vals[i, j] - vmin) / vrange
            v = max(0.0, min(1.0, v))
            r, g, b = colorsys.hsv_to_rgb(0.66 * (1 - v), 0.85, 0.55 + 0.45 * v)
            x0, y0 = margin_l + j * cw, margin_t + i * ch
            d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(int(r*255), int(g*255), int(b*255)))
            if ncols <= 8 and nrows <= 8:
                lab = f"{mat[i, j]:.1e}"
                d.text((x0 + 4, y0 + 4), lab, fill="white", font=font)

    for j, lab in enumerate(col_labels):
        d.text((margin_l + j * cw + cw/2 - 10, margin_t - 22), lab, fill="black", font=font)
    for i, lab in enumerate(row_labels):
        d.text((margin_l - 60, margin_t + i * ch + ch/2 - 8), lab, fill="black", font=font)

    d.text((w // 2, 15), title, fill="black", font=font_t)
    # colorbar
    cb_x = w - margin_r + 10
    for k in range(100):
        v = k / 99
        r, g, b = colorsys.hsv_to_rgb(0.66 * (1 - v), 0.85, 0.55 + 0.45 * v)
        d.rectangle([cb_x, margin_t + k * (ph / 99), cb_x + 15, margin_t + (k+1) * (ph / 99)],
                    fill=(int(r*255), int(g*255), int(b*255)))
    d.text((cb_x - 8, margin_t - 20), "max", fill="black", font=font)
    d.text((cb_x - 8, h - margin_b + 2), "min", fill="black", font=font)
    img.save(out)
