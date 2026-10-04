"""System and method schematic (the first figure of the article), drawn in vector by code.
Called by the fig1 function of ce1_figures.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Arc
import ce1_figures as F

MM = 1 / 25.4
W, H = 177.0, 106.0
INK, MUTED = F.INK, F.MUTED
NAVY, PVB, PVL = "#1f3b57", "#2f5b86", "#8fb3d6"
SUN, SUNE = "#f6c85f", "#d99a1e"
GEN, GENE, GREY = "#e8a33d", "#b8741a", "#d9d9d9"
BAT, BATE = "#5aa9e6", "#1f6fae"
LOAD, ROOF = "#f3ece2", "#c9694a"
BLUE, VERM = "#0072B2", "#D55E00"
LW = 0.6


def P(ax, pts, fc, ec=INK, lw=LW, ls="-", z=3, alpha=1):
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha, joinstyle="round"))


def R(ax, x, y, w, h, fc, ec=INK, lw=LW, r=0.6, ls="-", z=3, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z, alpha=alpha))


def arrow(ax, p0, p1, color=INK, lw=0.9, ms=7, z=4, ls="-", cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=z,
                                 linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0))


def T(ax, x, y, s, fs=6.4, ha="center", va="center", c=INK, w="normal", z=10, **k):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=c, fontweight=w, zorder=z, linespacing=1.2, **k)


# ------------------------------------------------------------------ icons
def sun(ax, x, y, r, z=3):
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([x + 1.35 * r * np.cos(a), x + 1.8 * r * np.cos(a)], [y + 1.35 * r * np.sin(a), y + 1.8 * r * np.sin(a)],
                color=SUNE, lw=0.8, solid_capstyle="round", zorder=z)
    ax.add_patch(Circle((x, y), r, fc=SUN, ec=SUNE, lw=LW, zorder=z + 0.1))


def cloud(ax, x, y, s, z=4, fc="white", ec="#8a98a8"):
    parts = [(-0.9, 0, 0.75), (0, 0.35, 1.0), (0.95, 0.05, 0.7), (0.0, -0.15, 0.8)]
    for dx, dy, r in parts:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, fc=fc, ec=ec, lw=LW, zorder=z))
    for dx, dy, r in parts:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s - 0.25, fc=fc, ec="none", zorder=z + 0.1))
    ax.add_patch(Rectangle((x - 0.9 * s, y - 0.75 * s), 1.85 * s, 0.6 * s, fc=fc, ec="none", zorder=z + 0.1))
    ax.plot([x - 0.9 * s, x + 0.95 * s], [y - 0.8 * s, y - 0.8 * s], color=ec, lw=LW, zorder=z + 0.2)


def pv_panel(ax, x, y, w, h, z=3):
    """Tilted panel with a 4 x 3 cell grid and two legs; (x, y) is the lower-left corner of the panel base."""
    sk = 0.35 * w
    P(ax, [(x, y), (x + w, y), (x + w - sk + w * 0.15, y + h), (x - sk + w * 0.15, y + h)], PVB, NAVY, z=z)
    for k in range(1, 4):
        f = k / 4
        ax.plot([x + f * w, x + f * w - sk + w * 0.15], [y, y + h], color=PVL, lw=0.35, zorder=z + 0.1)
    for k in range(1, 3):
        f = k / 3
        ax.plot([x + f * (w * 0.15 - sk), x + w + f * (w * 0.15 - sk)], [y + f * h, y + f * h], color=PVL, lw=0.35, zorder=z + 0.1)
    for lx in (x + 0.2 * w, x + 0.8 * w):
        ax.plot([lx, lx], [y - 1.6, y], color="#6b6b6b", lw=0.7, zorder=z - 0.1)
    ax.plot([x, x + w], [y, y], color=NAVY, lw=0.8, zorder=z + 0.2)


def genset(ax, x, y, w, h, on=True, z=3):
    fc = GEN if on else "#eeeeee"
    ec = GENE if on else "#9a9a9a"
    R(ax, x, y, w, h, fc, ec, r=0.5, z=z)
    for k in range(3):  # radiator louvres
        ax.plot([x + 0.18 * w, x + 0.18 * w], [y + (0.25 + 0.2 * k) * h, y + (0.25 + 0.2 * k) * h + 0.12 * h],
                color=ec, lw=0.5, zorder=z + 0.1)
        ax.plot([x + 0.12 * w, x + 0.3 * w], [y + (0.3 + 0.2 * k) * h] * 2, color=ec, lw=0.5, zorder=z + 0.1)
    ax.add_patch(Rectangle((x + 0.62 * w, y + h), 0.12 * w, 0.28 * h, fc="#7a7a7a" if on else "#bdbdbd", ec=ec, lw=0.4, zorder=z))
    for wx in (x + 0.12 * w, x + 0.88 * w):
        ax.add_patch(Circle((wx, y), 0.06 * w + 0.25, fc="#555555" if on else "#aaaaaa", ec="none", zorder=z + 0.2))
    if on:
        for k, dy in enumerate((0.5, 1.2)):
            ax.add_patch(Circle((x + 0.68 * w + 0.4 * k, y + h + 0.28 * h + dy + 0.4), 0.35 + 0.15 * k, fc="#d7d7d7", ec="none", zorder=z))


def battery(ax, x, y, w, h, z=3):
    R(ax, x, y, w, h, "#e6f1fa", BATE, r=0.8, ls=(0, (3, 1.5)), z=z, lw=0.8)
    ax.add_patch(Rectangle((x + w, y + 0.32 * h), 0.9, 0.36 * h, fc=BATE, ec="none", zorder=z))
    for k in range(3):
        R(ax, x + 0.8 + k * (w - 1.6) / 3 + 0.3, y + 0.8, (w - 1.6) / 3 - 0.6, h - 1.6, BAT, "none", r=0.3, z=z + 0.1, alpha=0.75)
    # bolt
    bx, by = x + w / 2, y + h / 2
    P(ax, [(bx + 0.4, by + 2.2), (bx - 1.2, by - 0.2), (bx, by - 0.1), (bx - 0.5, by - 2.2), (bx + 1.2, by + 0.3), (bx, by + 0.2)],
      "white", BATE, lw=0.4, z=z + 0.3)


def house(ax, x, y, w, h, z=3, pv=True):
    """A flat-roofed hotel block (the load is a large-hotel profile), not a dwelling."""
    hh = h * 1.25
    R(ax, x, y, w, hh, LOAD, INK, r=0.2, z=z)
    ax.add_patch(Rectangle((x - 0.3, y + hh), w + 0.6, 0.45, fc=ROOF, ec=INK, lw=0.35, zorder=z + 0.1))
    if pv:
        ax.add_patch(Rectangle((x + 0.15 * w, y + hh + 0.45), 0.7 * w, 0.5, fc=PVB, ec=NAVY, lw=0.35, zorder=z + 0.2))
    rows_, cols_ = 3, 3
    for i in range(rows_):
        for k in range(cols_):
            ax.add_patch(Rectangle((x + (0.12 + 0.3 * k) * w, y + (0.38 + 0.2 * i) * hh), 0.16 * w, 0.11 * hh,
                                   fc=PVL, ec=INK, lw=0.3, zorder=z + 0.2))
    ax.add_patch(Rectangle((x + 0.42 * w, y), 0.16 * w, 0.26 * hh, fc="#8c6a4f", ec=INK, lw=0.35, zorder=z + 0.2))


def pyranometer(ax, x, y, z=3):
    ax.plot([x, x], [y, y + 11], color="#555555", lw=1.0, zorder=z)
    ax.plot([x - 2.2, x + 2.2], [y, y], color="#555555", lw=1.0, zorder=z)
    R(ax, x - 3.2, y + 11, 6.4, 1.1, "#f2f2f2", INK, r=0.3, z=z + 0.1)
    ax.add_patch(Wedge((x, y + 12.1), 1.7, 0, 180, fc="#cfe2f2", ec=INK, lw=0.5, zorder=z + 0.2))
    ax.add_patch(Wedge((x, y + 12.1), 0.9, 0, 180, fc="#1f1f1f", ec="none", zorder=z + 0.3))


# ------------------------------------------------------------------ figure
def build(npz_path, day, cov_icp, cov_mcqr):
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # ===== (a) physical system =====
    T(ax, 2, H - 3, "(a)", fs=7.5, ha="left", w="bold"); T(ax, 9, H - 3, "The modelled Yulara mini-grid", fs=7, ha="left", c=MUTED)
    ybus = 64.0
    ax.plot([8, 164], [ybus, ybus], color=INK, lw=2.0, solid_capstyle="round", zorder=5)
    T(ax, 165.5, ybus, "AC bus", fs=6, ha="left", c=MUTED)
    cols = {"pv": 22, "pyr": 49, "gen": 83, "bat": 124, "load": 154}
    for i, (dx, dy) in enumerate([(-11, 77), (-2, 77), (7, 77), (-15.5, 70.5), (-6.5, 70.5), (2.5, 70.5)]):
        pv_panel(ax, cols["pv"] + dx, dy, 8.2, 5.2, z=3 + i * 0.01)
    sun(ax, cols["pv"] + 13, 92, 2.6); cloud(ax, cols["pv"] + 4, 90.5, 2.6)
    ax.add_patch(Ellipse((cols["pv"] - 3, 73), 13, 4.2, fc="#0b1a2a", ec="none", alpha=0.18, zorder=4))
    T(ax, cols["pv"], 69.3 + 0.2, "", fs=1)
    # pyranometer
    pyranometer(ax, cols["pyr"], 72)
    # diesel station: 1 large + 4 small (3 on, 2 off illustrates commitment)
    gx = cols["gen"] - 20
    R(ax, gx - 1.5, 69.5, 43, 20.0, "#f7f7f7", "#9a9a9a", r=1.2, z=2, lw=0.5)
    genset(ax, gx + 0.5, 72.5, 12, 8, on=True)
    for k in range(4):
        genset(ax, gx + 15 + k * 6.3, 72.5, 5.3, 5.6, on=k < 2)
    # battery (hypothetical)
    battery(ax, cols["bat"] - 8, 72, 15, 9)
    # load: hotel blocks (large-hotel profile)
    house(ax, cols["load"] - 11, 71, 9, 7); house(ax, cols["load"] - 0.5, 71, 11, 9); house(ax, cols["load"] + 12, 71, 7, 6, pv=False)
    # connections with power-flow arrows
    for key, x, y0, direction in [("pv", 22, 68, "down"), ("gen", 83, 69.5, "down"), ("bat", 124, 71.5, "both"), ("load", 154, 70.5, "up")]:
        ax.plot([x, x], [ybus, y0], color=INK, lw=0.8, zorder=4)
        if direction == "down":
            arrow(ax, (x + 2.2, y0 - 0.3), (x + 2.2, ybus + 1.2), color=MUTED, lw=0.7, ms=5)
        elif direction == "up":
            arrow(ax, (x + 2.2, ybus + 1.2), (x + 2.2, y0 - 0.3), color=MUTED, lw=0.7, ms=5)
        else:
            ax.add_patch(FancyArrowPatch((x + 2.2, ybus + 1.2), (x + 2.2, y0 - 0.3), arrowstyle="<|-|>", mutation_scale=5,
                                         lw=0.7, color=MUTED, zorder=4))
    ax.plot([49, 49], [72, 67], color=MUTED, lw=0.6, ls=(0, (1.5, 1.2)), zorder=4)
    # captions under icons (below the bus)
    lab = [(22, "PV arrays", "1.8 MWp, five arrays\ncapped at 1,407 kW"),
           (49, "Pyranometer", "5-min GHI\n(DKASC record)"),
           (83, "Diesel station", "2.5 MW + 4 x 1.0 MW\nminimum loading 40%"),
           (124, "Battery (hypothetical)", "4 MW / 2 MWh\nreserve credit"),
           (154, "Load", "representative large-\nhotel profile, shapes A, B")]
    for x, a, b in lab:
        T(ax, x, 59.6, a, fs=6.6, w="bold")
        T(ax, x, 54.6, b, fs=6.1, c=MUTED)
    T(ax, 83, 86.8, "sets online (orange) and offline (grey)", fs=6.0, c=MUTED)
    ax.plot([2, 175], [49.5, 49.5], color="#e2e2e2", lw=0.6, zorder=1)

    # ===== (b) chain of cards with mini-illustrations =====
    T(ax, 2, 46.5, "(b)", fs=7.5, ha="left", w="bold"); T(ax, 9, 46.5, "From irradiance record to reserve and fuel", fs=7, ha="left", c=MUTED)
    cw, ch, cy = 25.5, 30.0, 10.5
    xs = [2 + i * (cw + 3.6) for i in range(6)]
    titles = ["Irradiance record", "Interval forecast", "Regime calibration", "Reserve rule", "Diesel commitment", "Outcomes"]
    # ACI is not used for reserve, so card 3 does not list it
    subs = ["measured GHI, 5 min", "90% interval, lower bound", "ICP, Mondrian, CQR", "PV counted at level $\\theta$",
            "least-rating unit set", "fuel and unserved energy"]
    fills = ["#ffffff", "#ffffff", "#eef5fb", "#fdf0e6", "#f4f4f4", "#ffffff"]
    for i, x in enumerate(xs):
        R(ax, x, cy, cw, ch, fills[i], "#4a4a4a", r=1.2, z=2, lw=0.6)
        T(ax, x + 4.6, cy + ch - 3.0, titles[i], fs=6.2, w="bold", ha="left")
        T(ax, x + cw / 2, cy + 2.6, subs[i], fs=6.0, c=MUTED)
        ax.add_patch(Circle((x + 2.4, cy + ch - 3.0), 1.5, fc=INK, ec="none", zorder=6))
        T(ax, x + 2.4, cy + ch - 3.05, str(i + 1), fs=6.0, c="white", w="bold")
        if i:
            arrow(ax, (xs[i - 1] + cw + 0.4, cy + ch / 2), (x - 0.4, cy + ch / 2), lw=0.9, ms=6)
    d = np.load(npz_path)
    t = pd.to_datetime(d["ts_ns"])
    k = np.asarray(t.normalize() == pd.Timestamp(day))
    hrs = (t[k].hour + t[k].minute / 60).values
    y = d["y"][k]; p = d["p"][k]; wd = d["wid__icp|static|0.9"][k]; reg = d["reg_it"][k]

    def inset(i, x0f=0.25, x1f=0.94, y0=8.6, y1=23.2):
        a = fig.add_axes([(xs[i] + x0f * cw) / W, (cy + y0) / H, (x1f - x0f) * cw / W, (y1 - y0) / H])
        a.set_facecolor("none")
        for s in a.spines.values():
            s.set_linewidth(0.4)
        a.tick_params(length=1.2, width=0.4, labelsize=6.0, pad=1)
        return a

    a1 = inset(0)
    a1.plot(hrs, y, color=SUNE, lw=0.7)
    a1.fill_between(hrs, 0, y, color=SUN, alpha=0.35, lw=0)
    a1.set_xticks([9, 12, 15]); a1.set_yticks([0, 600, 1200]); a1.set_ylim(0, 1300)
    a1.text(0.98, 0.97, "W/m$^2$", fontsize=6.0, transform=a1.transAxes, ha="right", va="top", color=MUTED)
    a2 = inset(1)
    lo, hi = np.maximum(p - wd / 2, 0), p + wd / 2
    a2.fill_between(hrs, lo, hi, color=BAT, alpha=0.3, lw=0)
    a2.plot(hrs, p, color=BLUE, lw=0.6); a2.plot(hrs, y, color=INK, lw=0.4, alpha=0.6)
    a2.plot(hrs, lo, color=VERM, lw=0.6)
    a2.set_xticks([9, 12, 15]); a2.set_yticks([0, 600, 1200]); a2.set_ylim(0, 1300)
    a2.text(0.98, 0.97, "W/m$^2$", fontsize=6.0, transform=a2.transAxes, ha="right", va="top", color=MUTED)
    # 3: regime icons with transitional coverage bars
    x3 = xs[2]
    for j, (lab_, kind) in enumerate([("clear", "sun"), ("transitional", "mix"), ("cloudy", "cloud")]):
        cx = x3 + 5.5 + j * 7.2
        if kind in ("sun", "mix"):
            sun(ax, cx + (0.8 if kind == "mix" else 0), 34.3, 1.3, z=6)
        if kind in ("mix", "cloud"):
            cloud(ax, cx - (0.5 if kind == "mix" else 0), 33.0, 1.3, z=7, fc="#ffffff" if kind == "mix" else "#dfe5ec")
    bx0, bw_ = x3 + 3.5, 18.5
    for j, (lab_, v, col) in enumerate([("marginal ICP", cov_icp, VERM), ("Mondrian-CQR", cov_mcqr, BLUE)]):
        yy = 22.6 - j * 5.6
        ax.add_patch(Rectangle((bx0, yy), bw_, 2.2, fc="#e9e9e9", ec="none", zorder=5))
        ax.add_patch(Rectangle((bx0, yy), bw_ * v, 2.2, fc=col, ec="none", zorder=5.1))
        T(ax, bx0, yy + 3.4, lab_, fs=6.0, ha="left", c=INK)
        T(ax, bx0 + bw_ * v - 0.5, yy + 1.1, "%.3f" % v, fs=6.0, ha="right", c="white", w="bold")
    T(ax, x3 + cw / 2, 30.2, "transitional coverage", fs=6.0, c=MUTED)
    ax.plot([bx0 + bw_ * 0.9] * 2, [16.4, 25.2], color=INK, lw=0.5, ls=(0, (1.2, 0.8)), zorder=6)
    T(ax, bx0 + bw_ * 0.9 + 0.8, 15.2, "90% target", fs=6.0, c=INK, ha="right")
    T(ax, x3 + cw / 2, 27.9 - 0.5, "", fs=1)
    # 4: reserve rule stacked bars: handbook vs forecast
    x4 = xs[3]
    base = 18.5
    for j, (lab_, pvc) in enumerate([("handbook", 0.0), ("forecast", 0.55)]):
        bx = x4 + 3.5 + j * 10.5
        L = 9.5
        ax.add_patch(Rectangle((bx, base), 8, L * (1 - pvc) + 0.0, fc=GEN, ec=GENE, lw=0.5, zorder=5))
        if pvc:
            ax.add_patch(Rectangle((bx, base + L * (1 - pvc)), 8, L * pvc, fc=SUN, ec=SUNE, lw=0.5, zorder=5))
            T(ax, bx + 4, base + L * (1 - pvc) + L * pvc / 2, "PV", fs=6.0)
        T(ax, bx + 4, base + L * (1 - pvc) / 2, "diesel\nonline", fs=6.0)
        T(ax, bx + 4, base - 1.3 + 0.2, lab_, fs=6.0, c=MUTED, va="top")
    ax.plot([x4 + 2.5, x4 + 23.0], [base + 9.5, base + 9.5], color=INK, lw=0.5, ls=(0, (1.2, 0.8)), zorder=6)
    T(ax, x4 + 13, base + 10.7, "load", fs=6.0)
    # 5: commitment: small genset row, on/off
    x5 = xs[4]
    genset(ax, x5 + 2.2, 20.5, 8.0, 5.5, on=True, z=5)
    for j in range(4):
        genset(ax, x5 + 11.5 + j * 3.3, 20.5, 2.8, 3.6, on=j < 1, z=5)
    T(ax, x5 + cw / 2, 16.8, "fewer sets online", fs=6.0, c=INK)
    # 6: outcomes: fuel drop and UE gauge
    x6 = xs[5]
    dx, dy = x6 + 6.5, 20.5
    P(ax, [(dx, dy + 5.4), (dx - 2.6, dy + 1.2), (dx + 2.6, dy + 1.2)], "#6d4c2f", "#3e2a18", lw=0.4, z=5)
    ax.add_patch(Circle((dx, dy + 0.9), 2.62, fc="#6d4c2f", ec="#3e2a18", lw=0.4, zorder=5))
    T(ax, dx, dy - 3.8, "fuel", fs=6.0)
    gx_, gy_ = x6 + 18, 20.0
    ax.add_patch(Wedge((gx_, gy_), 5.0, 0, 180, width=1.4, fc="#e9e9e9", ec="none", zorder=5))
    ax.add_patch(Wedge((gx_, gy_), 5.0, 120, 180, width=1.4, fc="#009E73", ec="none", zorder=5.1))
    ang = np.deg2rad(138)
    ax.plot([gx_, gx_ + 4.0 * np.cos(ang)], [gy_, gy_ + 4.0 * np.sin(ang)], color=INK, lw=0.8, zorder=6)
    ax.add_patch(Circle((gx_, gy_), 0.5, fc=INK, ec="none", zorder=6.1))
    T(ax, gx_, gy_ - 3.8, "UE", fs=6.0)
    # feedback protocols
    xr, xo, xc = xs[3] + cw / 2, xs[5] + cw / 2, xs[2] + cw / 2
    yb = cy - 4.5
    ax.plot([xo, xo, xr + 3], [cy, yb, yb], color=BLUE, lw=1.0, zorder=4)
    arrow(ax, (xr + 3, yb), (xr + 3, cy), color=BLUE, lw=1.0, ms=6)
    T(ax, xr + 5, yb - 2.3, "A-D only: level updated daily from observed unserved energy", fs=6.0, c=BLUE, ha="left")
    ax.plot([xc, xc, xr - 3], [cy, yb, yb], color=VERM, lw=1.0, ls=(0, (3, 1.5)), zorder=4)
    arrow(ax, (xr - 3, yb), (xr - 3, cy), color=VERM, lw=1.0, ms=6)
    T(ax, xr - 5, yb - 2.3, "A-R and the static margin rules: level fixed before the test year", fs=6.0, c=VERM, ha="right")
    return fig


