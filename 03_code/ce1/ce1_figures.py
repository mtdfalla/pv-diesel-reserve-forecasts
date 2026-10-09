"""Draws every figure of the article and the graphical abstract from the stage outputs under
04_results/ce1/, asserts that the plotted values equal the reported values, and writes a
manifest (source and output md5, sizes, alt text). Output: 04_results/ce1/figs/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_figures.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce1_paths as PATHS  # noqa: E402  (result paths)
ROOT = Path.cwd()
R = PATHS.results(ROOT)
OUT = ROOT / "04_results" / "ce1" / "figs"
MM = 1 / 25.4
W1, W2 = 85.0, 177.0  # single and double column widths, mm
DPI_LINE = 1000
TARGET = 2e-5  # 0.002% of annual load energy
SOURCES, CHECKS, ALT, SIZES = set(), [], {}, {}

# ------------------------------------------------------------------ style (one module for all)
INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#d9d9d9"
M = {  # method: (label, colour, marker, linestyle)
    "icp":          ("Marginal ICP",        "#D55E00", "o", "-"),
    "tod":          ("Time-of-day rule",    "#CC79A7", "v", "--"),
    "tod_q":        ("Time-of-day rule",    "#CC79A7", "v", "--"),
    "cqr":          ("CQR",                 "#56B4E9", "s", "-"),
    "mondrian":     ("Mondrian ICP",        "#E69F00", "^", "-"),
    "mondrian_cqr": ("Mondrian-CQR",        "#0072B2", "D", "-"),
    "raw_q":        ("Raw quantile",        "#009E73", "P", ":"),
    "nicp_var":     ("Var.-normalized ICP", "#777777", "X", "-."),
    "frac_pv":      ("Fixed fraction of PV", "#000000", "*", (0, (1, 1))),
}


def style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    fam = [n for n in ("TeX Gyre Heros", "Liberation Sans", "Arial", "DejaVu Sans") if n in names]
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fam, "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "mathtext.fontset": "custom",
        "mathtext.rm": fam[0], "mathtext.it": fam[0] + ":italic", "mathtext.bf": fam[0] + ":bold",
    })
    return fam[0]


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, fontweight="bold", color=INK)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def src(rel):
    SOURCES.add(rel)
    return R / rel


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))


MIN_PT = 6.0  # smallest text at print size: checked for every figure except the pixel-sized abstract
GA_MIN_PT = 8.0  # graphical abstract text when shown 13 cm wide


def save(fig, name, width_mm, alt, px=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if not px:
        fig.canvas.draw()
        sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
        check("%s smallest text %.1f pt (min %.1f)" % (name, min(sizes), MIN_PT), min(sizes) >= MIN_PT - 1e-9)
    # every visible text lies inside the canvas (a clipped label fails the stage)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    out = []
    undrawn = set()  # tick labels outside the view limits are never drawn (matplotlib keeps them as Text objects)
    for a_ in fig.axes:
        for ax_ in (a_.xaxis, a_.yaxis):
            if not (a_.axison and a_.get_visible() and ax_.get_visible()):
                for t_ in ax_.get_major_ticks() + ax_.get_minor_ticks():
                    undrawn.update((id(t_.label1), id(t_.label2)))
                continue
            drawn = set(id(t_.label1) for t_ in ax_._update_ticks()) | set(id(t_.label2) for t_ in ax_._update_ticks())
            for t_ in ax_.get_major_ticks() + ax_.get_minor_ticks():
                for lab_ in (t_.label1, t_.label2):
                    if id(lab_) not in drawn:
                        undrawn.add(id(lab_))
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_text().strip() and t.get_visible()) or id(t) in undrawn:
            continue
        b = t.get_window_extent(rend)
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fw + 0.5 or b.y1 > fh + 0.5:
            out.append(t.get_text()[:40])
    check("%s every text inside the canvas" % name, not out, "; ".join(out))
    # no two drawn texts overlap (legend entries, labels, annotations, tick labels)
    boxes = [(t.get_text()[:25], t.get_window_extent(rend)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible() and id(t) not in undrawn]
    clash = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.0:
                clash.append("%s / %s" % (boxes[i][0], boxes[j][0]))
    check("%s no overlapping text" % name, not clash, "; ".join(clash[:8]))
    fig.savefig(OUT / (name + ".pdf"), bbox_inches=None)
    if px:  # graphical abstract: exact pixel size
        fig.savefig(OUT / (name + ".png"), dpi=px[0] / fig.get_size_inches()[0])
        Image.open(OUT / (name + ".png")).convert("RGB").save(OUT / (name + ".tif"), compression="tiff_lzw",
                                                                  dpi=(300, 300))
    else:
        fig.savefig(OUT / (name + ".png"), dpi=300)
        fig.savefig(OUT / (name + ".tif"), dpi=DPI_LINE, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    ALT[name] = alt
    SIZES[name] = width_mm


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, ls="-", fs=6.5, bold=False, tc=INK, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec,
                                lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p0, p1, ls="-", color=INK, cs="arc3,rad=0", lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0))


def example_day(npz):
    """Pre-stated rule: among fold-2024 days with at least 100 scored 5-min issue times, the day whose
    measured GHI has the largest mean absolute step-to-step change.
    """
    d = np.load(npz)
    df = pd.DataFrame(dict(day=pd.to_datetime(d["ts_ns"]).normalize(), y=d["y"]))
    g = df.groupby("day").agg(n=("y", "size"), sd=("y", lambda s: s.diff().abs().mean()))
    return g[g.n >= 100].sd.idxmax().strftime("%Y-%m-%d")


def fig1():
    import ce1_fig1
    npz = src("S5/yulara_f2024_h1_obs.npz")
    day = example_day(npz)
    c = pd.read_csv(src("S7/coverage_ci.csv")).set_index(["site", "fold", "horizon_min", "method", "variant", "nominal", "scope"])
    ci = round(float(c.loc[("yulara", 2024, 5, "icp", "static", 0.9, "transitional"), "PICP"]), 3)
    cm = round(float(c.loc[("yulara", 2024, 5, "mondrian_cqr", "static", 0.9, "transitional"), "PICP"]), 3)
    check("fig1 card 3 bars are 0.714 and 0.901", (ci, cm) == (0.714, 0.901))
    check("fig1 example day by the stated rule is 2024-01-25", day == "2024-01-25", day)
    fig = ce1_fig1.build(npz, day, ci, cm)
    save(fig, "fig1", W2,
         "Two-part icon figure. Part (a) shows the modelled Yulara mini-grid on one AC bus: PV arrays under a sun and "
         "a cloud, a pyranometer, a diesel station with one large and four small generator sets (some drawn online in "
         "orange, some offline in grey), a hypothetical battery drawn with a dashed outline, and hotel buildings as "
         "the load, with arrows for the direction of power flow. Part (b) is a chain of six numbered cards: (1) the "
         "measured irradiance of one broken-cloud day, (2) its 90% forecast interval and lower bound, (3) coverage in "
         "transitional periods of 0.714 for marginal ICP and 0.901 for Mondrian-CQR against a 90% target, (4) the reserve "
         "rule comparing diesel online under the handbook rule and with forecast PV counted, (5) fewer diesel sets online, "
         "and (6) outcomes of fuel and unserved energy. A dashed arrow below shows protocol A-R and the static "
         "margin rules fixing the level before the test year, and a separate loop shows protocol A-D, the only rule that "
         "updates it daily from observed unserved energy.")


STATIC = ["icp", "tod", "cqr", "mondrian", "mondrian_cqr"]


def errpoints(ax, x, m, v, lo, hi, dx=0.0, connect=True):
    lab, c, mk, ls = M[m]
    xx = np.asarray(x, float) + dx
    ax.errorbar(xx, v, yerr=[np.asarray(v) - lo, np.asarray(hi) - v], color=c, marker=mk, ms=3.2,
                lw=0.8 if connect else 0, ls=ls if connect else "none", elinewidth=0.7, capsize=0,
                mec="white", mew=0.4, label=lab)


def fig2():
    c = pd.read_csv(src("S7/coverage_ci.csv"))
    c = c[(c.nominal == 0.9) & (c.scope == "transitional") & (c.variant == "static")]
    s46 = pd.read_csv(src("S46/coverage.csv"))
    s46 = s46[(s46.definition == "base") & (s46.scope == "transitional") & (s46.horizon_min == 5)]
    fig, axs = plt.subplots(1, 3, figsize=(W2 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[1, 1.25, 1]))
    H = [5, 15, 30, 60]
    off = dict(zip(STATIC, np.linspace(-0.24, 0.24, len(STATIC))))
    for ax, site, letter, ttl in ((axs[0], "yulara", "a", "Yulara, fold 2024"), (axs[2], "asp", "c", "DKASC, fold 2024")):
        d = c[c.site == site]
        for m in STATIC:
            r = d[d.method == m].set_index("horizon_min").loc[H]
            errpoints(ax, np.arange(4), m, r.PICP.values, r.PICP_ci_lo.values, r.PICP_ci_hi.values, off[m])
        ax.set_xticks(range(4)); ax.set_xticklabels(H); ax.set_xlabel("Horizon (min)")
        ax.set_title(ttl, color=MUTED, loc="left", x=0.06); panel(ax, letter)
    folds = list(range(2019, 2025))
    for m in STATIC:
        r = s46[s46.method == m].set_index("fold").loc[folds]
        errpoints(axs[1], np.arange(6), m, r.picp.values, r.ci_lo.values, r.ci_hi.values, off[m] * 1.1)
    axs[1].set_xticks(range(6)); axs[1].set_xticklabels(folds); axs[1].set_xlabel("Test year (5 min)")
    axs[1].set_title("Yulara, six folds", color=MUTED, loc="left", x=0.05); panel(axs[1], "b")
    for ax in axs:
        ax.axhline(0.9, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=0)
        ax.set_ylim(0.62, 0.95); grid(ax)
    axs[0].set_ylabel("Transitional coverage at 90% nominal")
    axs[0].text(3.45, 0.903, "nominal", fontsize=6, color=MUTED, ha="right", va="bottom")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.25, wspace=0.28)
    y = c[(c.site == "yulara")].set_index(["method", "horizon_min"])
    check("fig2 ICP 5 min 0.714 [0.698, 0.731]",
          (round(y.loc[("icp", 5), "PICP"], 3), round(y.loc[("icp", 5), "PICP_ci_lo"], 3), round(y.loc[("icp", 5), "PICP_ci_hi"], 3)) == (0.714, 0.698, 0.731))
    check("fig2 Mondrian-CQR 5 min 0.901", round(y.loc[("mondrian_cqr", 5), "PICP"], 3) == 0.901)
    check("fig2 Mondrian-CQR fold 2024 0.896-0.901 over horizons",
          (round(y.loc["mondrian_cqr"].PICP.min(), 3), round(y.loc["mondrian_cqr"].PICP.max(), 3)) == (0.896, 0.901))
    f = s46.groupby("method").picp.agg(["min", "max"]).round(3)
    check("fig2 six-fold ICP 0.660-0.781, M-CQR 0.885-0.915, Mondrian 0.878-0.913",
          tuple(f.loc["icp"]) == (0.660, 0.781) and tuple(f.loc["mondrian_cqr"]) == (0.885, 0.915)
          and tuple(f.loc["mondrian"]) == (0.878, 0.913), str(f.loc[["icp", "mondrian", "mondrian_cqr"]].values.tolist()))
    # the drawn-below set is taken from the plotted series (S7 for a and c, S46 for b, upper
    # bound below 0.9), each cell is matched to S63/blocks7.csv on point value and both 1-day bounds, and the cells
    # whose 7-day interval is not below nominal must be exactly the ones the caption names
    b7 = pd.read_csv(src("S63/blocks7.csv"))
    b7 = b7[(b7.scope == "transitional") & b7.method.isin(STATIC)]
    plotted = []
    for site, pnl in (("yulara", "a"), ("asp", "c")):
        for r in c[(c.site == site) & c.method.isin(STATIC) & c.horizon_min.isin(H)].itertuples():
            plotted.append((pnl, site, 2024, r.horizon_min, r.method, r.PICP, r.PICP_ci_lo, r.PICP_ci_hi, 5.0001e-5))
    for r in s46[s46.method.isin(STATIC) & s46.fold.isin(folds)].itertuples():
        plotted.append(("b", "yulara", r.fold, 5, r.method, r.picp, r.ci_lo, r.ci_hi, 1e-9))
    below, mism = [], []
    for pnl, site, fo, h, m, v, lo, hi, tol in plotted:
        if hi >= 0.9:
            continue
        k = b7[(b7.site == site) & (b7.fold == fo) & (b7.horizon_min == h) & (b7.method == m)]
        if len(k) != 1 or max(abs(k.picp.iloc[0] - v), abs(k.ci1_lo.iloc[0] - lo), abs(k.ci1_hi.iloc[0] - hi)) > tol:
            mism.append((pnl, m, fo, h)); continue
        if k.status7.iloc[0] != "below":
            below.append((pnl, m, h if pnl != "b" else fo))
    check("fig2 drawn-below intervals match blocks7 on value and bounds (%d mismatches)" % len(mism), not mism, str(mism))
    a = c[c.site == "asp"].set_index(["method", "horizon_min"])
    check("fig2 DKASC ICP 0.710 (5) and 0.717 (60)",
          (round(a.loc[("icp", 5), "PICP"], 3), round(a.loc[("icp", 60), "PICP"], 3)) == (0.710, 0.717))
    check("fig2 DKASC Mondrian 0.893-0.898, CQR 0.881-0.889",
          (round(a.loc["mondrian"].PICP.min(), 3), round(a.loc["mondrian"].PICP.max(), 3)) == (0.893, 0.898)
          and (round(a.loc["cqr"].PICP.min(), 3), round(a.loc["cqr"].PICP.max(), 3)) == (0.881, 0.889))
    save(fig, "fig2", W2,
         "Three panels of dot-and-interval plots of transitional-period coverage at 90% nominal for five "
         "calibration methods, with a dashed line at 0.9. Panel (a), Yulara fold 2024 by horizon: marginal ICP "
         "and the time-of-day rule sit near 0.70 to 0.73, CQR near 0.84 to 0.89, Mondrian near 0.88 to 0.89, "
         "and Mondrian-CQR at 0.896 to 0.901 with bootstrap intervals containing 0.9. Panel (b), six test years at 5 min: "
         "marginal ICP ranges 0.66 to 0.78 while Mondrian-CQR ranges 0.885 to 0.915. Panel (c), DKASC, fold 2024: "
         "marginal ICP and the time-of-day rule again sit far below 0.9, and Mondrian (0.893 to 0.898) and "
         "Mondrian-CQR sit closest to 0.9, with CQR just below them.")


ACI = [("anticipative", "Anticipative", "#777777", "o", "--"),
       ("delayed", "Delayed", "#0072B2", "s", "-"),
       ("delayed_projected", "Delayed, projected", "#D55E00", "^", "-.")]


def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(W2 * MM, 58 * MM))
    stats = {}
    for site, lw_, alpha_ in (("yulara", 1.0, 1.0),):
        d = np.load(src("S5/%s_f2024_h12_obs.npz" % site))
        mon = pd.to_datetime(d["ts_ns"]).month
        for v, lab, col, mk, ls in ACI:
            cov = pd.Series(d["cov__aci|%s|0.9" % v]).groupby(mon).mean()
            wid = pd.Series(d["wid__aci|%s|0.9" % v]).groupby(mon).mean()
            axs[0].plot(cov.index, cov.values, color=col, marker=mk, ls=ls, label=lab, mec="white", mew=0.4)
            axs[1].plot(wid.index, wid.values, color=col, marker=mk, ls=ls, label=lab, mec="white", mew=0.4)
            stats[v] = (cov, d["cov__aci|%s|0.9" % v].mean())
    axs[0].axhline(0.9, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=0)
    axs[0].set_ylabel("Monthly coverage at 90% nominal"); axs[0].set_ylim(0.80, 1.0)
    axs[1].set_ylabel("Monthly mean width (W/m$^2$)"); axs[1].set_ylim(0, None)
    for ax, L in zip(axs, "ab"):
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_xlabel("Month of test year 2024")
        grid(ax); panel(ax, L)
    axs[0].legend(loc="lower right", ncol=1, handletextpad=0.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.18, wspace=0.26)
    rel = pd.read_csv(src("S5/reliability.csv"))
    rel = rel[(rel.site == "yulara") & (rel.fold == 2024) & (rel.horizon_min == 60) & (rel.method == "aci")]
    for v in ("anticipative", "delayed"):
        r = rel[rel.variant == v].PICP.values
        check("fig3 monthly coverage equals S5/reliability.csv (%s)" % v, np.allclose(np.sort(r), np.sort(stats[v][0].values), atol=5e-4))
    sd = {v: round(float(rel[rel.variant == v].PICP.std(ddof=1)), 4) for v in ("anticipative", "delayed")}
    check("fig3 monthly sd 0.0170 to 0.0242", (sd["anticipative"], sd["delayed"]) == (0.0170, 0.0242), str(sd))
    check("fig3 overall coverage 0.935 / 0.918 / 0.891 (3.3)",
          tuple(round(float(stats[v][1]), 3) for v in ("anticipative", "delayed", "delayed_projected")) == (0.935, 0.918, 0.891))
    save(fig, "fig3", W2,
         "Two line charts over the twelve months of test year 2024 at Yulara, 60-min horizon, for three ACI "
         "variants. Panel (a) monthly coverage at 90% nominal: the anticipative update stays above 0.9, the "
         "delayed update dips below 0.9 in some summer months, and the projected delayed update is lowest. "
         "Panel (b) monthly mean interval width: the delayed update is clearly the widest in every month, "
         "while the projected variant stays close to the anticipative width.")


POL = ["raw_q", "cqr", "mondrian_cqr", "nicp_var", "frac_pv", "icp", "mondrian", "tod_q"]


def envelope(ue, fuel):
    o = np.argsort(ue, kind="stable")
    return ue[o], np.minimum.accumulate(fuel[o])


def fig4():
    fold = 2024
    d = np.load(src("S6E/f%d_h3_daily.npz" % fold))
    a = json.loads(src("S6E/f%d_h3_select.json" % fold).read_text())["selections"]["main"]["anchors"]
    hb, orc = a["handbook"]["fuel_L"], a["oracle"]["fuel_L"]
    p, F, U, L = d["rows_policy"], d["main__fuel_L"].sum(1), d["main__ue_kWh"].sum(1), d["main__load_kWh"].sum(1)
    dr1 = pd.read_csv(src("S7E/dr1.csv"))
    dr1 = dr1[(dr1.fold == fold) & (dr1.h == 3) & (dr1.scenario == "main") & (dr1.protocol == "A_R")].set_index("policy")
    s51 = pd.read_csv(src("S51/cells.csv"))
    s51 = s51[(s51.primary) & (s51.fold == fold) & (s51.h == 3) & (s51.scenario == "main")].set_index("policy")
    xh = pd.read_csv(src("WP63/cases.csv"))
    xh = xh[(xh.fold == fold) & (xh.scenario == "main") & (xh.selection == "b_xfit_0.001")].set_index("policy")
    fig, axs = plt.subplots(1, 2, figsize=(W2 * MM, 80 * MM), gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax = axs[0]
    sav = lambda f: (hb - f) / hb * 100
    for m in POL:
        k = p == m
        u, e = envelope(U[k] / L[k] * 100, F[k])
        lab, c, mk, ls = M[m]
        ax.step(np.r_[0, u], sav(np.r_[hb, e]), where="post", color=c, ls=ls, lw=0.9, label=lab)
        r = dr1.loc[m]
        ax.plot(r.ue_share * 100, sav(r.fuel_L), marker=mk, color=c, mfc="white", mew=0.8, ms=4.2, ls="none")
        s = s51.loc[m]
        ax.plot(s.ue_share * 100, sav(s.fuel_L), marker=mk, color=c, mec="white", mew=0.4, ms=4.2, ls="none")
        x = xh.loc[m]
        ax.plot(x.ue_share * 100, sav(x.fuel_L), marker=mk, color=c, mec=INK, mew=0.9, ms=4.6, ls="none", zorder=4)
    ax.plot(0, 0, marker="o", color=INK, ms=4, ls="none", clip_on=False, zorder=5)
    # label placed right of the A-R markers so it overlaps no marker
    ax.annotate("Handbook rule (origin)", (0, 0), (0.0085, -0.13), fontsize=6, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0, shrinkB=2))
    ax.axhline(sav(orc), color=INK, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(0.0198, sav(orc) + 0.012, "Oracle (realized PV)", fontsize=6, color=INK, ha="right", va="bottom")
    ax.axvline(TARGET * 100, color=INK, lw=0.6, ls=(0, (4, 2)))
    ax.text(TARGET * 100 + 0.0003, 1.17, "UE target 0.002%", fontsize=6, color=INK, va="top")
    ax.set_xlim(0, 0.02); ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("Unserved energy (% of load energy)"); ax.set_ylabel("Fuel saving vs handbook rule (%)")
    grid(ax, "both"); panel(ax, "a")
    ax.plot([], [], marker="o", color=MUTED, mfc="white", ls="none", label="A-R level (open)")
    ax.plot([], [], marker="o", color=MUTED, ls="none", label="A-D controller (filled)")
    ax.plot([], [], marker="o", color=MUTED, mec=INK, mew=0.9, ls="none", label="XH static level (black edge)")
    ax.set_xticks([0, 0.005, 0.01, 0.015, 0.02]); ax.set_xticklabels(["0", "0.005", "0.010", "0.015", "0.020"])
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=1.2)
    # (b) share of the handbook-to-oracle gap captured at the target, six folds
    fr = pd.read_csv(src("S7E/frontier_F.csv"))
    fr = fr[(fr.h == 3) & (fr.scenario == "main")]
    cap = {}
    for _, r in fr.iterrows():
        an = json.loads(src("S6E/f%d_h3_select.json" % r.fold).read_text())["selections"]["main"]["anchors"]
        cap.setdefault(r.policy, []).append((an["handbook"]["fuel_L"] - r.F) / (an["handbook"]["fuel_L"] - an["oracle"]["fuel_L"]))
    ax = axs[1]
    order = sorted(POL, key=lambda m: -np.median(cap[m]))
    rng = np.random.default_rng(42)
    for i, m in enumerate(order):
        lab, c, mk, ls = M[m]
        v = np.array(cap[m])
        ax.plot(v, i + rng.uniform(-0.16, 0.16, len(v)), marker=mk, color=c, ls="none", ms=3.4, mec="white", mew=0.3, alpha=0.9)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([M[m][0] for m in order]); ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.5); ax.axvline(1, color=INK, lw=0.5, ls=(0, (1, 1.5)))
    ax.set_xlabel("Share of handbook-to-oracle gap\ncaptured at the UE target, ex post\non the test year (bar = median)")
    lo = min(min(v) for v in cap.values())
    ax.set_xlim(min(-0.1, lo - 0.05), 1.05); grid(ax, "x"); panel(ax, "b", x=-0.02)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.33, wspace=0.62)
    med = {m: round(float(np.median(cap[m])), 2) for m in POL}
    check("fig4 gap medians 0.76/0.76/0.76/0.56/0.46/0.39/0.44 (v2.1)",
          [med[m] for m in ("raw_q", "cqr", "mondrian_cqr", "nicp_var", "frac_pv", "icp", "mondrian")]
          == [0.76, 0.76, 0.76, 0.56, 0.46, 0.39, 0.44], str(med))
    check("fig4 (b) order in the alt text: quantile, var.-norm. ICP, fixed fraction, Mondrian ICP, time-of-day, marginal ICP",
          order[3:] == ["nicp_var", "frac_pv", "mondrian", "tod_q", "icp"] and med["mondrian"] > med["tod_q"] > med["icp"], str(order))
    check("fig4 (a) oracle line of fold 2024 near 1.05% (alt text; range 1.05 to 1.37) and inside the axis",
          round(float(sav(orc)), 2) == 1.05 and sav(orc) < 1.2, "%.4f" % sav(orc))
    xq = xh.loc[["raw_q", "cqr", "mondrian_cqr"]]
    check("fig4 (a) XH markers: 8 policies, fold 2024 shape A; 7 meet the target (the fixed fraction misses); quantile XH "
          "savings 0.58 to 0.63% and UE below the A-D markers (alt text)",
          len(xh) == 8 and int(xh.meets.sum()) == 7 and not bool(xh.loc["frac_pv"].meets)
          and (round(float(xq.saving_pct.min()), 2), round(float(xq.saving_pct.max()), 2)) == (0.58, 0.63)
          and float(xh.drop("frac_pv").ue_share.max()) < float(s51.ue_share.min())
          and bool((xh.fuel_L > 0).all()) and bool((xh.ue_share * 100 < 0.02).all()),
          f"met {int(xh.meets.sum())}, quantile {xq.saving_pct.min():.3f}-{xq.saving_pct.max():.3f}, ue max {xh.ue_share.max():.2e}")
    check("fig4 panel (a) envelope reproduces frontier F at the target (fold 2024)",
          all(abs(np.interp(TARGET * 100, *envelope(U[p == m] / L[p == m] * 100, F[p == m])) -
                  fr[(fr.fold == fold) & (fr.policy == m)].F.iloc[0]) < 0.02 * (hb - orc) for m in POL))
    save(fig, "fig4", W2,
         "Panel (a) is a step chart of fuel saving against the 100%-of-PV handbook rule versus test-year "
         "unserved energy for eight reserve policies, fold 2024, 15 min, load shape A. The handbook rule sits at "
         "the origin, the oracle is a dotted horizontal line near 1.05%, and a dashed vertical line marks the "
         "0.002% target. Open markers show the level chosen a year ahead (A-R), filled markers the daily "
         "controller (A-D), which sits closer to zero unserved energy, and black-edged markers the XH static level, "
         "which lies left of the controller at lower unserved energy for the seven level-based policies, with savings of "
         "0.58 to 0.63% for the three quantile policies, while the fixed fraction's XH level sits right of the target. "
         "Panel (b) is a dot plot of the share of "
         "the handbook-to-oracle gap captured at the target over six folds, an ex post quantity on the test year: the quantile policies lead with "
         "medians of 0.76, then variability-normalized ICP 0.56, fixed fraction 0.46, Mondrian ICP 0.44, "
         "the time-of-day rule just below it and marginal ICP 0.39.")


def fig5():
    c = pd.read_csv(src("S51/cells.csv"))
    c = c[(c.primary) & (c.h == 3) & (c.scenario.isin(["main", "shapeB"]))]
    fig = plt.figure(figsize=(W2 * MM, 70 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.6, 1], hspace=0.12)
    axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1])]
    axl = fig.add_subplot(gs[1, 1], sharex=axs[1])
    ax = axs[0]
    floor = 0.02  # zero UE drawn on its own strip below the axis break
    zero_y = 0.012
    for i, m in enumerate(POL):
        lab, col, mk, ls = M[m]
        r = c[c.policy == m]
        x = np.full(len(r), i, float)
        ar = r.ue_share_AR.values / TARGET
        ad = r.ue_share.values / TARGET
        ar = np.where(ar <= 0, zero_y, np.maximum(ar, floor))
        ad = np.where(ad <= 0, zero_y, np.maximum(ad, floor))
        j = np.random.default_rng(i).uniform(-0.09, 0.09, len(r))
        ax.plot(x - 0.18 + j, ar, marker=mk, color=col, mfc="white", mew=0.7, ms=3.6, ls="none")
        ax.plot(x + 0.18 + j, ad, marker=mk, color=col, mec="white", mew=0.3, ms=3.6, ls="none")
    ax.set_yscale("log"); ax.set_ylim(zero_y * 0.75, 8)
    ax.axhspan(zero_y * 0.75, 0.016, color="#f4f4f4", zorder=0, lw=0)
    ax.axhline(1, color=INK, lw=0.6, ls=(0, (4, 2)))
    ax.axhspan(1, 8, color="#f4f4f4", zorder=0, lw=0)
    ax.set_yticks([zero_y, 0.02, 0.1, 1]); ax.set_yticklabels(["0", "0.02", "0.1", "1"])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    # the note sits at the top of the shaded band, clear of the time-of-day markers
    ax.text(len(POL) - 0.5, 7.0, "above target", fontsize=6, color=MUTED, ha="right", va="top")
    SHORT = {"raw_q": "Raw\nquantile", "cqr": "CQR", "mondrian_cqr": "Mondrian-\nCQR", "nicp_var": "Var.-norm.\nICP",
              "frac_pv": "Fixed\nfraction", "icp": "Marginal\nICP", "mondrian": "Mondrian\nICP", "tod_q": "Time-of-\nday rule"}
    ax.set_xticks(range(len(POL))); ax.set_xticklabels([SHORT[m] for m in POL], fontsize=6)
    ax.set_xlim(-0.6, len(POL) - 0.4)
    ax.set_ylabel("Test-year unserved energy / target")
    ax.plot([], [], marker="o", color=MUTED, mfc="white", ls="none", label="A-R (year ahead)")
    ax.plot([], [], marker="o", color=MUTED, ls="none", label="A-D (daily feedback)")
    ax.legend(loc="upper left", ncol=2, borderaxespad=0.1)
    grid(ax); panel(ax, "a")
    # note inside the zero strip, left of the zero-UE markers, so it covers no marker
    ax.text(-0.5, zero_y, "no UE (values in (0, 0.02) drawn at 0.02)", fontsize=6, color=MUTED, va="center")
    # the A-D value closest to the target line is given, so the marker does not read as a miss
    ad_all = c.ue_share.values / TARGET
    k_ = int(np.argmax(ad_all)); m_ = c.policy.values[k_]
    i_ = POL.index(m_)
    ax.annotate("%.3f" % ad_all[k_], (i_ + 0.18, ad_all[k_]), (i_ + 0.3, 0.3), fontsize=6, color=INK, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0, shrinkB=2))
    check("fig5 highest A-D value is the time-of-day quantile at 0.999 of the target (below 1)",
          m_ == "tod_q" and "%.3f" % ad_all[k_] == "0.999" and ad_all[k_] < 1, "%s %.4f" % (m_, ad_all[k_]))
    # (b) one A-D level path for CQR, shape A: fold 2022, one of the two folds whose days at the limit lie nearest the
    # six-fold median (2019 at 268 and 2022 at 283 around a median of 275.5; checked below).
    fold, pol = 2022, "cqr"
    d = np.load(src("S51/f%d_h3_paths.npz" % fold))
    k = list(d["specs"]).index("%s|0.005|0.5" % pol)
    lev = d["main__level"][k]
    ue = d["main__ue_kWh"][k]
    ax = axs[1]
    days = np.arange(1, len(lev) + 1)
    ax.step(days, lev, where="post", color=M[pol][1], lw=0.9)
    hit = ue > 0
    ybot = np.floor(lev.min() * 100) / 100 - 0.01
    ax.plot(days[hit], np.full(hit.sum(), ybot + 0.003), marker="|", color="#D55E00", ls="none", ms=4, mew=0.7)
    ax.set_ylim(ybot, 1.004); ax.set_xlim(1, len(lev))
    tk = [t for t in np.arange(1.0, ybot, -0.02)][::-1]
    ax.set_yticks(tk); ax.set_yticklabels(["%.2f" % t for t in tk[:-1]] + ["limit"])
    ax.set_ylabel("Level applied, CQR")
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.text(len(lev) * 0.98, ybot + 0.006, "days with unserved energy", fontsize=6, color="#D55E00", ha="right", va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.6))
    # the latent (unclipped) level beside the applied level, on its own axis (AW record, h = infinity)
    zl = np.load(src("AW/latent.npz"))
    lat = zl["f%d|main|%s|inf" % (fold, pol)]
    # own panel, one axis, the limit drawn as a line (no second y-axis)
    axl.plot(np.arange(1, len(lat) + 1), lat, color=INK, lw=0.7, ls=(0, (2, 1.5)))
    axl.axhline(1.0, color="#D55E00", lw=0.6, ls=(0, (1, 1.2)))
    axl.text(len(lev) * 0.98, 1.0 + 0.03, "limit", fontsize=6, color="#D55E00", ha="right", va="bottom",
             bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)  # white box keeps the label clear of the latent line
    axl.set_ylim(np.floor(float(lat.min()) * 10) / 10, np.ceil(float(lat.max()) * 10) / 10 + 0.05)
    axl.set_ylabel("Latent level")
    axl.set_xlabel("Day of test year %d" % fold)
    grid(ax); grid(axl); panel(ax, "b")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.18, wspace=0.3)
    cs = c[c.policy.isin(["icp", "mondrian", "cqr", "mondrian_cqr"])]
    check("fig5 conformal 48/48 A-D vs 21/48 A-R; all 96/96 vs 45/96 (v2.1)",
          ((cs.ue_share <= TARGET).sum(), (cs.ue_share_AR <= TARGET).sum(), (c.ue_share <= TARGET).sum(), (c.ue_share_AR <= TARGET).sum(), len(c))
          == (48, 21, 96, 45, 96))
    _dl = c[(c.policy == pol) & (c.scenario == "main")].set_index("fold").days_at_limit
    _gap = (_dl - _dl.median()).abs()
    check("fig5 (b) fold %d is one of the folds nearest the six-fold median of CQR days at the limit" % fold,
          abs(_gap.loc[fold] - _gap.min()) < 1e-9, str(_dl.to_dict()))
    check("fig5 (a) highest A-R value below 4 times the target (alt text 'up to about four times')",
          3 < float((c.ue_share_AR / TARGET).max()) < 4, "%.3f" % float((c.ue_share_AR / TARGET).max()))
    r = c[(c.fold == fold) & (c.scenario == "main") & (c.policy == pol)].iloc[0]
    check("fig5 path days at limit equals S51 cells (%d)" % r.days_at_limit, int((lev >= 1).sum()) == int(r.days_at_limit))
    check("fig5 path UE equals S51 cells", abs(float(ue.sum()) - r.ue_kWh) < 1e-3 * max(1, r.ue_kWh))
    check("fig5 latent path starts at the A-R level and the applied level is at or above it (capped at the limit)",
          abs(float(lat[0]) - float(r.theta0)) < 1e-6 and bool(np.all(lev >= np.minimum(lat[:-1], 1.0) - 1e-6)))

    _md = c[c.policy.isin(["raw_q", "cqr", "mondrian_cqr"])].groupby("policy").days_at_limit.median()
    check("fig5 median days at limit 255-271 over quantile policies (v2.1; 270.5 printed as 271, half days rounded up)",
          (int(np.floor(_md.min() + 0.5)), int(np.floor(_md.max() + 0.5))) == (255, 271), str(_md.to_dict()))
    save(fig, "fig5", W2,
         "Panel (a) is a dot plot on a logarithmic axis of test-year unserved energy divided by the target for "
         "eight policies, six folds and two load shapes, 15 min, with cases of zero unserved energy on a strip at the "
         "bottom. Open markers (level chosen a year ahead) are often above the dashed target line at 1, up to about "
         "four times the target. Filled markers (daily controller) are all below 1, the highest at 0.999. Panel (b) has two "
         "stacked charts over the days of test year 2022: above, a step chart of the reserve level the controller applied "
         "to CQR in load shape A, with tick marks on the days that had unserved energy, and below it the unclipped latent level as a dashed "
         "line, with the limit at 1. After an unserved-energy event the latent level jumps above the limit, and the "
         "applied level stays at the handbook level until the latent level decays below the limit.")


def fig6():
    t = pd.read_csv(src("S7E/dr4.csv"))
    pts = []
    for _, r in t.iterrows():
        for col in [c for c in t.columns if c.startswith(("s0_", "s1_"))]:
            v = r[col].strip("()").split(",")
            pts.append((r.policy, float(v[0]), float(v[2])))
    P = pd.DataFrame(pts, columns=["policy", "nb", "wb"])
    fig, axs = plt.subplots(1, 2, figsize=(W2 * MM, 64 * MM), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    TH = [m for m in POL if m != "frac_pv"]  # f of the fixed fraction is not a level: counted, not drawn
    # (a) level difference (with minus without battery) against the level without battery; identical
    # cases are merged and drawn with marker area proportional to their count
    P["d"] = (P.wb - P.nb).round(6)
    Q = P[P.policy != "frac_pv"]
    offs = dict(zip(TH, np.linspace(-0.0012, 0.0012, len(TH))))
    for m in TH:
        r = Q[Q.policy == m].groupby(["nb", "d"]).size().reset_index(name="n")
        lab, col, mk, ls = M[m]
        ax.scatter(r.nb + offs[m], r.d, s=6 * r.n, marker=mk, color=col, edgecolors="white", linewidths=0.3,
                   alpha=0.85, label=lab, zorder=3)
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (4, 2)))
    ax.set_xlim(np.floor(Q.nb.min() * 50) / 50 - 0.005, 1.005)
    ax.set_ylim(np.floor(Q.d.min() * 50) / 50 - 0.01, 0.01)  # largest rise 0.002: no band above 0.01
    ax.set_xlabel("A-R level without battery"); ax.set_ylabel("Level with battery minus without")
    grid(ax, "both"); panel(ax, "a")
    h_, l_ = ax.get_legend_handles_labels()
    for hh in h_:
        hh.set_sizes([14])
    ax.legend(h_, l_, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6, handletextpad=0.2, labelspacing=0.35,
              title="Area: number of cases\n(most points: one)", title_fontsize=6)
    # (b) fuel decomposition by fold, quantile policies pooled (median per fold)
    w = pd.read_csv(src("WP52/cells.csv"))
    q = w[(w.h_min == 15) & (w["shape"] == "A") & (w.policy.isin(["raw_q", "cqr", "mondrian_cqr"]))]
    g = q.groupby("fold")[["battery_alone_pct", "forecast_alone_pct", "combined_pct"]].median()
    ax = axs[1]
    x = np.arange(len(g))
    bars = [("battery_alone_pct", "Battery alone", "#56B4E9", ""), ("forecast_alone_pct", "Forecast alone", "#E69F00", ""),
            ("combined_pct", "Battery and forecast", "#0072B2", "")]
    for k, (col, lab, fc, hatch) in enumerate(bars):
        ax.bar(x + (k - 1) * 0.26, g[col].values, width=0.24, color=fc, label=lab, edgecolor="white", linewidth=0.5)
    ax.plot(x + 0.26, g.battery_alone_pct.values + g.forecast_alone_pct.values, ls="none", marker="_", ms=7,
            mew=1.0, color=INK, label="Sum of the two alone")
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_xlabel("Test year (15 min, shape A, A-R)")
    ax.set_ylabel("Fuel saving vs handbook, no battery (%)")
    hh, ll = ax.get_legend_handles_labels(); o = [1, 2, 3, 0]
    ax.legend([hh[i] for i in o], [ll[i] for i in o], loc="upper right", ncol=2, fontsize=6, columnspacing=0.8, borderaxespad=0.1)
    ax.set_ylim(0, 2.1)
    # misses of the three quantile policies at the A-R level, marked on the bar they belong to:
    # forecast alone (no battery, level theta_nb) and battery and forecast (level theta_wb)
    miss_nb = (~q.groupby("fold").target_met_nb.apply(lambda s_: s_.astype(bool))).groupby(level=0).sum()
    miss_wb = (~q.groupby("fold").target_met_wb.apply(lambda s_: s_.astype(bool))).groupby(level=0).sum()
    for i_, fo in enumerate(g.index):
        for dx_, col_, nmiss in ((0.0, "forecast_alone_pct", int(miss_nb.loc[fo])), (0.26, "combined_pct", int(miss_wb.loc[fo]))):
            if nmiss:
                # on the dark combined bar the digit sits inside the bar top in white, so it never meets the sum dash
                _in = bool(dx_)
                ax.text(i_ + dx_, g[col_].loc[fo] - 0.04 if _in else g[col_].loc[fo] + 0.04, "%d" % nmiss, fontsize=6,
                        color="white" if _in else "#D55E00", ha="center", va="top" if _in else "bottom",
                        fontweight="bold")
    grid(ax); panel(ax, "b")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.17, wspace=0.62)  # panel letters inside the canvas
    _yl = axs[0].get_ylim()
    check("fig6 (a) largest rise below the fixed upper axis limit %.2f (the lower limit follows the data)" % _yl[1], Q.d.max() < _yl[1], "%s" % Q.d.max())
    check("fig6 DR4 289 lower, 93 equal, 2 higher of 384 (v2.1)",
          ((P.wb < P.nb).sum(), (P.wb == P.nb).sum(), (P.wb > P.nb).sum(), len(P)) == (289, 93, 2, 384))
    _gq = Q.groupby(["policy", "nb", "d"]).size()
    check("fig6 (a) alt text: drawn falls of at most 0.04, median fall 0.001 to 0.01 per policy, two rises, most markers single cases",
          round(float(Q.d.min()), 2) == -0.04 and Q.d.max() <= 0.01 and float(Q.groupby("policy").d.median().min()) >= -0.0105
          and float(Q.groupby("policy").d.median().max()) <= -0.0005 and (_gq == 1).sum() > len(_gq) / 2
          and int((Q.d > 0).sum()) == 2 and float(Q.d.max()) <= 0.0025 and (Q.d < 0).sum() > len(Q) / 2,
          "%s %s %d of %d" % (Q.d.min(), Q.groupby("policy").d.median().to_dict(), (_gq == 1).sum(), len(_gq)))
    s = pd.read_csv(src("WP52/summary.csv"))
    s = s[(s.h_min == 15) & (s["shape"] == "A") & (s.policy == "quantile_pooled")].iloc[0]
    check("fig6 pooled medians 0.40 / 0.93 / 1.07 (0.90-1.20) (v2.1)",
          (round(s.battery_alone_pct_median, 2), round(s.forecast_alone_pct_median, 2), round(s.combined_pct_median, 2),
           round(s.combined_pct_min, 2), round(s.combined_pct_max, 2)) == (0.40, 0.93, 1.07, 0.90, 1.20))
    check("fig6 forecast alone, plain median over the 18 cells 0.93",
          (round(q.forecast_alone_pct.median(), 2), len(q)) == (0.93, 18))
    check("fig6 misses marked: no battery 3 in 2020, 2021 and 2023; with battery 2 in 2019 and 3 in 2020, 2021, 2023, 2024 (4 and 9 of 18 met)",
          {int(k): int(v) for k, v in miss_nb.items() if v} == {2020: 3, 2021: 3, 2023: 3}
          and {int(k): int(v) for k, v in miss_wb.items() if v} == {2019: 2, 2020: 3, 2021: 3, 2023: 3, 2024: 3}
          and int(q.target_met_wb.astype(bool).sum()) == 4 and int(q.target_met_nb.astype(bool).sum()) == 9 and len(q) == 18)
    check("fig6 (b) alt text ranges: battery alone 0.35 to 0.45, forecast alone 0.75 to 1.1, combined 0.9 to 1.2, combined below the sum",
          0.35 <= g.battery_alone_pct.round(2).min() and g.battery_alone_pct.round(2).max() <= 0.45 and 0.75 <= g.forecast_alone_pct.round(2).min()
          and g.forecast_alone_pct.round(2).max() <= 1.1 and 0.9 <= g.combined_pct.round(2).min() and g.combined_pct.round(2).max() <= 1.2
          and bool((g.combined_pct < g.battery_alone_pct + g.forecast_alone_pct).all()), g.round(3).to_dict())
    check("fig6 bars within the pooled range", g.combined_pct.min() >= s.combined_pct_min - 1e-9 and g.combined_pct.max() <= s.combined_pct_max + 1e-9)
    save(fig, "fig6", W2,
         "Panel (a) is a scatter of the change in the year-ahead reserve level when the hypothetical battery "
         "is added (with minus without) against the level without it, seven level-based policies "
         "over folds, horizons and both load shapes, with marker area proportional to the number of identical cases "
         "and a dashed line at zero. Most points lie below zero, many on it and two just above it. Falls are small, a median of "
         "0.001 to 0.01 per policy and at most about 0.04. Panel (b) is a grouped bar chart for each test "
         "year 2019 to 2024 of the fuel saving against the handbook rule without a battery, under A-R at 15 min in load shape A, median of the three quantile policies, misses included: battery alone about "
         "0.35 to 0.45%, forecast alone about 0.75 to 1.1%, and both together about 0.9 to 1.2%, below the sum of "
         "the two, marked by a dash. Digits count the quantile policies whose year-ahead level misses the target: three "
         "of three in 2020, 2021 and 2023 without and with the battery, and with the battery also two of three in 2019 and three of three in 2024.")


def graphical_abstract():
    ts = pd.read_csv(src("WP63/table4_sets.csv")).set_index(["selection", "set"])
    ar, xf, ad = ts.loc[("A_R", "conformal")], ts.loc[("b_xfit_0.001", "quantile")], ts.loc[("A_D", "quantile")]
    ylv = json.loads(src("WP67/year_level.json").read_text())["rows"]["XH|quantile"]  # years lead
    _cd = {}
    for _f in ["WP67/sim/f%d_cases.csv" % _y for _y in range(2019, 2025)]:
        _z = pd.read_csv(src(_f))
        _z = _z[(_z.selection == "b_xfit_0.001") & _z.policy.isin(["raw_q", "cqr", "mondrian_cqr"])]
        _cd[_f] = bool((_z.ue_share_complete <= 2e-5).all())
    cd_years = sum(_cd.values())
    _ls = pd.read_csv(src("WP61/load_scales.csv"))
    _ce = _ls[_ls.scenario.isin(["main", "shapeB"])].ceiling_pct
    ceil_txt = "%.1f-%.1f%%" % (_ce.min(), _ce.max())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ce1_s64 as _S64
    su_txt = "%.2f to %.1f L" % (_S64.su_litres("low"), _S64.su_litres("high"))
    check("graphical abstract source strings, not drawn since kept as regression pins: 5 of 6 complete-day years, ceiling 0.9-1.4% (v2.1), start-up 0.85 to 8.5 L",
          (cd_years, len(_cd), ceil_txt, su_txt) == (5, 6, "0.9-1.4%", "0.85 to 8.5 L"), (cd_years, len(_cd), ceil_txt, su_txt))
    fig = plt.figure(figsize=(13.28, 5.31))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.28); ax.set_ylim(0, 5.31); ax.axis("off")
    fig.patch.set_facecolor("white")
    GA_TITLE = "Can probabilistic irradiance forecasts replace\nthe full-PV spinning-reserve rule?"
    GA_SUB = "Yulara PV-diesel mini-grid model, six held-out test years, 15-min horizon"
    ax.text(0.4, 4.72, GA_TITLE, fontsize=26, fontweight="bold", color=INK, va="center", linespacing=1.1)
    ax.text(0.4, 3.98, GA_SUB, fontsize=21, color=MUTED, va="center")
    steps = [("GHI\nrecord", "#ffffff"), ("Quantile\nforecast", "#ffffff"), ("Conformal\ncalibration", "#e8f1f8"),
             ("Reserve at\nlevel $\\theta$", "#fde9d9"), ("Diesel\ncommitment", "#e3e3e3")]
    widths = [1.45, 2.05, 2.45, 2.15, 2.45]
    gap, y, h = 0.42, 2.18, 1.0
    x = 0.4
    for i, ((t, fc), w) in enumerate(zip(steps, widths)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=INK, lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=21, color=INK, linespacing=1.05)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.04, y + h / 2), (x - 0.04, y + h / 2), arrowstyle="-|>",
                                         mutation_scale=20, lw=1.8, color=INK))
        if i == 2:  # calibration shapes coverage, not fuel (DR2, DR3)
            ax.text(x + w / 2, y + h + 0.25, "shapes coverage, not fuel", ha="center", va="center", fontsize=21,
                    color="#0072B2", style="italic")
        x += w + gap
    LBL_STATIC = LBL_FEED = None  # no arrow labels
    nums = [("%d of %d" % (ar.met, ar.n), "conformal cases meet the\ntarget, year-ahead levels\n(registered test)", "#D55E00"),
            ("%d of 6" % ylv["years_met"], "test years met on measured\nsteps, exploratory static\nlevel with a margin", "#0072B2"),
            ("%.2f%%" % xf.saving_median_pct, "median fuel saving, quantile\npolicies, before start-ups\n(break-even %.1f L per start)" % xf.breakeven_L_per_start, INK)]
    for i, (n, t, c) in enumerate(nums):
        xx = 0.4 + i * 4.35
        ax.text(xx, 1.5, n, fontsize=30, fontweight="bold", color=c, ha="left", va="center")
        ax.text(xx, 0.74, t, fontsize=21, color=INK, ha="left", va="center", linespacing=1.05)
    check("graphical abstract: 6 of 6 years, break-even 27.8 L (v2.1)", (int(ylv["years_met"]), "%.1f" % xf.breakeven_L_per_start) == (6, "27.8"))
    check("graphical abstract numbers equal the result tables (21 of 48; 36 of 36; 0.76%; v2.1)",
          (int(ar.met), int(ar.n), int(xf.met), int(xf.n), "%.2f" % xf.saving_median_pct) == (21, 48, 36, 36, "0.76"))
    check("graphical abstract complete-day years read from source (5 of 6)", (cd_years, len(_cd)) == (5, 6))
    flat = lambda t: " ".join(t.replace("\n", " ").split()).replace("$\\theta$", "theta")
    alt = ("Graphical abstract titled '" + flat(GA_TITLE) + "' Subtitle: " + GA_SUB + ". A five-step chain ("
           + ", ".join(flat(t) for t, _ in steps)
           + ") shows conformal calibration labelled 'shapes coverage, not fuel'. Three results at 15 min: "
           + ". ".join(n + " " + flat(t) for n, t, _ in nums) + ".")
    _drawn = [t_.get_text() for t_ in ax.texts if t_.get_text().strip()]
    _miss = [d for d in _drawn if flat(d) not in alt]
    check("graphical abstract alt text carries every text object drawn on the figure (%d read from the axes)" % len(_drawn),
          not _miss, _miss)
    _pt = [t_.get_fontsize() * 130.0 / (13.28 * 25.4) for t_ in ax.texts if t_.get_text().strip()]
    check("graphical abstract smallest text %.1f pt when shown 13 cm wide (min %.1f)" % (min(_pt), GA_MIN_PT), min(_pt) >= GA_MIN_PT - 1e-9)
    save(fig, "graphical_abstract", 1328 / 300 * 25.4, alt, px=(1328, 531))


# ------------------------------------------------------------------ figure checker
def figure_checker():
    res = []
    man = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {"figures": {}}
    for name, rec in man["figures"].items():
        tif = Image.open(OUT / (name + ".tif"))
        wpx, hpx = tif.size
        dpi = tif.info.get("dpi", (0, 0))[0]
        if name == "graphical_abstract":
            ok = (wpx, hpx) == (1328, 531)
            res.append(("%s size %dx%d px (1328 x 531)" % (name, wpx, hpx), ok))
        else:
            wmm = wpx / dpi * 25.4
            res.append(("%s width %.1f mm, %d dpi, %d px" % (name, wmm, round(dpi), wpx),
                        abs(wmm - rec["width_mm"]) < 0.6 and dpi >= 999))
        res.append(("%s alt text %d words" % (name, len(rec["alt_text"].split())), 30 <= len(rec["alt_text"].split()) <= (250 if name == "graphical_abstract" else 200)))
        res.append(("%s pdf md5 matches manifest" % name, md5(OUT / (name + ".pdf")) == rec["pdf_md5"]))
    return res


def main(argv):
    if "--check" not in argv:
        style()
        for f in (fig1, fig2, fig3, fig4, fig5, fig6, graphical_abstract):
            f()
        man = {"stage": "S8", "script": "03_code/ce1/ce1_figures.py", "script_md5": md5(Path(__file__)),
               "sources": {s: md5(R / s) for s in sorted(SOURCES)},
               "figures": {n: {"width_mm": round(SIZES[n], 1), "alt_text": ALT[n], "pdf_md5": md5(OUT / (n + ".pdf")),
                               "tif_md5": md5(OUT / (n + ".tif"))} for n in ALT},
               "checks": [dict(label=l, ok=o, detail=d) for l, o, d in CHECKS]}
        (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        (OUT / "alt_text.md").write_text("# Alt text for every figure (WP-23, S8)\n\n" +
                                         "\n\n".join("**%s.** %s" % (n, ALT[n]) for n in ALT) + "\n", encoding="utf-8")
    fails = 0
    for l, o, d in CHECKS:
        print("  %-4s %s%s" % ("ok" if o else "FAIL", l, "" if o else "  [%s]" % (d,)))
        fails += not o
    for l, o in figure_checker():
        print("  %-4s %s" % ("ok" if o else "FAIL", l))
        fails += not o
    print("S8 figures: %s" % ("ALL CHECKS PASS" if not fails else "%d FAIL" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
