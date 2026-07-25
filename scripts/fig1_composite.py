#!/usr/bin/env python3
"""Figure 1 (composite): vibrational input and viscosity output.

Panels:
  (a) SrTiO3 harmonic dispersion (PBEsol solid, PBE audit underlay) with
      the renormalized 300 K soft-mode anchor (Vogt 1995);
  (b) BaTiO3 harmonic dispersion with the 453 K INS points (Tomeno 2020);
  (c) mode-resolved decomposition of eta_xyxy(300 K) with the 175 cm-1
      Route S / Route H partition line.
The spectra show what the modes are; the decomposition shows where the
viscosity lives.

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/harmonic_dispersion_<material>.csv
  data/processed/ins_reference_points_<material>.csv
  data/processed/eta_spectral_density_SrTiO3.csv
    <- scripts/audit_eta_assembly.py::export_spectral_density

Writes figures/fig1_composite.{png,pdf}.
Usage: uv run python scripts/fig1_composite.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "fig1_composite"
CUTOFF = 175.0

INK = "#1a1a1a"
GRID = "#d9d9d9"
BRANCH = "#6b7280"
BRANCH_PBE = "#c7cbd1"
ANCHOR_INS = "#C2410C"
ANCHOR_RENORM = "#1D4ED8"
PATH_POINTS = {0: r"$\Gamma$", 40: "X", 80: "M", 120: r"$\Gamma$", 160: "R", 200: "X"}

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
})


def anchor_positions(dispersion, anchors):
    points = dispersion.drop_duplicates("path_index")[
        ["path_index", "qx", "qy", "qz", "path_coord"]]
    out = []
    for _, row in anchors.iterrows():
        q = np.array([row.qx, row.qy, row.qz], dtype=float)
        d = np.linalg.norm(points[["qx", "qy", "qz"]].values - q, axis=1)
        best = points.iloc[int(np.argmin(d))]
        out.append((float(best.path_coord), float(row.value_cm1)))
    return out


def draw_dispersion(ax, material, label):
    dispersion = pd.read_csv(
        REPO / "data" / "processed" / f"harmonic_dispersion_{material}.csv", comment="#")
    ticks = [dispersion[dispersion.path_index == i].path_coord.iloc[0]
             for i in PATH_POINTS]
    functionals = list(dispersion.functional.unique())
    primary = "pbesol" if "pbesol" in functionals else "pbe"
    for functional, color, width, z in (("pbe", BRANCH_PBE, 0.6, 1.5),
                                        (primary, BRANCH, 0.8, 2)):
        if functional not in functionals:
            continue
        sub = dispersion[dispersion.functional == functional]
        for branch in sorted(sub.branch.unique()):
            block = sub[sub.branch == branch].sort_values("path_index")
            ax.plot(block.path_coord, block.omega_cm1, lw=width, color=color, zorder=z)
    ax.axhline(0.0, lw=0.6, color=GRID, zorder=1)
    for t in ticks[1:-1]:
        ax.axvline(t, lw=0.5, color=GRID, zorder=1)
    dispersion = dispersion[dispersion.functional == primary]
    anchors = pd.read_csv(
        REPO / "data" / "processed" / f"ins_reference_points_{material}.csv", comment="#")
    if material == "BaTiO3":
        shown = anchors[anchors.method.isin(["digitized", "printed"])]
        xy = anchor_positions(dispersion, shown)
        ax.scatter([p[0] for p in xy], [p[1] for p in xy], s=22,
                   facecolor=ANCHOR_INS, edgecolor="white", linewidth=0.6, zorder=3,
                   label="INS 453 K (Tomeno 2020)")
    else:
        ax.scatter([0.0], [89.24], s=30, marker="D", facecolor=ANCHOR_RENORM,
                   edgecolor="white", linewidth=0.6, zorder=4, clip_on=False,
                   label="soft mode 300 K, renorm. (Vogt 1995)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.8, handletextpad=0.4)
    ax.set_xticks(ticks)
    ax.set_xticklabels(PATH_POINTS.values())
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(-300, 850)
    ax.set_ylabel(r"$\omega$ (cm$^{-1}$)")
    pretty = {"SrTiO3": r"SrTiO$_3$", "BaTiO3": r"BaTiO$_3$"}[material]
    ax.set_title(f"({label}) {pretty}", loc="left", fontsize=9)
    ax.tick_params(length=3)


def draw_decomposition(ax, label):
    rows = [line.split(",") for line in
            (REPO / "data" / "processed" / "eta_spectral_density_SrTiO3.csv")
            .read_text().splitlines()
            if line and not line.startswith("#") and not line.startswith("omega0")]
    lo = np.array([float(r[0]) for r in rows])
    hi = np.array([float(r[1]) for r in rows])
    centers, width = 0.5 * (lo + hi), hi - lo
    routeS = np.array([float(r[2]) for r in rows])
    h_stab = np.array([float(r[3]) for r in rows])
    h_unst = np.array([float(r[4]) for r in rows])
    g_sec = np.array([float(r[5]) for r in rows])
    bottom = np.zeros_like(centers)
    for series, lab, color in [
            (h_stab, "Route H, stable", "#4878a8"),
            (h_unst + g_sec, r"Route H, formerly unstable + $\Gamma$ sector", "#a84848"),
            (routeS, "Route S", "#6aa86a")]:
        ax.bar(centers, series * 1e3, width=width * 0.92, bottom=bottom * 1e3,
               label=lab, color=color, alpha=0.88)
        bottom = bottom + series
    peak = float((h_stab + h_unst + g_sec + routeS).max())
    ax.set_ylim(0, peak * 1e3 * 1.12)
    ax.axvline(CUTOFF, color="k", ls="--", lw=0.9)
    ax.annotate("Route S / H partition\n175 cm$^{-1}$ (Richardson)",
                xy=(CUTOFF, peak * 1e3 * 0.35), xytext=(300, peak * 1e3 * 0.55),
                fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.axvspan(lo[0], 0.0, color="0.9", zorder=0)
    ax.text(-50, peak * 1e3 * 0.98, "bare-imaginary\nmanifold",
            fontsize=6.2, color="0.35", ha="center", va="top")
    total = (routeS + h_stab + h_unst + g_sec).sum()
    ax.set_xlabel(r"bare harmonic frequency $\omega_0$ (cm$^{-1}$)")
    ax.set_ylabel(r"$\eta$ per 25 cm$^{-1}$ bin ($10^{-3}$ Pa s)")
    ax.set_title(rf"(c) SrTiO$_3$ $\eta_{{xyxy}}$ decomposition, 300 K "
                 rf"(total {total * 1e3:.2f}$\times 10^{{-3}}$ Pa s)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=6.8, loc="upper right", frameon=False)
    ax.set_xlim(lo[0], 900)
    ax.tick_params(length=3)


def main() -> None:
    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.34, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_dispersion(ax_a, "SrTiO3", "a")
    draw_dispersion(ax_b, "BaTiO3", "b")
    draw_decomposition(ax_c, "c")
    OUT.parent.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{OUT}.{suffix}", dpi=220, bbox_inches="tight")
    print(f"-> {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
