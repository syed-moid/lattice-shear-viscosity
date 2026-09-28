#!/usr/bin/env python3
"""Figure 1 (composite): vibrational input and viscosity output.

Panels:
  (a) SrTiO3, hybrid model: bare harmonic dispersion of the example harmonic set (grey) and its SCPH
      renormalisation at 300 K (inner mesh 12^3; blue), with the 300 K soft-mode anchor (Vogt 1995);
  (b) BaTiO3 harmonic dispersion with the 453 K INS points (Tomeno 2020);
  (c) mode-resolved decomposition of eta_xyxy(300 K) of the hybrid model by the bare frequency of each
      renormalised mode's partner, for the three coupling constructions.
The spectra show what the modes are; the decomposition shows where the
viscosity lives.

Data provenance (committed CSVs only, no hand-edited data):
  data/processed/harmonic_dispersion_<material>.csv
  data/processed/ins_reference_points_<material>.csv
  data/processed/dispersion_hybrid_SrTiO3.csv  <- scripts/export_hybrid_dispersion.py
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


def draw_hybrid(ax, label):
    d = pd.read_csv(REPO / "data" / "processed" / "dispersion_hybrid_SrTiO3.csv", comment="#")
    ticks = [d[d.path_index == i].path_coord.iloc[0] for i in PATH_POINTS]
    for b in sorted(d.branch.unique()):
        blk = d[d.branch == b].sort_values("path_index")
        ax.plot(blk.path_coord, blk.omega_bare_cm1, lw=0.6, color=BRANCH_PBE, zorder=1.5,
                label="bare harmonic (example set)" if b == 1 else None)
        ax.plot(blk.path_coord, blk.omega_scph300_cm1, lw=0.8, color=ANCHOR_RENORM, zorder=2,
                label="SCPH 300 K (inner mesh 12³)" if b == 1 else None)
    ax.axhline(0.0, lw=0.6, color=GRID, zorder=1)
    for t in ticks[1:-1]:
        ax.axvline(t, lw=0.5, color=GRID, zorder=1)
    ax.scatter([0.0], [89.24], s=30, marker="D", facecolor=ANCHOR_INS, edgecolor="white", linewidth=0.6, zorder=4,
               clip_on=False, label="soft mode 300 K (Vogt 1995)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.4)
    ax.set_xticks(ticks)
    ax.set_xticklabels(PATH_POINTS.values())
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(-300, 850)
    ax.set_ylabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_title(f"({label}) SrTiO$_3$, hybrid model", loc="left", fontsize=9)
    ax.tick_params(length=3)


def draw_decomposition(ax, label):
    rows = [line.split(",") for line in
            (REPO / "data" / "processed" / "eta_spectral_density_SrTiO3.csv")
            .read_text().splitlines()
            if line and not line.startswith("#") and not line.startswith("omega0")]
    lo = np.array([float(r[0]) for r in rows])
    hi = np.array([float(r[1]) for r in rows])
    centers, width = 0.5 * (lo + hi), hi - lo
    eta_c = np.array([float(r[2]) for r in rows])
    eta_g = np.array([float(r[3]) for r in rows])
    eta_b = np.array([float(r[4]) for r in rows])
    eta_a = np.array([float(r[5]) for r in rows])
    ax.bar(centers, eta_c * 1e3, width=width * 0.92, label="projected SCPH coupling (C)", color="#4878a8", alpha=0.88)
    ax.bar(centers, eta_g * 1e3, width=width * 0.92, bottom=eta_c * 1e3, label=r"$\Gamma$ TO1 triplet (Vogt)",
           color="#c8a848", alpha=0.88)
    ax.step(np.append(lo, hi[-1]), np.append(eta_b, eta_b[-1]) * 1e3, where="post", color="#a84848", lw=1.0,
            label="projected harmonic coupling (B)")
    ax.step(np.append(lo, hi[-1]), np.append(eta_a, eta_a[-1]) * 1e3, where="post", color="0.3", lw=0.9, ls="--",
            label="maximum-overlap transfer (A)")
    peak = float(max((eta_c + eta_g).max(), eta_b.max(), eta_a.max()))
    ax.set_ylim(0, peak * 1e3 * 1.12)
    ax.axvspan(lo[0], 0.0, color="0.9", zorder=0)
    ax.text(-50, peak * 1e3 * 0.98, "bare-imaginary\npartner manifold",
            fontsize=6.2, color="0.35", ha="center", va="top")
    total = (eta_c + eta_g).sum()
    ax.set_xlabel(r"bare frequency $\omega_0$ of the partner mode (cm$^{-1}$)")
    ax.set_ylabel(r"$\eta$ per 25 cm$^{-1}$ bin ($10^{-3}$ Pa s)")
    ax.set_title(rf"(c) SrTiO$_3$ $\eta_{{xyxy}}$ decomposition, hybrid model, 300 K "
                 rf"(C total {total * 1e3:.3f}$\times 10^{{-3}}$ Pa s, finite-mesh result)",
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
    draw_hybrid(ax_a, "a")
    draw_dispersion(ax_b, "BaTiO3", "b")
    draw_decomposition(ax_c, "c")
    OUT.parent.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{OUT}.{suffix}", dpi=220, bbox_inches="tight")
    print(f"-> {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
