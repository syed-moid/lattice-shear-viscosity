#!/usr/bin/env python3
"""Figure 1: harmonic phonon dispersions with experimental anchor points.

Panels: (a) SrTiO3, (b) BaTiO3, along Gamma-X-M-Gamma-R-X. PBEsol branches
are drawn solid; the PBE audit-reference branches are the faint underlay.
Imaginary harmonic modes are plotted as negative frequencies (standard
convention).
Anchor points come from data/processed/ins_reference_points_<material>.csv:
BaTiO3 INS points (Tomeno 2020, 453 K) sit at their q positions; the
SrTiO3 zone-center point is the renormalized soft mode (Vogt 1995, 300 K),
which anchors the renormalized theory, not the bare harmonic curve.

The anharmonic linewidth panel is added in Stage B once phono3py
linewidths exist.

Reads : data/processed/harmonic_dispersion_<material>.csv
        data/processed/ins_reference_points_<material>.csv
Writes: figures/fig1_dispersion_linewidths.pdf and .png

Usage: uv run python scripts/fig1_dispersion_linewidths.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MEV_TO_CM1 = 8.06554

INK = "#1a1a1a"
GRID = "#d9d9d9"
BRANCH = "#6b7280"          # PBEsol (production functional)
BRANCH_PBE = "#c7cbd1"      # PBE audit reference, faint underlay
ANCHOR_INS = "#C2410C"  # BaTiO3 INS points (measured, 453 K)
ANCHOR_RENORM = "#1D4ED8"  # SrTiO3 renormalized soft mode (300 K)

PATH_POINTS = {0: r"$\Gamma$", 40: "X", 80: "M", 120: r"$\Gamma$", 160: "R", 200: "X"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
    }
)


def anchor_positions(dispersion: pd.DataFrame, anchors: pd.DataFrame) -> list[tuple]:
    """Map each anchor q to a path coordinate by nearest dispersion q-point."""
    points = dispersion.drop_duplicates("path_index")[["path_index", "qx", "qy", "qz", "path_coord"]]
    positions = []
    for _, row in anchors.iterrows():
        q = np.array([row.qx, row.qy, row.qz], dtype=float)
        distances = np.linalg.norm(points[["qx", "qy", "qz"]].values - q, axis=1)
        best = points.iloc[int(np.argmin(distances))]
        positions.append((float(best.path_coord), float(row.value_cm1)))
    return positions


def draw_panel(ax, material: str, label: str) -> None:
    dispersion = pd.read_csv(
        REPO / "data" / "processed" / f"harmonic_dispersion_{material}.csv", comment="#"
    )
    ticks = [
        dispersion[dispersion.path_index == i].path_coord.iloc[0] for i in PATH_POINTS
    ]
    functionals = list(dispersion.functional.unique())
    primary = "pbesol" if "pbesol" in functionals else "pbe"
    for functional, color, width, z in (
        ("pbe", BRANCH_PBE, 0.7, 1.5),
        (primary, BRANCH, 0.9, 2),
    ):
        if functional not in functionals or (functional == "pbe" and primary == "pbe" and color == BRANCH_PBE):
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
        REPO / "data" / "processed" / f"ins_reference_points_{material}.csv", comment="#"
    )
    if material == "BaTiO3":
        shown = anchors[anchors.method.isin(["digitized", "printed"])]
        xy = anchor_positions(dispersion, shown)
        ax.scatter(
            [p[0] for p in xy], [p[1] for p in xy],
            s=26, facecolor=ANCHOR_INS, edgecolor="white", linewidth=0.6, zorder=3,
            label="INS 453 K (Tomeno 2020)",
        )
    else:
        # single zone-center renormalized soft-mode anchor at 300 K
        omega_300 = 89.24
        ax.scatter(
            [0.0], [omega_300],
            s=34, marker="D", facecolor=ANCHOR_RENORM, edgecolor="white", linewidth=0.6,
            zorder=4, clip_on=False, label="soft mode 300 K, renorm. (Vogt 1995)",
        )
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.4)

    ax.set_xticks(ticks)
    ax.set_xticklabels(PATH_POINTS.values())
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylabel(r"$\omega$ (cm$^{-1}$)")
    pretty = {"SrTiO3": r"SrTiO$_3$", "BaTiO3": r"BaTiO$_3$"}[material]
    ax.set_title(f"({label}) {pretty}", loc="left", fontsize=9)
    ax.tick_params(length=3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
    draw_panel(axes[0], "SrTiO3", "a")
    draw_panel(axes[1], "BaTiO3", "b")
    axes[0].set_ylim(-300, 850)
    axes[1].set_ylim(-300, 850)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        out = REPO / "figures" / f"fig1_dispersion_linewidths.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
