#!/usr/bin/env python3
"""Partition-sensitivity scan for the Route S / Route H cutoff (SrTiO3).

The production assembly (scripts/compute_eta_SrTiO3.py) partitions the mode
sum at omega0 = 175 cm-1: Route S (bare-harmonic denominator,
gamma = -D/(2*omega0^2)) above the cut, Route H (renormalized denominator,
gamma = -D/(2*omega_r^2)) below it. Both routes consume the SAME
strained-cell eigenvalue derivative D = d(omega^2)/d(eps), so for every
stable mode in the 150-200 cm-1 shell both gamma_S and gamma_H are
computable from existing data — no new DFT is involved.

This script:
  1. inventories the shell coverage (Route S values in [150,175), Route H
     values in [175,200], counting any modes where the bare->renormalized
     eigenvalue map fails);
  2. recomputes the total eta_xyxy(300 K) with the partition at 150, 175,
     and 200 cm-1, holding every other ingredient fixed;
  3. tabulates gamma_S vs gamma_H per mode over the full 150-200 shell and
     reports the distribution of relative differences
     (gamma_H - gamma_S)/|gamma_S| = omega0^2/omega_r^2 - 1  (D cancels).

Reads : same inputs as compute_eta_SrTiO3.py
Writes: data/processed/reports/partition_sensitivity_SrTiO3.csv   (summary)
        data/processed/reports/partition_overlap_shell_SrTiO3.csv (per mode)

Usage: uv run python scripts/scan_partition_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402
from compute_eta_SrTiO3 import REPO, assemble, build_maps, load_vogt  # noqa: E402

CUTS_CM1 = [150.0, 175.0, 200.0]
SHELL = (150.0, 200.0)
T_K = 300


def shell_table(rows, map_lambda):
    """Per-mode gamma_S / gamma_H comparison over the 150-200 cm-1 shell."""
    out = []
    for r in rows:
        omega0 = r["omega_ref"]
        if r["acoustic"] or not (SHELL[0] <= omega0 <= SHELL[1]):
            continue
        D = r["D"]
        lam_r = float(map_lambda(np.sign(omega0) * omega0 * omega0))
        gamma_s = -D / (2.0 * omega0 * omega0)
        if lam_r > 0:
            omega_r = float(np.sqrt(lam_r))
            gamma_h = -D / (2.0 * lam_r)
            rel = (gamma_h - gamma_s) / abs(gamma_s) if gamma_s != 0 else float("nan")
        else:
            omega_r = float("nan")
            gamma_h = float("nan")
            rel = float("nan")
        out.append({"iq": r["iq"], "branch": r["branch"], "omega0": omega0,
                    "omega_r": omega_r, "D": D, "gamma_S": gamma_s,
                    "gamma_H": gamma_h, "rel_diff": rel})
    return out


def main() -> None:
    rows = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    vogt, _ = load_vogt()
    map_lambda = build_maps(T_K)[0]

    # 1. coverage inventory
    n_low = sum(1 for r in rows
                if not r["acoustic"] and 150.0 <= r["omega_ref"] < 175.0)
    n_high = sum(1 for r in rows
                 if not r["acoustic"] and 175.0 <= r["omega_ref"] <= 200.0)
    table = shell_table(rows, map_lambda)
    n_h_fail = sum(1 for t in table if not np.isfinite(t["gamma_H"]))
    print(f"shell coverage: {n_low} modes in [150,175), {n_high} in [175,200]; "
          f"renormalized-map failures in shell: {n_h_fail}")

    # 2. partition scan at 300 K
    etas = {}
    for cut in CUTS_CM1:
        eta, sec, flags = assemble(T_K, rows, vogt, cutoff_cm1=cut)
        etas[cut] = (eta, sec)
        print(f"cut {cut:5.0f} cm-1: eta_xyxy(300 K) = {eta:.4e} Pa s  "
              f"[S {sec['routeS']:.3e} | H-stable {sec['routeH_stable']:.3e} | "
              f"H-unstable {sec['routeH_unstable']:.3e} | Gamma "
              f"{sec['gamma_sector']:.3e}]")
    eta175 = etas[175.0][0]
    spread = (max(e for e, _ in etas.values()) - min(e for e, _ in etas.values())) / eta175
    print(f"total spread across the scan: {100.0 * spread:.2f}% of the "
          f"production (175 cm-1) value")

    # 3. route agreement in the overlap shell
    rel = np.array([t["rel_diff"] for t in table if np.isfinite(t["rel_diff"])])
    abs_rel = np.abs(rel)
    print(f"route agreement, {len(rel)} shell modes with both routes: "
          f"median |rel diff| = {100.0 * np.median(abs_rel):.2f}%, "
          f"mean = {100.0 * np.mean(abs_rel):.2f}%, "
          f"p90 = {100.0 * np.percentile(abs_rel, 90):.2f}%, "
          f"max = {100.0 * abs_rel.max():.2f}%")

    rep = REPO / "data" / "processed" / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    lines = [
        "# partition_sensitivity_SrTiO3.csv - scripts/scan_partition_sensitivity.py",
        "# Route S/H partition moved to 150/175/200 cm-1 with all other inputs fixed;",
        "# 300 K totals and sector decomposition. Shell coverage: "
        f"{n_low} modes in [150,175), {n_high} in [175,200], "
        f"{n_h_fail} renormalized-map failures.",
        f"# Route agreement over [150,200] ({len(rel)} modes, both routes): "
        f"median |(gamma_H-gamma_S)/gamma_S| = {100.0 * np.median(abs_rel):.2f}%, "
        f"mean = {100.0 * np.mean(abs_rel):.2f}%, "
        f"p90 = {100.0 * np.percentile(abs_rel, 90):.2f}%, "
        f"max = {100.0 * abs_rel.max():.2f}%",
        f"# Total spread across the scan: {100.0 * spread:.2f}% of the production value.",
        "cutoff_cm1,eta_total_Pas,eta_routeS_Pas,eta_routeH_stable_Pas,"
        "eta_routeH_unstable_Pas,eta_gamma_sector_Pas,delta_vs_175_percent",
    ]
    for cut in CUTS_CM1:
        eta, sec = etas[cut]
        lines.append(f"{cut:.0f},{eta:.6e},{sec['routeS']:.6e},"
                     f"{sec['routeH_stable']:.6e},{sec['routeH_unstable']:.6e},"
                     f"{sec['gamma_sector']:.6e},"
                     f"{100.0 * (eta / eta175 - 1.0):+.3f}")
    out = rep / "partition_sensitivity_SrTiO3.csv"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out.relative_to(REPO)}")

    lines = [
        "# partition_overlap_shell_SrTiO3.csv - scripts/scan_partition_sensitivity.py",
        "# Per-mode Route S vs Route H comparison over the 150-200 cm-1 shell at 300 K.",
        "# gamma_S = -D/(2*omega0^2); gamma_H = -D/(2*omega_r^2), omega_r from the",
        "# 300 K bare->renormalized eigenvalue map; rel_diff = (gamma_H-gamma_S)/|gamma_S|.",
        "q_index,branch,omega0_cm1,omega_r_cm1,D_cm2,gamma_S,gamma_H,rel_diff",
    ]
    for t in table:
        lines.append(f"{t['iq']},{t['branch']},{t['omega0']:.4f},"
                     f"{t['omega_r']:.4f},{t['D']:.4f},{t['gamma_S']:.6f},"
                     f"{t['gamma_H']:.6f},{t['rel_diff']:.6f}")
    out = rep / "partition_overlap_shell_SrTiO3.csv"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
