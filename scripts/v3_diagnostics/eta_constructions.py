#!/usr/bin/env python3
"""eta_constructions.py - zone-wide eta_A, eta_B, eta_C on the own-surface reference
(diagnostic model only; no production file is touched).

Reference held FIXED for the three constructions: own-surface unstrained SCPH set at 300 K
(IFC supercell 2x2x2 quartic set, KMESH_INTERPOLATE 2x2x2, KMESH_SCPH 2x2x2, SELF_OFFDIAG = 0 or 1),
renormalised frequencies omega_r and eigenvectors e_r from dynmat.py (validated against anphon),
linewidths by the production character-aware frequency-class map built from the own-surface
8x8x8 RTA result of the same SCPH variant, Bose weights at omega_r, stress-correlator kernel
tau = 1/(2 Gamma) + 2 Gamma/omega^2, Vogt anchoring of the Gamma-point soft triplet.  The three
constructions differ ONLY in the coupling Lambda(nu):
  A  Lambda_bare(mu*) of the max-overlap bare partner (Route-H basis transfer),
  B  multiplet-projected e_r^dagger K_HA e_r,
  C  multiplet-projected e_r^dagger K_SCPH e_r,
each also with the Richardson-combined step (0.010 / 0.020).  gamma = -Lambda/(2 omega_r^2) for
every sector (the production's bare-frequency rule for Route S is NOT applied here, so that the
same omega_r enters all three; the Route-S sector is 6 % of eta).  No floor (H1).
Character class of nu (for the linewidth rule) = bare omega_0 of the max-overlap partner.

Writes data/processed/v3_diagnostics/eta_constructions_zone.csv        (totals per variant and mesh, shells, top-20 share)
       data/processed/v3_diagnostics/eta_constructions_bins.csv        (11^3: sector decomposition by bare-omega_0 bin, ratios)
       data/processed/v3_diagnostics/eta_constructions_modes_<variant>_mesh11.csv (per-mode table, 11^3)
Usage: uv run python eta_constructions.py od1 [od0] [--meshes 9 11 13 15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from coupling_kernel import Sets, evaluate_q  # noqa: E402
from compute_eta_SrTiO3 import (  # noqa: E402
    CM1, CUTOFF_CM1, OMEGA_MIN, SOFT_CHAR_CM1, V_CELL, _gamma_map_from_results, load_vogt,
)
from crosscheck_alamode_sto_tau import parse_result  # noqa: E402
from latvisc.viscosity import bose_einstein, tau_two_pole_stress  # noqa: E402
from scipy.constants import Boltzmann as K_B, hbar as HBAR  # noqa: E402

ALAMODE = REPO / "data" / "raw" / "alamode_sto"
RTA = {"od0": ALAMODE / "own_surface_i2s2" / "STO_RTA_own_i2s2_300K.result",
       "od1": ALAMODE / "own_surface_6p2" / "STO_RTA_own_reference_od1_300K.result"}
RTA_BARE = ALAMODE / "own_surface_i2s2" / "STO_RTA_own_i2s2_bare.result"
BINS = [(-np.inf, 0), (0, 25), (25, 50), (50, 100), (100, 150), (150, 175), (175, 300), (300, np.inf)]
T = 300


def bin_label(w):
    for lo, hi in BINS:
        if lo <= w < hi:
            return f"[{lo:.0f},{hi if np.isfinite(hi) else 'inf'})" if np.isfinite(lo) else "imaginary"
    return "?"


def mesh_points(n):
    return [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)]


def shell(q, n):
    d = np.abs((np.asarray(q) + 0.5) % 1.0 - 0.5) * n
    return int(round(d.max()))


def zone(sets, n, map_gamma, soft_median, vogt_w, vogt_g):
    qs = mesh_points(n)
    norm = 1.0 / (V_CELL * len(qs) * K_B * T)
    rows = []
    t0 = time.time()
    for iq, q in enumerate(qs):
        res = evaluate_q(sets, q, matched_C=False)
        wr, wb = res["omega_r"], res["omega_b"]
        mu = res["A"]["mu_star"]
        for nu in range(15):
            omega0 = wb[mu[nu]]
            omega_r = wr[nu]
            if iq == 0 and omega_r < OMEGA_MIN:
                continue                                   # acoustic zeros at Gamma
            if omega_r <= 0:
                continue
            if iq == 0 and omega0 < OMEGA_MIN:
                sector, omega_use, gamma_hwhm = "gamma_sector", vogt_w, vogt_g
            elif omega0 < OMEGA_MIN:
                sector, omega_use, gamma_hwhm = "routeH_unstable", omega_r, soft_median
            elif omega0 < SOFT_CHAR_CM1:
                sector, omega_use, gamma_hwhm = "routeH_stable", omega_r, soft_median
            else:
                sector = "routeS" if omega0 >= CUTOFF_CM1 else "routeH_stable"
                omega_use, gamma_hwhm = omega_r, float(map_gamma(omega_r))
            w = omega_use * CM1
            occ = bose_einstein(w, T)
            tau = float(tau_two_pole_stress(w, gamma_hwhm * CM1))
            pref = (HBAR * w) ** 2 * occ * (occ + 1.0) * tau * norm / (4.0 * omega_use ** 4)   # eta = pref * Lambda^2
            C = res["C"]
            lam = {"A": res["A"]["lam"][nu], "B": res["B"]["005"]["eig"][nu], "C": C["005"]["eig"][nu],
                   "Brich": res["B"]["rich"]["eig"][nu], "Crich": C.get("rich", C["005"])["eig"][nu],
                   "B020": res["B"]["010"]["eig"][nu], "C020": C.get("010", C["005"])["eig"][nu]}
            rows.append({"iq": iq, "qa": q[0], "qb": q[1], "qc": q[2], "shell": shell(q, n), "nu": nu + 1,
                         "omega_r_cm1": omega_r, "omega_use_cm1": omega_use, "omega0_partner_cm1": omega0,
                         "partner_mu": int(mu[nu]) + 1, "overlap_single": res["A"]["overlap_single"][nu],
                         "overlap_multiplet": res["A"]["overlap_summed"][nu], "sector": sector,
                         "omega0_bin": bin_label(omega0), "Gamma_hwhm_cm1": gamma_hwhm, "tau_ps": tau * 1e12,
                         "Lambda_A": lam["A"], "Lambda_B": lam["B"], "Lambda_C": lam["C"],
                         "Lambda_B_rich": lam["Brich"], "Lambda_C_rich": lam["Crich"],
                         "Lambda_B_h020": lam["B020"], "Lambda_C_h020": lam["C020"], "pref_eta_per_Lambda2": pref,
                         "eta_A": pref * lam["A"] ** 2, "eta_B": pref * lam["B"] ** 2, "eta_C": pref * lam["C"] ** 2,
                         "eta_B_rich": pref * lam["Brich"] ** 2, "eta_C_rich": pref * lam["Crich"] ** 2})
        if iq % 500 == 0:
            print(f"    mesh {n}: {iq}/{len(qs)} q ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=["od1", "od0"])
    ap.add_argument("--meshes", type=int, nargs="*", default=[9, 11, 13, 15])
    ap.add_argument("--out-prefix", default="eta_constructions", help="output file prefix (default eta_constructions)")
    ap.add_argument("--surface-dir", type=Path, default=None, help="directory holding renorm_own_<set>_<variant>_300K.xml (default own_surface_6p2)")
    ap.add_argument("--rta", type=Path, default=None, help="RTA .result of the unstrained surface (default per variant)")
    ap.add_argument("--xml-pattern", default=None, help="renormalised XML name pattern with {tag} and {variant}")
    args = ap.parse_args()
    P = args.out_prefix
    vogt, _ = load_vogt()
    vogt_w, vogt_g = vogt(T)
    freq_bare, _ = parse_result(RTA_BARE, target_temp=T)
    zone_rows, bin_rows = [], []
    for variant in args.variants:
        freq_ren, gamma_ren = parse_result(args.rta or RTA[variant], target_temp=T)
        map_gamma, soft_median, _ = _gamma_map_from_results(freq_bare, freq_ren, gamma_ren)
        print(f"[{variant}] soft-manifold Gamma median {soft_median:.3f} cm-1; Vogt omega_s {vogt_w:.2f}, Gamma {vogt_g:.2f}")
        sets = Sets(variant, surface_dir=args.surface_dir, xml_pattern=args.xml_pattern)
        for n in args.meshes:
            df = zone(sets, n, map_gamma, soft_median, vogt_w, vogt_g)
            tot = {k: float(df[f"eta_{k}"].sum()) for k in ("A", "B", "C", "B_rich", "C_rich")}
            rec = {"variant": variant, "mesh": n, "n_modes": len(df), **{f"eta_{k}_Pas": v for k, v in tot.items()}}
            for k in ("A", "B", "C"):
                col = f"eta_{k}"
                rec[f"top20_share_{k}"] = float(df[col].nlargest(20).sum() / tot[k])
                rec[f"shell1_frac_{k}"] = float(df[df.shell == 1][col].sum() / tot[k])
                rec[f"shell2_frac_{k}"] = float(df[df.shell == 2][col].sum() / tot[k])
                for sec in ("routeS", "routeH_stable", "routeH_unstable", "gamma_sector"):
                    rec[f"{sec}_{k}_Pas"] = float(df[df.sector == sec][col].sum())
            rec["frac_eta_A_overlap_lt_0p9"] = float(df[df.overlap_multiplet < 0.9].eta_A.sum() / tot["A"])
            rec["frac_eta_B_overlap_lt_0p9"] = float(df[df.overlap_multiplet < 0.9].eta_B.sum() / tot["B"])
            zone_rows.append(rec)
            print(f"[{variant}] mesh {n}^3: eta_A {tot['A']:.4e}  eta_B {tot['B']:.4e}  eta_C {tot['C']:.4e}  "
                  f"(Richardson B {tot['B_rich']:.4e}, C {tot['C_rich']:.4e}) Pa s", flush=True)
            if n == 11:
                with open(DIAG / f"{P}_modes_{variant}_mesh11.csv", "w") as fh:
                    fh.write(f"# per-mode table, variant {variant}, 11^3 mesh (eta_constructions.py); Lambda in cm^-2\n"
                             "# per unit engineering shear; eta_X in Pa s (already weighted 1/N_q).\n")
                    df.to_csv(fh, index=False, float_format="%.6g")
                for lab, g in df.groupby("omega0_bin"):
                    eA, eB, eC = g.eta_A.sum(), g.eta_B.sum(), g.eta_C.sum()
                    wts = g.eta_A.values
                    rBA = (g.Lambda_B ** 2 / g.Lambda_A ** 2).replace([np.inf, -np.inf], np.nan).values
                    rCB = (g.Lambda_C ** 2 / g.Lambda_B ** 2).replace([np.inf, -np.inf], np.nan).values

                    def wq(r, w, p):
                        m = np.isfinite(r)
                        if m.sum() == 0 or w[m].sum() == 0:
                            return np.nan
                        o = np.argsort(r[m]); cw = np.cumsum(w[m][o]) / w[m].sum()
                        return float(r[m][o][np.searchsorted(cw, p)])
                    bin_rows.append({"variant": variant, "omega0_bin": lab, "n_modes": len(g),
                                     "eta_A_Pas": eA, "eta_B_Pas": eB, "eta_C_Pas": eC,
                                     "eta_B_over_A": eB / eA if eA else np.nan, "eta_C_over_B": eC / eB if eB else np.nan,
                                     "ratio_B2_A2_p10": wq(rBA, wts, 0.1), "ratio_B2_A2_p50": wq(rBA, wts, 0.5),
                                     "ratio_B2_A2_p90": wq(rBA, wts, 0.9),
                                     "ratio_C2_B2_p10": wq(rCB, g.eta_B.values, 0.1), "ratio_C2_B2_p50": wq(rCB, g.eta_B.values, 0.5),
                                     "ratio_C2_B2_p90": wq(rCB, g.eta_B.values, 0.9)})
    hdr = ("# (eta_constructions.py): own-surface diagnostic model, 300 K; reference fixed (own SCPH omega_r,\n"
           "# freq-class linewidth map from the own 8^3 RTA of the same variant, stress kernel, Vogt Gamma anchor, no floor);\n"
           "# A = Route-H transfer, B = e_r K_HA e_r, C = e_r K_SCPH e_r (step h = 0.010; _rich = Richardson 0.010/0.020).\n"
           "# variant od0/od1 = SELF_OFFDIAG 0/1 (KMESH_INTERPOLATE 2x2x2, KMESH_SCPH 2x2x2, IFC supercell 2x2x2).\n")
    mode = "a" if (DIAG / f"{P}_zone.csv").exists() and len(args.variants) == 1 and args.variants[0] == "od0" else "w"
    with open(DIAG / f"{P}_zone.csv", mode) as fh:
        if mode == "w":
            fh.write(hdr)
        pd.DataFrame(zone_rows).to_csv(fh, index=False, float_format="%.6g", header=(mode == "w"))
    with open(DIAG / f"{P}_bins.csv", mode) as fh:
        if mode == "w":
            fh.write(hdr.replace("", "bins (11^3; eta-weighted percentiles of the per-mode ratios)"))
        pd.DataFrame(bin_rows).to_csv(fh, index=False, float_format="%.6g", header=(mode == "w"))
    print("written")


if __name__ == "__main__":
    main()
