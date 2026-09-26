#!/usr/bin/env python3
"""k5b_indicator_11p3.py - (K5b): sensitivity to the reference eigenbasis. The SAME strain perturbation K_QE
(bare QE strained pair +-0.005, engineering shear h = 0.010, QE/matdyn convention) projected on the bare QE eigenbasis
e_QE (own harmonic set) and on the bare tutorial eigenbasis e_tut (Z reference, re-expressed on 4x4x4) on the 11^3 mesh.
Multiplet couplings by projected-block diagonalisation (latvisc.coupling.project_coupling); the QE and tutorial bare modes
at each q are paired by maximum multiplet overlap; weights = production eta contributions (mode table v5) attributed to
the bare QE partner of each renormalised mode. Writes data/processed/v3_diagnostics/k5b_indicator_11p3.csv (totals + per bare-omega_0 bin).
Label: sensitivity to the reference eigenbasis; does NOT bound K5a (the untested K_tut - K_QE)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from coupling_kernel import FC, hermitian  # noqa: E402
from latvisc.coupling import QESet, diagonalise, multiplet_groups, project_coupling  # noqa: E402

Z = REPO / "data" / "raw" / "alamode_sto" / "z_tut"
H = 0.010
BINS = [(-np.inf, 0), (0, 25), (25, 50), (50, 100), (100, 150), (150, 175), (175, 300), (300, np.inf)]


def bin_label(w):
    for lo, hi in BINS:
        if lo <= w < hi:
            return "imaginary" if not np.isfinite(lo) else f"[{lo:.0f},{hi if np.isfinite(hi) else 'inf'})"
    return "?"


def pair_by_overlap(w_a, v_a, w_b, v_b, tol=0.5):
    """for each multiplet of a, the multiplet of b with the largest summed overlap; returns per-mode index map a->b and overlaps."""
    ga, gb = multiplet_groups(w_a, tol), multiplet_groups(w_b, tol)
    Pb = [v_b[:, g] @ v_b[:, g].conj().T for g in gb]
    idx = np.zeros(len(w_a), dtype=int); ov = np.zeros(len(w_a))
    for g in ga:
        Pa = v_a[:, g] @ v_a[:, g].conj().T
        scores = [np.real(np.trace(Pa @ P)) / len(g) for P in Pb]
        j = int(np.argmax(scores))
        for m in g:
            idx[m] = gb[j][0]; ov[m] = scores[j]
    return idx, ov


def main():
    t = pd.read_csv(DIAG / "mode_table_SrTiO3_300K_v5.csv", comment="#")
    eta_by_partner = t.groupby(["iq", "partner_mu"]).eta_contrib_Pas.sum().to_dict()   # (iq, mu 1-based) -> eta
    E = t.eta_contrib_Pas.sum()
    qe0, qep, qem = QESet(FC["reference"]), QESet(FC["shear_xy_p005"]), QESet(FC["shear_xy_m005"])
    z0 = QESet(Z / "z_reference.fc")
    n = 11
    qs = [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)]
    rows = []
    for iq, q in enumerate(qs):
        K = hermitian((qep.dynmat(q) - qem.dynmat(q)) / (2 * H))
        wq, vq = diagonalise(qe0.dynmat(q)); oq = np.argsort(wq); wq, vq = wq[oq], vq[:, oq]
        wt, vt = diagonalise(z0.dynmat(q)); ot = np.argsort(wt); wt, vt = wt[ot], vt[:, ot]
        pq, pt = project_coupling(K, vq, wq), project_coupling(K, vt, wt)
        idx, ov = pair_by_overlap(wq, vq, wt, vt)
        for mu in range(15):
            if iq == 0 and abs(wq[mu]) < 0.5:
                continue
            eta_w = eta_by_partner.get((iq, mu + 1), 0.0)
            rows.append({"iq": iq, "mu_QE": mu + 1, "omega0_QE": wq[mu], "omega0_tut": wt[idx[mu]], "overlap": ov[mu],
                         "trK2_QE": pq["trk2"][mu] / pq["group_size"][mu], "trK2_tut": pt["trk2"][idx[mu]] / pt["group_size"][idx[mu]],
                         "eta_weight": eta_w, "bin": bin_label(wq[mu])})
        if iq % 200 == 0:
            print(f"  {iq}/{len(qs)}", flush=True)
    df = pd.DataFrame(rows)
    w = df.eta_weight.values
    tot = {"scope": "all modes (eta-weighted)", "n_modes": len(df),
           "ratio_sum_eta_trK2_tut_over_QE": float(np.sum(w * df.trK2_tut) / np.sum(w * df.trK2_QE)),
           "ratio_plain_sum_trK2_tut_over_QE": float(df.trK2_tut.sum() / df.trK2_QE.sum()),
           "eta_weighted_mean_abs_domega0_cm1": float(np.sum(w * np.abs(df.omega0_tut - df.omega0_QE)) / w.sum()),
           "frac_eta_overlap_lt_0p9": float(w[df.overlap < 0.9].sum() / w.sum()),
           "frac_eta_overlap_lt_0p99": float(w[df.overlap < 0.99].sum() / w.sum()),
           "eta_covered_frac_of_production": float(w.sum() / E)}
    out = [tot]
    for lab, g in df.groupby("bin"):
        wg = g.eta_weight.values
        r = (g.trK2_tut / g.trK2_QE).replace([np.inf, -np.inf], np.nan).values
        m = np.isfinite(r) & (wg > 0)
        med = np.nan
        if m.sum():
            o = np.argsort(r[m]); cw = np.cumsum(wg[m][o]) / wg[m].sum(); med = float(r[m][o][np.searchsorted(cw, 0.5)])
        out.append({"scope": f"bin {lab}", "n_modes": len(g), "eta_share": float(wg.sum() / w.sum()),
                    "ratio_sum_eta_trK2_tut_over_QE": float(np.sum(wg * g.trK2_tut) / np.sum(wg * g.trK2_QE)) if np.sum(wg * g.trK2_QE) else np.nan,
                    "eta_weighted_median_ratio": med,
                    "eta_weighted_mean_abs_domega0_cm1": float(np.sum(wg * np.abs(g.omega0_tut - g.omega0_QE)) / wg.sum()) if wg.sum() else np.nan,
                    "frac_eta_overlap_lt_0p9": float(wg[g.overlap.values < 0.9].sum() / wg.sum()) if wg.sum() else np.nan})
    res = pd.DataFrame(out)
    o = DIAG / "k5b_indicator_11p3.csv"; tmp = o.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# (k5b_indicator_11p3.py): the SAME K_QE (bare QE +-0.005 pair, h = 0.010) projected on e_QE (own bare set) and on e_tut\n"
                 "# (Z reference = tutorial set re-expressed on 4x4x4), 11^3; trK2 = Tr K_sub^2 / multiplet size (gauge invariant); QE<->tutorial\n"
                 "# bare modes paired by maximum multiplet overlap; weights = production eta (mode table v5) by bare QE partner.\n"
                 "# LABEL: sensitivity to the reference eigenbasis; does NOT bound K5a.\n")
        res.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(o)
    df.to_csv(DIAG / "k5b_indicator_11p3_modes.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250); print(res.to_string())


if __name__ == "__main__":
    main()
