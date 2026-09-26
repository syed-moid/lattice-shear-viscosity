#!/usr/bin/env python3
"""loop_weight_estimate_9p5.py - (ESTIMATE, not the vertex contraction): distribution of the SCPH loop
weight (2 n' + 1) / (2 omega') over the inner mesh q' for each converged surface, using that surface's own
renormalised frequencies (AlamodeSet builder on the dfc2 XML, asr_onsite) on its KMESH_SCPH grid at 300 K.
The loop shift of a mode is sum_{q' nu'} F(q nu; q' nu') (2n'+1)/(2 omega'); F is NOT computed here, so the
shares below are shares of the WEIGHT SUM ONLY (vertex assumed uniform) - an estimate to be labelled as such.
Shares reported: Gamma-point optical modes (12 slots), the near-Gamma shell |q'| <= 0.15 (optical, all),
the near-R shell |q' - R| <= 0.15, the rest; plus the lowest omega' on the grid and the weight of the ten
softest slots. Writes data/processed/v3_diagnostics/loop_weight_estimate_9p5.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402

OWN = REPO / "data" / "raw" / "alamode_sto" / "own_od1"
FC = REPO / "dft" / "qe" / "SrTiO3" / "dispersion_pbesol" / "SrTiO3_pbesol.444.fc"
MESHES = {"i2s2": (2, 2), "i2s4": (2, 4), "i2s8": (2, 8), "i2s12": (2, 12), "i2s16": (2, 16),
          "i3s12": (3, 12), "i4s4": (4, 4), "i4s8": (4, 8), "i4s12": (4, 12)}
KT_CM1 = 208.51  # k_B * 300 K in cm^-1
R = np.array([0.5, 0.5, 0.5])


def weight(w):
    """(2n+1)/(2 omega) with omega in cm^-1 (hbar omega / k_B T dimensionless), imaginary -> |omega| (anphon rule)."""
    w = np.abs(w)
    x = w / KT_CM1
    return (1.0 / np.tanh(0.5 * x)) / (2.0 * w)


def main():
    rows = []
    for tag, (kint, ks) in MESHES.items():
        xml = OWN / tag / f"renorm_reference_{tag}_od1_300K.xml"
        if not xml.exists():
            print(f"[{tag}] missing"); continue
        al = AlamodeSet(xml, FC)
        qs = np.array([(i / ks, j / ks, k / ks) for i in range(ks) for j in range(ks) for k in range(ks)])
        W = np.zeros((len(qs), 15)); OM = np.zeros((len(qs), 15))
        for iq, q in enumerate(qs):
            w, _ = diagonalise(al.dynmat(q, asr_onsite=True))
            w = np.sort(w)
            OM[iq] = w
            ww = weight(np.where(np.abs(w) < 1e-6, 1e-6, w))
            if iq == 0:
                ww[:3] = 0.0     # acoustic zeros at Gamma excluded
            W[iq] = ww
        tot = W.sum()
        dG = np.linalg.norm((qs + 0.5) % 1.0 - 0.5, axis=1)
        dR = np.linalg.norm((qs - R + 0.5) % 1.0 - 0.5, axis=1)
        flat = W.ravel(); order = np.argsort(-flat)
        rec = {"mesh": tag, "KMESH_INTERPOLATE": kint, "KMESH_SCPH": ks, "n_slots": 15 * len(qs) - 3,
               "share_Gamma_optical": W[0].sum() / tot, "share_Gamma_TO1_triplet": W[0][3:6].sum() / tot,
               "share_nearGamma_shell_0p15": W[dG <= 0.15].sum() / tot,
               "share_nearR_shell_0p15": W[dR <= 0.15].sum() / tot,
               "share_R_point": W[dR < 1e-6].sum() / tot, "share_R_AFD_triplet": W[dR < 1e-6][0][:3].sum() / tot,
               "share_ten_softest_slots": flat[order[:10]].sum() / tot,
               "min_omega_grid_cm1": float(OM[1:].min()) if len(qs) > 1 else float(OM[0][3:].min()),
               "R_AFD_cm1": float(OM[dR < 1e-6][0][0]), "Gamma_TO1_cm1": float(OM[0][3]),
               "mean_weight_per_slot_cm1inv": tot / (15 * len(qs) - 3), "uniform_share_Gamma_optical": 12.0 / (15 * len(qs) - 3)}
        rows.append(rec)
        print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    df = pd.DataFrame(rows)
    out = DIAG / "loop_weight_estimate_9p5.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# ESTIMATE (loop_weight_estimate_9p5.py): shares of the SCPH loop WEIGHT SUM (2n'+1)/(2 omega') over the\n"
                 "# KMESH_SCPH grid at 300 K, own renormalised frequencies of each surface; the quartic vertex F(q nu; q' nu') is NOT\n"
                 "# included (uniform-vertex assumption). uniform_share_Gamma_optical = 12 / n_slots is the share Gamma would have\n"
                 "# with equal weights; imaginary omega' enter as |omega'| (anphon rule); Gamma acoustic zeros excluded.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)


if __name__ == "__main__":
    main()
