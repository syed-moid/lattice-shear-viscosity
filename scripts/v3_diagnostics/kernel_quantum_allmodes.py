#!/usr/bin/env python3
"""Quantum correction of the two lifetime kernels for every production mode at 300 K.

For each mode of the production table (hybrid model, construction C; frequency omega_used and linewidth Gamma as used in
the assembly) the frequency-domain two-pole integrals with the Bose factors inside the integral (the method of
overdamped_kernel_check.py, manuscript Eq. A7) are compared with the classical kernels multiplied by n(n+1):
    r_E = I_E / [n(n+1) tau_E],   tau_E = 1/(2 Gamma) + Gamma/(2 omega^2)   (energy variable)
    r_S = I_S / [n(n+1) tau_S],   tau_S = 1/(2 Gamma) + 2 Gamma/omega^2     (stress correlator, production)
The relative correction of a mode is r - 1; the eta-weighted correction of the total is sum_k eta_k (r_S,k - 1) / eta.

Output: data/processed/v3_diagnostics/kernel_quantum_allmodes.csv (per distinct (omega, Gamma) pair, with the number of modes
and their summed eta contribution) and kernel_quantum_allmodes_summary.txt.
Usage: uv run python scripts/v3_diagnostics/kernel_quantum_allmodes.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG  # noqa: E402  (sets sys.path)

from overdamped_kernel_check import CM1, HBAR, K_B, quantum_integrals  # noqa: E402
from latvisc.viscosity import tau_two_pole_exact, tau_two_pole_stress  # noqa: E402

T_K = 300.0


def one(args):
    w_cm1, g_cm1 = args
    w, g = w_cm1 * CM1, g_cm1 * CM1
    n = 1.0 / np.expm1(HBAR * w / (K_B * T_K))
    wgt = n * (n + 1.0)
    ie, is_ = quantum_integrals(w, g, T_K)
    return ie / (wgt * float(tau_two_pole_exact(w, g))), is_ / (wgt * float(tau_two_pole_stress(w, g)))


def main():
    mt = pd.read_csv(DIAG / "mode_table_SrTiO3_300K_v6.csv", comment="#")
    mt = mt[mt["eta_contrib_Pas"] > 0].copy()
    eta = mt["eta_contrib_Pas"].sum()
    mt["key_w"] = mt["omega_used_cm1"].round(6)
    mt["key_g"] = mt["Gamma_hwhm_cm1"].round(6)
    grp = mt.groupby(["key_w", "key_g"]).agg(n_modes=("eta_contrib_Pas", "size"), eta_sum=("eta_contrib_Pas", "sum")).reset_index()
    pairs = list(zip(grp["key_w"], grp["key_g"]))
    print(f"{len(mt)} modes with eta > 0, {len(pairs)} distinct (omega, Gamma) pairs", flush=True)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, pairs, chunksize=32))
    grp["r_energy"] = [r[0] for r in res]
    grp["r_stress"] = [r[1] for r in res]
    grp["corr_energy"] = grp["r_energy"] - 1.0
    grp["corr_stress"] = grp["r_stress"] - 1.0
    grp["hbar_omega_over_kT"] = HBAR * grp["key_w"] * CM1 / (K_B * T_K)
    grp["Gamma_over_omega"] = grp["key_g"] / grp["key_w"]
    grp = grp.rename(columns={"key_w": "omega_cm1", "key_g": "Gamma_hwhm_cm1"})
    wts = grp["eta_sum"] / eta
    lines = [
        f"modes with eta > 0: {len(mt)}; distinct (omega, Gamma): {len(grp)}; eta = {eta:.6e} Pa s (production, 300 K)",
        f"hbar omega / k_B T: {grp['hbar_omega_over_kT'].min():.4f} - {grp['hbar_omega_over_kT'].max():.4f}; "
        f"Gamma/omega: {grp['Gamma_over_omega'].min():.5f} - {grp['Gamma_over_omega'].max():.4f}",
        f"stress kernel (production): max |r - 1| = {grp['corr_stress'].abs().max():.4%}; eta-weighted mean r - 1 = "
        f"{(wts * grp['corr_stress']).sum():.4%} (= relative change of eta if the quantum integral replaced the kernel)",
        f"energy kernel: max |r - 1| = {grp['corr_energy'].abs().max():.4%}; eta-weighted mean r - 1 = {(wts * grp['corr_energy']).sum():.4%}",
        f"largest stress-kernel corrections at: " + "; ".join(
            f"omega {r.omega_cm1:.1f} cm-1, Gamma/omega {r.Gamma_over_omega:.3f}, hw/kT {r.hbar_omega_over_kT:.2f}: {r.corr_stress:+.4%}"
            for r in grp.reindex(grp["corr_stress"].abs().sort_values(ascending=False).index).head(3).itertuples()),
    ]
    out = DIAG / "kernel_quantum_allmodes.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# kernel_quantum_allmodes.py: frequency-domain two-pole integrals with Bose factors inside vs the classical kernels x n(n+1),\n"
                 "# every production mode at 300 K (grouped by distinct (omega, Gamma)); r = ratio, corr = r - 1; eta_sum = summed eta\n"
                 "# contribution (Pa s) of the modes in the group.\n")
        grp.to_csv(fh, index=False, float_format="%.8g")
    tmp.rename(out)
    txt = DIAG / "kernel_quantum_allmodes_summary.txt"
    tmp = txt.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(txt)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
