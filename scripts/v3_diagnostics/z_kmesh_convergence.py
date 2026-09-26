#!/usr/bin/env python3
"""z_kmesh_convergence.py - (Z surface = tutorial harmonic set re-expressed on 4x4x4 + QE strain
perturbation; T&T protocol KMESH_INTERPOLATE 2), same row set as internal SCPH-mesh convergence of construction C (own surface,
SELF_OFFDIAG = 1, 300 K, 11^3 assembly) over the meshes present under data/raw/alamode_sto/own_od1/<tag>.

Per mesh: SCPH iterations / residual / wall time (runs.csv); iteration noise (last DIFF values, anphon internal
Ry units converted with 1 Ry = 109737.3 cm^-1; DIFF = rms frequency change over the interpolation grid) against
the strain signal rms|omega(+h) - omega(-h)| on 11^3; omega_r at Gamma TO1/TO2/LO1, X lowest, M lowest, R AFD/R5+
(AlamodeSet builder on the dfc2-renormalised XML, validated against anphon in ); imaginary counts on the
SCPH solution grid, 8^3 and 11^3 (threshold -0.5 cm^-1, three Gamma translations excluded); eta_C with (a) the
own RTA of the same surface and (b) the linewidths fixed at the 2/2 values; eta_B, eta_A (a); R-region share;
kappa; acoustic tau medians. Writes data/processed/v3_diagnostics/kmesh_convergence_9p1.csv (temp-then-rename); rows for meshes whose
files are incomplete are skipped, so the script can be rerun as runs finish.
Usage: uv run python kmesh_convergence_9p1.py [--out NAME]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from crosscheck_alamode_sto_tau import parse_result  # noqa: E402
from latvisc.coupling import AlamodeSet, diagonalise  # noqa: E402

OWN = REPO / "data" / "raw" / "alamode_sto" / "z_tut"
FC = OWN / "z_reference.fc"
RY_CM1 = 109737.31568
MESHES = {"i2s2": (2, 2), "i2s4": (2, 4), "i2s8": (2, 8), "i2s12": (2, 12), "i2s16": (2, 16)}
SETS = {"reference": "ref", "shear_xy_p005": "p005", "shear_xy_m005": "m005"}
R = (0.5, 0.5, 0.5)
# experiment (anchors of ) and Tadano-Tsuneyuki 2015 Table I (300 K, q1 = 12x12x12 / 2x2x2)
EXP = {"TO1": 88.96, "TO2": 175.0, "LO1": 171.0, "R_AFD": 52.0, "R5+": 145.0}
TT12 = {"TO1": 135.0, "R_AFD": 35.0, "M_lowest": 85.0}
TT2 = {"TO1": 144.0, "R_AFD": 69.0, "M_lowest": 103.0}
BANDS = [(0, 50), (50, 100), (100, 150)]
THR = -0.5


def grid(n):
    return [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)]


def n_imag(al, n):
    cnt = 0
    for q in grid(n):
        w, _ = diagonalise(al.dynmat(q, asr_onsite=True))
        w = np.sort(w)
        if all(abs(x) < 1e-9 for x in q):
            w = w[3:]
        cnt += int((w < THR).sum())
    return cnt


def qdist(q):
    d = (np.asarray(q) - np.array(R) + 0.5) % 1.0 - 0.5
    return np.linalg.norm(d, axis=-1)


def row_for(tag):
    d = OWN / tag
    kint, kscph = MESHES[tag]
    files = {s: d / f"renorm_z_{s}_{tag}_od1_300K.xml" for s in SETS}
    rta = d / f"STO_RTA_{tag}_od1_300K"
    if not files["reference"].exists() or not rta.with_suffix(".result").exists() or not rta.with_suffix(".kl").exists():
        return None
    pa = f"eta_z_{tag}"
    pb = pa if tag == "i2s2" else f"eta_z_{tag}_fixlw"
    za, zb = DIAG / f"{pa}_zone.csv", DIAG / f"{pb}_zone.csv"
    strained = all(f.exists() for f in files.values()) and za.exists() and zb.exists()
    rec = {"mesh": tag, "KMESH_INTERPOLATE": kint, "KMESH_SCPH": kscph, "nk_scph": kscph ** 3,
           "status": "strained + eta" if strained else "reference only (strained sets not feasible in memory)"}
    runs = pd.read_csv(d / "runs.csv")
    for _, r in runs[runs.stage == "scph"].iterrows():
        key = SETS.get(str(r["set"]).replace("z_", "", 1))
        if key is None:
            continue
        it = re.search(r"300K:(\d+)", str(r["iterations"]))
        rec[f"iter_{key}"] = int(it.group(1)) if it else -1
        rec[f"resid_{key}"] = float(r["final_DIFF"])
        rec[f"wall_s_{key}"] = float(r["wall_s"])
    # iteration noise: last three DIFF values of the reference log, in cm^-1
    log = d / f"scph_z_reference_{tag}_od1.log"
    diffs = [float(x) for x in re.findall(r"DIFF =\s+([\d.eE+-]+)", log.read_text())] if log.exists() else []
    rec["last3_DIFF_cm1"] = ";".join(f"{x * RY_CM1:.2e}" for x in diffs[-3:])
    rec["resid_ref_cm1"] = rec.get("resid_ref", np.nan) * RY_CM1
    # strain signal on 11^3 (sorted frequencies)
    if not strained:
        rec["strain_signal_rms_cm1"] = np.nan; rec["noise_over_signal"] = np.nan
    if strained:
        wp = np.load(d / f"mesh11_z_shear_xy_p005_{tag}_od1_300K.npz")["omega_cm1"]
        wm = np.load(d / f"mesh11_z_shear_xy_m005_{tag}_od1_300K.npz")["omega_cm1"]
        rec["strain_signal_rms_cm1"] = float(np.sqrt(np.mean((wp - wm) ** 2)))
        rec["strain_signal_median_abs_cm1"] = float(np.median(np.abs(wp - wm)))
        rec["noise_over_signal"] = rec["resid_ref_cm1"] / rec["strain_signal_rms_cm1"]
    # frequencies
    al = AlamodeSet(files["reference"], FC)
    wR, _ = diagonalise(al.dynmat(R, asr_onsite=True))
    wG, _ = diagonalise(al.dynmat((0, 0, 0), asr_onsite=True))
    wGx, _ = diagonalise(al.dynmat((0.001, 0, 0), asr_onsite=True))
    wX, _ = diagonalise(al.dynmat((0.5, 0, 0), asr_onsite=True))
    wM, _ = diagonalise(al.dynmat((0.5, 0.5, 0), asr_onsite=True))
    g = np.delete(wG, np.argsort(np.abs(wG))[:3]); gx = np.delete(wGx, np.argsort(np.abs(wGx))[:3])
    lo = [v for v in gx if np.abs(g - v).min() > 0.2]
    rec.update({"TO1_cm1": float(g[0]), "TO2_cm1": float(g[3]), "LO1_cm1": float(lo[0]) if lo else np.nan,
                "X_lowest_cm1": float(np.sort(wX)[0]), "M_lowest_cm1": float(np.sort(wM)[0]),
                "R_AFD_cm1": float(np.sort(wR)[0]), "R_second_cm1": float(np.sort(wR)[1]), "R5+_cm1": float(np.sort(wR)[3])})
    for k, v in EXP.items():
        rec[f"{k}_vs_exp_pct"] = 100.0 * (rec[f"{k}_cm1"] / v - 1.0)
    for k, v in TT12.items():
        rec[f"{k}_TT2015_12cubed_cm1"] = v
        rec[f"{k}_vs_TT12_pct"] = 100.0 * (rec[f"{k}_cm1"] / v - 1.0)
    for k, v in TT2.items():
        rec[f"{k}_TT2015_2cubed_cm1"] = v
    rec["n_imag_solution_grid"] = n_imag(al, kscph)
    rec["n_imag_8cubed"] = n_imag(al, 8)
    w11 = np.load(d / f"mesh11_z_reference_{tag}_od1_300K.npz")["omega_cm1"]
    rec["n_imag_11cubed"] = int((w11[1:] < THR).sum() + (np.sort(w11[0])[3:] < THR).sum())
    for s in ("shear_xy_p005", "shear_xy_m005"):
        if strained:
            w = np.load(d / f"mesh11_z_{s}_{tag}_od1_300K.npz")["omega_cm1"]
            rec[f"n_imag_11cubed_{SETS[s]}"] = int((w[1:] < THR).sum() + (np.sort(w[0])[3:] < THR).sum())
    # eta
    for lab, zf in ((("a", za), ("b", zb)) if strained else ()):
        z = pd.read_csv(zf, comment="#")
        z = z[(z.variant == "od1") & (z.mesh == 11)].iloc[0]
        rec[f"eta_C_{lab}_Pas"] = float(z.eta_C_Pas)
        if lab == "a":
            rec["eta_B_a_Pas"] = float(z.eta_B_Pas); rec["eta_A_a_Pas"] = float(z.eta_A_Pas)
            rec["top20_share_C_a"] = float(z.top20_share_C)
    if strained:
        rec["eta_C_b_over_a"] = rec["eta_C_b_Pas"] / rec["eta_C_a_Pas"]
        m = pd.read_csv(DIAG / f"{pa}_modes_od1_mesh11.csv", comment="#")
        near = qdist(m[["qa", "qb", "qc"]].values) <= 0.15
        rec["frac_eta_C_Rregion_R4branch"] = float(m[near & (m.omega0_partner_cm1 < 0)].eta_C.sum() / rec["eta_C_a_Pas"])
        rec["frac_eta_C_Rshell"] = float(m[near].eta_C.sum() / rec["eta_C_a_Pas"])
        rec["eta_weighted_tau_ps_a"] = float((m.eta_C * m.tau_ps).sum() / m.eta_C.sum())
        if "Lambda_C_h020" in m and (m.Lambda_C_h020 != m.Lambda_C).any():
            ratio = m.Lambda_C_h020 / m.Lambda_C.replace(0, np.nan)
            asym = (np.abs(ratio - 1) < 0.25) & (ratio > 0)
            rec["eta_C_h020_Pas"] = float((m.pref_eta_per_Lambda2 * m.Lambda_C_h020 ** 2).sum())
            rec["eta_C_rich_Pas"] = float(m.eta_C_rich.sum())
            rec["asymptotic_class_frac_modes"] = float(asym.mean())
            rec["asymptotic_class_frac_eta_C"] = float(m.eta_C[asym].sum() / m.eta_C.sum())
            ra = m[asym].eta_C_rich.sum() + m[~asym].eta_C.sum()
            rec["eta_C_rich_asymptotic_only_Pas"] = float(ra)
    else:
        for k in ("eta_C_a_Pas", "eta_C_b_Pas", "eta_B_a_Pas", "eta_A_a_Pas"):
            rec[k] = np.nan
    kl = [ln for ln in rta.with_suffix(".kl").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    rec["kappa_300K_W_mK"] = float(kl[-1].split()[1])
    rec["kappa_vs_calc_8p7_pct"] = 100.0 * (rec["kappa_300K_W_mK"] / 8.7 - 1.0)
    rec["kappa_vs_meas_11p0_pct"] = 100.0 * (rec["kappa_300K_W_mK"] / 11.0 - 1.0)
    freq, gam = parse_result(rta.with_suffix(".result"), target_temp=300)
    for lo_, hi_ in BANDS:
        taus = [1.0 / (2.0 * gam[k] * 2.0 * np.pi * 2.99792458e10) * 1e12 for k, w in freq.items()
                if lo_ <= w < hi_ and gam.get(k, 0) > 0]
        rec[f"tau_med_ps_{lo_}_{hi_}"] = float(np.median(taus)) if taus else np.nan
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="z_kmesh_convergence.csv")
    args = ap.parse_args()
    rows = []
    for tag in MESHES:
        rec = row_for(tag)
        if rec is None:
            print(f"[{tag}] incomplete, skipped")
            continue
        rows.append(rec)
        print(f"[{tag}] {rec['status']}: eta_C(a) {rec['eta_C_a_Pas']:.4e}  (b) {rec['eta_C_b_Pas']:.4e}  R_AFD {rec['R_AFD_cm1']:.1f}  "
              f"TO1 {rec['TO1_cm1']:.1f}  imag sol/8/11 {rec['n_imag_solution_grid']}/{rec['n_imag_8cubed']}/{rec['n_imag_11cubed']}  "
              f"kappa {rec['kappa_300K_W_mK']:.2f}  noise/signal {rec['noise_over_signal']:.1e}", flush=True)
    df = pd.DataFrame(rows)
    # convergence deltas within each fixed-interpolation chain (ordered by KMESH_SCPH)
    for lab in ("a", "b"):
        df[f"delta_prev_{lab}_pct"] = np.nan
    for kint, g in df.groupby("KMESH_INTERPOLATE"):
        g = g.sort_values("KMESH_SCPH")
        prev = None
        for idx, r in g.iterrows():
            if prev is not None and np.isfinite(r["eta_C_a_Pas"]) and np.isfinite(prev["eta_C_a_Pas"]):
                for lab in ("a", "b"):
                    df.loc[idx, f"delta_prev_{lab}_pct"] = 100.0 * abs(r[f"eta_C_{lab}_Pas"] - prev[f"eta_C_{lab}_Pas"]) / abs(r[f"eta_C_{lab}_Pas"])
            if np.isfinite(r["eta_C_a_Pas"]):
                prev = r
    out = DIAG / args.out
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# (z_kmesh_convergence.py): internal SCPH-mesh dependence of construction C on the Z surface (tutorial harmonic set\n# re-expressed on 4x4x4 + QE strain perturbation, T&T protocol),\n"
                 "# SELF_OFFDIAG = 1, 300 K, 11^3 assembly. eta_C_a = own RTA linewidths of the same surface (production rule);\n"
                 "# eta_C_b = linewidths and frequency-class map fixed at the 2/2-surface values (isolates frequencies + couplings).\n"
                 "# resid = anphon DIFF (rms frequency change per iteration over the interpolation grid, Ry; *_cm1 converted);\n"
                 "# strain signal = rms |omega(+h) - omega(-h)| over 11^3 (h = 0.010). Imaginary counts: threshold -0.5 cm^-1,\n"
                 "# three Gamma translations excluded. TT2015 = Tadano & Tsuneyuki, PRB 92, 054301, Table I (300 K).\n"
                 "# delta_prev = |eta_C(n) - eta_C(n_prev)| / |eta_C(n)| within the chain of fixed KMESH_INTERPOLATE.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 100)
    print(df[["mesh", "eta_C_a_Pas", "eta_C_b_Pas", "delta_prev_a_pct", "delta_prev_b_pct", "R_AFD_cm1", "TO1_cm1",
              "n_imag_solution_grid", "n_imag_11cubed", "kappa_300K_W_mK", "noise_over_signal"]].to_string())


if __name__ == "__main__":
    main()
