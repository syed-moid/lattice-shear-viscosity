#!/usr/bin/env python3
"""Symmetry check of the shear-strain perturbation K(q) = [D(q, +h) - D(q, -h)] / (2h) of the hybrid model.

A pure xy shear is invariant (up to sign) under the 16 operations R of O_h that map the xy plane onto itself and z onto
+-z: R eps_xy R^T = chi(R) eps_xy with chi = +-1. Each such R maps the +h crystal onto the chi*h crystal (every atom sits on
an inversion centre, internal coordinates fixed), and maps the reduced wave vector q of the sheared cell onto R q. Hence
the exact derivative obeys K(Rq) = chi(R) U(R,q) K(q) U(R,q)^dagger, and the symmetric part is

    K_sym(q) = (1/16) sum_R chi(R) U(R,q)^dagger K(Rq) U(R,q).

U(R,q) (15x15, mass-weighted displacements) = atom permutation x Cartesian rotation x lattice-translation phase; it is
verified on the unstrained cubic sets by D(Rq) = U D(q) U^dagger. The component K - K_sym is forbidden by symmetry and is a
numerical error of the strained inputs.

Outputs (data/processed/v3_diagnostics/):
  strain_symmetry_check.csv        per-q norms, forbidden acoustic couplings on the principal axes, block eigenvalues,
                                   eta_B and eta_C with K and with K_sym (production hybrid model, 300 K, 11^3)
  strain_symmetry_summary.txt      summary numbers
Usage: uv run python scripts/v3_diagnostics/strain_symmetry_check.py
Needs: data/raw/alamode_sto/z_tut (hybrid model, raw archive); dft/qe/SrTiO3 (QE strained pairs).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG, REPO  # noqa: E402,F401  (sets sys.path)

import compute_eta_SrTiO3 as E  # noqa: E402
from latvisc.coupling import (QESet, acoustic_character, diagonalise, multiplet_groups, project_coupling,  # noqa: E402
                              strain_derivative_matrix, transfer_bare_coupling)

H = E.H_ENG
T_K = 300
MESH_N = 11
NAT = 5
X_FRAC = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])   # QE atom order (Sr, Ti, O1, O2, O3)
EPS_XY = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
TOL = E.COUPLING_DEGEN_TOL      # production coupling rule: block treatment only inside exactly degenerate eigenspaces


# --------------------------------------------------------------------------- symmetry operations
def oh_operations():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = signs[i]
            ops.append(R)
    return ops


def shear_group():
    """the 16 operations with R eps_xy R^T = chi eps_xy, and chi."""
    out = []
    for R in oh_operations():
        M = R @ EPS_XY @ R.T
        for chi in (1, -1):
            if np.allclose(M, chi * EPS_XY):
                out.append((R, chi))
    assert len(out) == 16
    return out


def atom_map(R):
    """R x_a = x_b + L_a (fractional, cubic cell): returns perm[a] = b and L (5x3 integer)."""
    perm, L = np.zeros(NAT, dtype=int), np.zeros((NAT, 3))
    for a in range(NAT):
        y = R @ X_FRAC[a]
        for b in range(NAT):
            d = y - X_FRAC[b]
            if np.allclose(d, np.round(d), atol=1e-9):
                perm[a], L[a] = b, np.round(d)
                break
        else:
            raise RuntimeError("atom not mapped")
    return perm, L


def u_matrix(R, q, sign):
    perm, L = atom_map(R)
    Rq = R @ np.asarray(q, dtype=float)
    U = np.zeros((3 * NAT, 3 * NAT), dtype=complex)
    for a in range(NAT):
        b = perm[a]
        U[3 * b:3 * b + 3, 3 * a:3 * a + 3] = R * np.exp(sign * 2j * np.pi * (Rq @ L[a]))
    return U


def verify_u(dyn, group, sign, qs):
    worst = 0.0
    for q in qs:
        D = dyn(q)
        for R, _ in group:
            U = u_matrix(R, q, sign)
            worst = max(worst, np.linalg.norm(dyn(R @ q) - U @ D @ U.conj().T) / np.linalg.norm(D))
    return worst


def pick_sign(dyn, group, label):
    qs = [np.array([0.13, 0.29, 0.41]), np.array([0.37, -0.11, 0.23])]
    res = {s: verify_u(dyn, group, s, qs) for s in (+1, -1)}
    s = min(res, key=res.get)
    print(f"  U verification ({label}): max ||D(Rq) - U D U^+|| / ||D|| = {res[s]:.2e} (phase sign {s:+d}; other sign {res[-s]:.1e})")
    return s, res[s]


def symmetrise(Kfun, q, group, sign):
    acc = np.zeros((3 * NAT, 3 * NAT), dtype=complex)
    for R, chi in group:
        U = u_matrix(R, q, sign)
        acc += chi * U.conj().T @ Kfun(R @ np.asarray(q, dtype=float)) @ U
    return acc / len(group)


def little_group(q, group):
    """operations of the shear group with R q = q (mod G) and their chi."""
    return [(R, chi) for R, chi in group if np.allclose(((R @ q - q) + 0.5) % 1.0 - 0.5, 0.0, atol=1e-9)]


# --------------------------------------------------------------------------- sets
def load_sets():
    surf = E.load_construction_surface(T_K)
    S = surf["sets"]
    zb = {t: E.qe_set(t, E.PROD) for t in E.STRAIN_SETS}                   # production harmonic hybrid sets (for A, B)
    zq = {t: QESet(E.Z_FC[t]) for t in E.STRAIN_SETS}                      # same sets, QE rigid-ion convention (comparison)
    qe = {(t, a): QESet(E.QE_FC[t], asr=a) for t in E.STRAIN_SETS for a in ("none", "distributed")}
    K = {
        "K_HA (production, B)": lambda q: strain_derivative_matrix(zb["shear_xy_p005"].dynmat(q), zb["shear_xy_m005"].dynmat(q), H),
        "K_HA (QE rigid-ion convention)": lambda q: strain_derivative_matrix(zq["shear_xy_p005"].dynmat(q), zq["shear_xy_m005"].dynmat(q), H),
        "K_SCPH (production, C)": lambda q: strain_derivative_matrix(S["shear_xy_p005"].dynmat(q, asr_onsite=True),
                                                                    S["shear_xy_m005"].dynmat(q, asr_onsite=True), H),
        "K_QE raw pair, no sum rule": lambda q: strain_derivative_matrix(qe[("shear_xy_p005", "none")].dynmat(q),
                                                                        qe[("shear_xy_m005", "none")].dynmat(q), H),
        "K_QE pair, separate distributed sum rule": lambda q: strain_derivative_matrix(qe[("shear_xy_p005", "distributed")].dynmat(q),
                                                                                      qe[("shear_xy_m005", "distributed")].dynmat(q), H),
    }
    ref = {"bare": lambda q: zb["reference"].dynmat(q), "scph": lambda q: S["reference"].dynmat(q, asr_onsite=True),
           "qe": lambda q: qe[("reference", "none")].dynmat(q), "bare_qe": lambda q: zq["reference"].dynmat(q)}
    return surf, S, zb, K, ref


def main():
    group = shear_group()
    print(f"shear group: {len(group)} operations, chi = +1 for {sum(c > 0 for _, c in group)}, -1 for {sum(c < 0 for _, c in group)}")
    surf, S, zb, Ks, ref = load_sets()
    sign_bare, err_bare = pick_sign(ref["bare"], group, "bare hybrid reference")
    sign_scph, err_scph = pick_sign(ref["scph"], group, "SCPH hybrid reference, 300 K, 2/12")
    sign_qe, err_qe = pick_sign(ref["qe"], group, "QE reference")
    sign_bare_qe = pick_sign(ref["bare_qe"], group, "bare hybrid reference, QE convention")[0]
    sign_of = {"K_HA (production, B)": sign_bare, "K_SCPH (production, C)": sign_scph, "K_HA (QE rigid-ion convention)": sign_bare_qe,
               "K_QE raw pair, no sum rule": sign_qe, "K_QE pair, separate distributed sum rule": sign_qe}
    rows = []
    summary = [f"U verification (max relative residual of D(Rq) = U D(q) U^+ over the 16 operations, two generic q): "
               f"bare hybrid {err_bare:.1e}; SCPH hybrid {err_scph:.1e}; QE {err_qe:.1e}"]

    # (b1) per-q norm of the forbidden part on the 11^3 mesh (the 11^3 mesh is closed under R)
    n = MESH_N
    qmesh = np.array(E.mesh_points(n))
    index = {tuple(np.round(q * n).astype(int) % n): i for i, q in enumerate(qmesh)}
    Kmesh = {}
    for label in ("K_HA (production, B)", "K_SCPH (production, C)"):
        Kmesh[label] = np.array([Ks[label](q) for q in qmesh])
        print(f"  {label}: K on {len(qmesh)} mesh points", flush=True)
    Ksym_mesh = {}
    for label, arr in Kmesh.items():
        s = sign_of[label]
        out = np.zeros_like(arr)
        for i, q in enumerate(qmesh):
            acc = np.zeros((3 * NAT, 3 * NAT), dtype=complex)
            for R, chi in group:
                j = index[tuple(np.round(R @ q * n).astype(int) % n)]
                U = u_matrix(R, q, s)
                acc += chi * U.conj().T @ arr[j] @ U
            out[i] = acc / len(group)
        Ksym_mesh[label] = out
        rel = np.array([np.linalg.norm(arr[i] - out[i]) / max(np.linalg.norm(arr[i]), 1e-30) for i in range(len(qmesh))])
        qs = np.percentile(rel, [50, 90, 99, 100])
        summary.append(f"{label}: ||K - K_sym|| / ||K|| on the 11^3 mesh: median {qs[0]:.2e}, 90 % {qs[1]:.2e}, "
                       f"99 % {qs[2]:.2e}, max {qs[3]:.2e}")
        rows.append({"block": "mesh_norm", "K": label, "median": qs[0], "p90": qs[1], "p99": qs[2], "max": qs[3]})

    # (b2) forbidden and allowed acoustic couplings along the principal lines, raw vs symmetrised
    lines = {"[100] Gamma-X (reduced (q,0,0))": (1, 0, 0), "[010]": (0, 1, 0), "[001]": (0, 0, 1),
             "[110] Gamma-M": (1, 1, 0), "[1-10]": (1, -1, 0), "[111] Gamma-R": (1, 1, 1)}
    masses = zb["reference"].masses_amu
    for lname, d in lines.items():
        dvec = np.array(d, float)
        for qm in (0.1, 0.05, 0.025, 0.0125):
            q = qm * dvec
            wr, Er = diagonalise(S["reference"].dynmat(q, asr_onsite=True))
            labels, _, _ = acoustic_character(Er, masses, dvec)
            lg = little_group(q, group)
            chis = sorted(c for _, c in lg)
            lg_label = f"order {len(lg)}, chi=-1 ops {sum(c < 0 for c in chis)}"
            for klabel in Ks:
                s = sign_of[klabel]
                K = Ks[klabel](q)
                Ksym = symmetrise(Ks[klabel], q, group, s)
                p_raw = project_coupling(K, Er, wr, TOL)
                p_sym = project_coupling(Ksym, Er, wr, TOL)
                groups = multiplet_groups(wr, TOL)
                acoustic = [g for g in groups if min(g) < 3]
                for g in acoustic:
                    blk_raw = p_raw["lam"][g]
                    blk_sym = p_sym["lam"][g]
                    rows.append({"block": "acoustic_line", "K": klabel, "line": lname, "q_mag": qm, "little_group": lg_label,
                                 "modes": "+".join(labels[i] for i in g), "omega_r_cm1": " ".join(f"{wr[i]:.3f}" for i in g),
                                 "lam_raw": " ".join(f"{v:.3f}" for v in blk_raw), "lam_sym": " ".join(f"{v:.3f}" for v in blk_sym),
                                 "trace_raw": float(blk_raw.sum()), "trace_sym": float(blk_sym.sum()),
                                 "lam_raw_over_q2": " ".join(f"{v / qm ** 2:.1f}" for v in blk_raw),
                                 "lam_sym_over_q2": " ".join(f"{v / qm ** 2:.1f}" for v in blk_sym)})

    # (c) sum-rule hypothesis: forbidden LA coupling along [100] with and without the separate sum rule
    for qm in (0.05, 0.0125):
        q = np.array([qm, 0, 0])
        wr, Er = diagonalise(S["reference"].dynmat(q, asr_onsite=True))
        for klabel in Ks:
            la = project_coupling(Ks[klabel](q), Er, wr, TOL)["lam"][2]
            summary.append(f"forbidden LA coupling along [100], |q| = {qm}: {klabel}: {la:.2f} cm^-2 (Lambda/q^2 = {la / qm ** 2:.0f})")

    # (d) eta_B and eta_C with K and with K_sym on the production hybrid model (frozen frequencies, linewidths, eigenvectors)
    vogt, _ = E.load_vogt()
    modes = E.construction_modes(T_K, MESH_N, surf)
    eta = {}
    for c in ("B", "C"):
        eta[c], _, _ = E.assemble_construction(T_K, construction=c, modes=modes, surface=surf, vogt=vogt)
    bare = {t: E.qe_set(t, surf["spec"]) for t in E.STRAIN_SETS}
    mods = []
    lamB_check, lamC_check = [], []
    for iq, q in enumerate(qmesh):
        omega_r, E_r = diagonalise(S["reference"].dynmat(q, asr_onsite=True))
        pB = project_coupling(Ksym_mesh["K_HA (production, B)"][iq], E_r, omega_r, TOL)
        pC = project_coupling(Ksym_mesh["K_SCPH (production, C)"][iq], E_r, omega_r, TOL)
        pB0 = project_coupling(Kmesh["K_HA (production, B)"][iq], E_r, omega_r, TOL)
        pC0 = project_coupling(Kmesh["K_SCPH (production, C)"][iq], E_r, omega_r, TOL)
        for nu in range(3 * NAT):
            m = dict(modes[iq * 3 * NAT + nu])
            assert m["iq"] == iq and m["nu"] == nu + 1
            lamB_check.append(abs(pB0["lam"][nu] - m["lam_B"])); lamC_check.append(abs(pC0["lam"][nu] - m["lam_C"]))
            m["lam_B"], m["lam_C"] = float(pB["lam"][nu]), float(pC["lam"][nu])
            mods.append(m)
    summary.append(f"reproduction of the production couplings from the stored K: max |d lam_B| = {max(lamB_check):.2e}, "
                   f"max |d lam_C| = {max(lamC_check):.2e} cm^-2")
    eta_sym = {}
    for c in ("B", "C"):
        eta_sym[c], _, _ = E.assemble_construction(T_K, construction=c, modes=mods, surface=surf, vogt=vogt)
        d = eta_sym[c] / eta[c] - 1.0
        summary.append(f"eta_{c}(300 K): K {eta[c]:.6e} Pa s; K_sym {eta_sym[c]:.6e} Pa s; change {100 * d:+.3f} %")
        rows.append({"block": "eta", "K": f"construction {c}", "eta_K_Pas": eta[c], "eta_Ksym_Pas": eta_sym[c], "rel_change": d})
    summary.append(f"eta_C/eta_B: K {eta['C'] / eta['B']:.4f}; K_sym {eta_sym['C'] / eta_sym['B']:.4f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = DIAG / "strain_symmetry_check.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# strain_symmetry_check.py: symmetry projection of the shear perturbation K(q) onto the channel of eps_xy "
                 "(16 operations of O_h, chi = +-1);\n# hybrid model, 300 K, correction mesh 2^3 / inner mesh 12^3; Lambda in cm^-2 per "
                 "unit engineering shear (h = 0.010); lam_raw from K, lam_sym from K_sym;\n# block = mesh_norm | acoustic_line | eta.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    txt = DIAG / "strain_symmetry_summary.txt"
    tmp = txt.with_suffix(".tmp")
    tmp.write_text("\n".join(summary) + "\n")
    tmp.rename(txt)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
