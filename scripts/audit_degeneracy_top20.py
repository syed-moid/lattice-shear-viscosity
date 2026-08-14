#!/usr/bin/env python3
"""Degeneracy audit of the largest shear Grueneisen parameters (SrTiO3).

Within a degenerate multiplet the individual gamma values are
basis-dependent (the eigenvector-tracked pairing fixes a gauge), but the
viscosity sum is not: every member of a near-degenerate multiplet shares
omega_tilde, n(n+1), and tau to high accuracy, so the sum uses the
gauge-invariant multiplet total Sigma gamma^2 — equivalently
Sigma Lambda^2 = Sigma (d omega^2/d eps)^2, the squared Frobenius norm of
the strain perturbation projected onto the subspace.

This audit, for the 20 largest-|gamma_xy| modes of the production 300 K
sum:
  1. tabulates q-point, branch, omega_tilde, gamma, tau, and the
     contribution to eta;
  2. identifies each mode's near-degenerate multiplet (splitting tolerance
     2 cm-1 in omega_tilde; 5 cm-1 reported alongside to show
     tolerance-insensitivity) and reports the intra-multiplet maximum
     relative spread of omega_tilde, n(n+1), and tau;
  3. computes the multiplet-summed Sigma gamma^2 and verifies that the
     summed contribution equals the common-prefactor form
     (hbar*omega)^2 n(n+1) tau * Sigma gamma^2 to within the spread;
  4. gauge check: reconstructs the +/-0.005 strained dynamical matrices
     from the committed mode files (complete eigenbasis => exact
     reconstruction), projects Delta D / Delta eps onto each reference
     near-degenerate subspace, diagonalizes the block, and compares the
     eigenvalue-squared sum against the tracked-assignment
     Sigma (d omega^2/d eps)^2. Agreement means the tracked pairing spans
     the same invariant subspace as the direct block diagonalization —
     the per-mode gauge cannot move Sigma gamma^2.

Reads : data/raw/gruneisen_modes/SrTiO3/*.modes + assembly inputs
Writes: data/processed/reports/degeneracy_audit_top20_SrTiO3.csv

Usage: uv run python scripts/audit_degeneracy_top20.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import H, MASSES, MODES_DIR, compute_dataset  # noqa: E402
from compute_eta_SrTiO3 import REPO, assemble, load_vogt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import orthonormal_eigenvectors  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402
from latvisc.viscosity import bose_einstein  # noqa: E402

CM1 = 2.0 * np.pi * speed_of_light * 100.0
T_K = 300
TOL_MAIN = 2.0   # cm-1, omega_tilde splitting tolerance
TOL_WIDE = 5.0
BARE_DEGEN_TOL = 2.0  # cm-1, bare-subspace tolerance for the gauge check


def group_by_tolerance(values: np.ndarray, tol: float) -> list[list[int]]:
    """Chain-group sorted values: consecutive gaps < tol join a group."""
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for i in order[1:]:
        if values[i] - values[current[-1]] < tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def multiplet_of(details_at_q: list[dict], member: dict, tol: float) -> list[dict]:
    omegas = np.array([d["omega_r"] for d in details_at_q])
    for group in group_by_tolerance(omegas, tol):
        if any(details_at_q[i] is member for i in group):
            return [details_at_q[i] for i in group]
    raise AssertionError("member not found in its own q-point grouping")


def reconstruct_dynmat(freqs_cm1: np.ndarray, vectors, masses) -> np.ndarray:
    """Mass-weighted dynamical matrix (signed omega^2, cm-2) from a
    complete eigen-decomposition: Dyn = Z^dag diag(sign(w) w^2) Z with Z
    the orthonormalized mode rows."""
    z = orthonormal_eigenvectors(vectors, masses)
    lam = np.sign(freqs_cm1) * freqs_cm1**2
    return z.conj().T @ (lam[:, None] * z)


def gauge_check(iq: int, branches: list[int], reference, plus, minus, masses):
    """Project (Dyn(+eps)-Dyn(-eps))/(2 eps) onto the reference subspace
    spanned by `branches` (1-based), diagonalize the Hermitized block, and
    return (sum of eigenvalue^2, block eigenvalues)."""
    _, freq_ref, vec_ref = reference[iq]
    _, freq_p, vec_p = plus[iq]
    _, freq_m, vec_m = minus[iq]
    dyn_p = reconstruct_dynmat(freq_p, vec_p, masses)
    dyn_m = reconstruct_dynmat(freq_m, vec_m, masses)
    delta = (dyn_p - dyn_m) / (2.0 * H)
    z_ref = orthonormal_eigenvectors(vec_ref, masses)
    idx = [b - 1 for b in branches]
    # rows-as-bras convention, consistent with reconstruct_dynmat
    # (Dyn = Z^dag Lambda Z  <=>  Z Dyn Z^dag = Lambda): the projected
    # block is Z[m] Delta Z[m]^dag. The conjugate-transposed variant is
    # equivalent only for real representations and is wrong at
    # complex-representation q-points.
    block = z_ref[idx] @ delta @ z_ref[idx].conj().T
    block = 0.5 * (block + block.conj().T)
    eigvals = np.linalg.eigvalsh(block)
    return float(np.sum(eigvals**2)), eigvals


def max_rel_spread(values) -> float:
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    return float((values.max() - values.min()) / abs(mean)) if mean else float("nan")


def main() -> None:
    masses = MASSES["SrTiO3"]
    directory = MODES_DIR / "SrTiO3"
    rows = compute_dataset(directory, masses, mesh_n=11)
    vogt, _ = load_vogt()
    eta_total, _, _, details = assemble(T_K, rows, vogt, return_details=True)

    reference = read_modes(directory / "reference.modes")
    plus = read_modes(directory / "shear_xy_p005.modes")
    minus = read_modes(directory / "shear_xy_m005.modes")
    d_tracked = {(r["iq"], r["branch"]): r["D"] for r in rows}
    bare_omega = {(r["iq"], r["branch"]): r["omega_ref"] for r in rows}

    by_iq: dict[int, list[dict]] = {}
    for d in details:
        by_iq.setdefault(d["iq"], []).append(d)

    top = sorted(details, key=lambda d: abs(d["gruneisen"]), reverse=True)[:20]

    print(f"eta_xyxy(300 K) = {eta_total:.4e} Pa s; auditing top-20 |gamma_xy| modes")
    out_lines = []
    gauge_devs = []
    audited_multiplets = set()
    n_isolated = 0

    for rank, d in enumerate(top, start=1):
        iq, branch = d["iq"], d["branch"]
        q = reference[iq][0]
        mult2 = multiplet_of(by_iq[iq], d, TOL_MAIN)
        mult5 = multiplet_of(by_iq[iq], d, TOL_WIDE)
        branches2 = sorted(m["branch"] for m in mult2)

        w = np.array([m["omega_r"] for m in mult2])
        occ = bose_einstein(w * CM1, T_K)
        taus = np.array([m["tau_s"] for m in mult2])
        spread_w = max_rel_spread(w)
        spread_nn = max_rel_spread(occ * (occ + 1.0))
        spread_tau = max_rel_spread(taus)

        gamma2_sum = float(sum(m["gruneisen"] ** 2 for m in mult2))
        contrib_sum = float(sum(m["eta_contrib"] for m in mult2))
        # common-prefactor reconstruction: mean per-gamma^2 prefactor
        prefac = np.array([m["eta_contrib"] / m["gruneisen"] ** 2 for m in mult2])
        recon = float(np.mean(prefac) * gamma2_sum)
        recon_dev = abs(recon - contrib_sum) / contrib_sum if contrib_sum else float("nan")

        isolated = len(mult2) == 1
        if isolated:
            n_isolated += 1

        # gauge check on the bare-degenerate subspace (leading-order
        # perturbation theory applies there); skip if the multiplet's bare
        # frequencies are not themselves near-degenerate
        bare = np.array([bare_omega[(iq, b)] for b in branches2])
        gauge_dev = float("nan")
        if not isolated and (bare.max() - bare.min()) < BARE_DEGEN_TOL:
            key = (iq, tuple(branches2))
            block_sum, _ = gauge_check(iq, branches2, reference, plus, minus, masses)
            tracked_sum = float(sum(d_tracked[(iq, b)] ** 2 for b in branches2))
            gauge_dev = abs(block_sum - tracked_sum) / tracked_sum
            if key not in audited_multiplets:
                audited_multiplets.add(key)
                gauge_devs.append((key, gauge_dev))

        if np.isfinite(gauge_dev):
            gauge_note = f"gauge_dev={gauge_dev:.2e}"
        elif isolated:
            gauge_note = "ISOLATED (gauge moot)"
        else:
            gauge_note = "bare-split>tol (no gauge check)"
        print(f"#{rank:2d} iq={iq:4d} q=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}) "
              f"br={branch:2d} {d['sector']:15s} omega_t={d['omega_r']:7.2f} "
              f"gamma={d['gruneisen']:+8.2f} tau={d['tau_s']*1e12:6.2f} ps "
              f"eta_i={d['eta_contrib']:.3e} ({100*d['eta_contrib']/eta_total:.2f}%) "
              f"mult(2cm)={len(mult2)} mult(5cm)={len(mult5)} "
              f"spread[w={spread_w:.2e},nn={spread_nn:.2e},tau={spread_tau:.2e}] "
              f"Sg2={gamma2_sum:9.1f} recon_dev={recon_dev:.2e} {gauge_note}")

        out_lines.append(
            f"{rank},{iq},{q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{branch},"
            f"{d['sector']},{d['omega0']:.4f},{d['omega_r']:.4f},"
            f"{d['gruneisen']:.4f},{d['tau_s']:.6e},{d['eta_contrib']:.6e},"
            f"{100*d['eta_contrib']/eta_total:.3f},{len(mult2)},{len(mult5)},"
            f"\"{';'.join(str(b) for b in branches2)}\",{spread_w:.3e},"
            f"{spread_nn:.3e},{spread_tau:.3e},{gamma2_sum:.4f},"
            f"{recon_dev:.3e},{gauge_dev:.3e}")

    finite = [g for _, g in gauge_devs]
    max_gauge = max(finite) if finite else float("nan")
    print(f"\naudited {len(gauge_devs)} distinct bare-degenerate multiplets; "
          f"max |Sigma eigval^2 - Sigma D_tracked^2| / Sigma D_tracked^2 = "
          f"{max_gauge:.3e}")
    print(f"isolated (non-degenerate at 2 cm-1) top-20 modes: {n_isolated} "
          f"(gauge concern moot for these)")

    rep = REPO / "data" / "processed" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    header = [
        "# degeneracy_audit_top20_SrTiO3.csv - scripts/audit_degeneracy_top20.py",
        f"# Top-20 |gamma_xy| modes of the production 300 K sum "
        f"(eta_xyxy = {eta_total:.4e} Pa s).",
        "# Multiplets grouped by omega_tilde splitting < 2 cm-1 (mult_n_2cm1)",
        "# and < 5 cm-1 (mult_n_5cm1). spread_* = intra-multiplet (max-min)/|mean|.",
        "# sum_gamma2 = multiplet Sigma gamma^2; recon_dev = relative deviation of",
        "# the common-prefactor reconstruction prefac*Sigma gamma^2 from the summed",
        "# contribution. gauge_dev = |Sigma eig^2 - Sigma D_tracked^2|/Sigma D_tracked^2",
        "# for the strain perturbation projected onto the bare-degenerate subspace",
        "# (nan = isolated mode or bare splitting > 2 cm-1).",
        f"# Audited {len(gauge_devs)} distinct multiplets; max gauge_dev = {max_gauge:.3e}.",
        f"# Isolated top-20 modes at 2 cm-1 tolerance: {n_isolated}.",
        "rank,q_index,qx,qy,qz,branch,sector,omega0_cm1,omega_tilde_cm1,gruneisen_xy,"
        "tau_s,eta_contrib_Pas,contrib_percent,mult_n_2cm1,mult_n_5cm1,"
        "mult_branches_2cm1,spread_omega,spread_nn1,spread_tau,sum_gamma2,"
        "recon_dev,gauge_dev",
    ]
    out = rep / "degeneracy_audit_top20_SrTiO3.csv"
    out.write_text("\n".join(header) + "\n" + "\n".join(out_lines) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
