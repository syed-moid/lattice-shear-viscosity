#!/usr/bin/env python3
"""Degeneracy audit on the renormalised multiplets of the production model (SrTiO3, 300 K).

For the 20 largest-|gamma_C| modes: the multiplet (renormalised frequencies within 0.5 cm-1, and 2 cm-1
for tolerance insensitivity), the projected-block eigenvalues that the production assigns as
couplings, the gauge-invariant Tr K_sub^2, and a numerical gauge check: the block is rotated by a
random unitary inside the multiplet and the eigenvalues and Tr K_sub^2 are recomputed (must agree to
rounding), while the diagonal elements e^dagger K e change. Also the intra-multiplet spread of
omega_r, n(n+1) and tau that the common-prefactor form assumes small.
Writes: data/processed/reports/degeneracy_audit_renormalised_SrTiO3.csv
Usage: uv run python scripts/audit_degeneracy_renormalised.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.coupling import diagonalise, multiplet_groups, project_coupling, strain_derivative_matrix  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
T_K = 300
rng = np.random.default_rng(11)


def main() -> None:
    vogt, _ = eta_mod.load_vogt()
    surface = eta_mod.load_construction_surface(T_K)
    modes = eta_mod.construction_modes(T_K, 11, surface)
    eta, _, _, details = eta_mod.assemble_construction(T_K, modes=modes, surface=surface, vogt=vogt, return_details=True)
    sets = surface["sets"]
    top = sorted(details, key=lambda d: -abs(d["gruneisen"]))[:20]
    lines = ["iq,nu,omega_r_cm1,gamma_C,eta_contrib_Pas,multiplet_size_0p5,multiplet_size_2,block_eigenvalues_cm2,"
             "TrK2_cm4,TrK2_after_random_rotation_cm4,max_eigenvalue_change_cm2,max_diag_change_cm2,"
             "spread_omega_r_pct,spread_n_n1_pct,spread_tau_pct"]
    print(f"eta(300 K) = {eta:.4e} Pa s; 20 largest |gamma_C|:")
    worst = 0.0
    for d in top:
        q = d["q"]
        w, E = diagonalise(sets["reference"].dynmat(q, asr_onsite=True))
        K = strain_derivative_matrix(sets["shear_xy_p005"].dynmat(q, asr_onsite=True), sets["shear_xy_m005"].dynmat(q, asr_onsite=True), eta_mod.H_ENG)
        p = project_coupling(K, E, w, 0.5)
        nu = d["nu"] - 1
        g = [g for g in multiplet_groups(w, 0.5) if nu in g][0]
        g2 = [g for g in multiplet_groups(w, 2.0) if nu in g][0]
        Kb = E[:, g].conj().T @ K @ E[:, g]
        ev = np.linalg.eigvalsh(0.5 * (Kb + Kb.conj().T))
        U = np.linalg.qr(rng.normal(size=(len(g), len(g))) + 1j * rng.normal(size=(len(g), len(g))))[0]
        E2 = E.copy(); E2[:, g] = E[:, g] @ U
        p2 = project_coupling(K, E2, w, 0.5)
        dev_eig = float(np.abs(np.sort(p2["lam"][g]) - np.sort(p["lam"][g])).max())
        dev_diag = float(np.abs(p2["diag"][g] - p["diag"][g]).max())
        worst = max(worst, dev_eig / max(1.0, np.abs(ev).max()))
        members = [x for x in details if x["iq"] == d["iq"] and x["nu"] - 1 in g]
        spread = lambda key: 100.0 * (max(x[key] for x in members) - min(x[key] for x in members)) / abs(np.mean([x[key] for x in members]))  # noqa: E731
        lines.append(f"{d['iq']},{d['nu']},{d['omega_r']:.4f},{d['gruneisen']:.4f},{d['eta_contrib']:.6e},{len(g)},{len(g2)},"
                     f"\"{' '.join(f'{v:.6g}' for v in ev)}\",{p['trk2'][nu]:.6e},{p2['trk2'][nu]:.6e},{dev_eig:.3e},{dev_diag:.3e},"
                     f"{spread('omega_r'):.4f},{spread('n_n_plus_1'):.4f},{spread('tau_s'):.4f}")
        print(f"  iq {d['iq']:4d} nu {d['nu']:2d} omega_r {d['omega_r']:7.2f} gamma_C {d['gruneisen']:+8.3f} multiplet {len(g)} "
              f"eig {np.round(ev, 1)} TrK2 {p['trk2'][nu]:.4e} rot-dev eig {dev_eig:.1e} diag {dev_diag:.1e}")
    print(f"largest relative eigenvalue change under a random in-multiplet rotation: {worst:.2e} (must be rounding)")
    out = REPO / "data" / "processed" / "reports" / "degeneracy_audit_renormalised_SrTiO3.csv"
    header = ["# degeneracy_audit_renormalised_SrTiO3.csv - produced by scripts/audit_degeneracy_renormalised.py",
              "# production model (hybrid model tut_z_od1, SELF_OFFDIAG = 1, inner mesh 12^3, construction C), 300 K, 11^3: projected-block",
              "# eigenvalues and Tr K_sub^2 of the multiplets of the 20 largest |gamma_C| modes; gauge check by a random",
              "# unitary rotation inside the multiplet (eigenvalues and Tr K_sub^2 invariant, diagonal elements not)."]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
