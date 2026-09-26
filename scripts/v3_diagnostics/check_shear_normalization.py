#!/usr/bin/env python3
"""Audit 0.A: shear-strain normalisation of the mode Grueneisen component.

Question. The strained cells are built by scripts/make_strained_inputs.py::
cell_vectors with the symmetric deformation F = I + s (x y + y x), i.e.
eps_xy = eps_yx = s. The pipeline forms the derivative along that path,
d/ds, and divides by 2s (central difference). Is the quantity that enters
the viscosity sum the tensor component gamma_xy = -d ln omega / d eps_xy
(nine-component convention, dH = V sum_ij sigma_ij d eps_ij, so that the
Green-Kubo eta_xyxy equals the Newtonian eta_44 used in the acoustic
conversion), or the path derivative -d ln omega / ds = 2 gamma_xy?

Method. A one-mode toy with an analytically known tensor coupling,
omega^2(eps) = omega0^2 + Lambda_t (eps_xy + eps_yx), is pushed through the
actual production functions: cell_vectors -> strain tensor read back from
the cell -> omega(+s), omega(-s) -> latvisc.gruneisen.mode_gruneisen_finite_strain
and the D = d(omega^2)/ds route of check_shear_nonlinearity / compute_eta_SrTiO3.
The hydrostatic path (no factor-2 ambiguity) is run through the same
machinery as a control, and a quadratic elastic-energy toy shows the same
factor between the path-conjugate force and the tensor stress.

Writes: data/processed/v3_diagnostics/shear_normalization_toy.csv
"""

from __future__ import annotations

import numpy as np

from diag_common import REPO, DIAG_DIR  # noqa: F401  (sets sys.path)
from make_strained_inputs import cell_vectors  # noqa: E402
from latvisc.gruneisen import mode_gruneisen_finite_strain, mode_gruneisen_volume  # noqa: E402

A_ANG = 3.8930
S = 0.005                     # production step
OMEGA0 = 100.0                # cm-1, toy mode
LAMBDA_T = -2.0e4             # cm-2 per unit tensor strain eps_xy (nine-component convention)
GAMMA_VOL_T = 2.0             # toy hydrostatic Grueneisen, -d ln omega / d ln V


def strain_tensor(vectors, a):
    """Read the symmetric strain back from the cell: rows are a_i = F . a e_i,
    so F = (rows / a)^T and eps = (F + F^T)/2 - I."""
    F = np.asarray(vectors).T / a
    return 0.5 * (F + F.T) - np.eye(3)


def toy_omega_shear(eps):
    """omega^2 = omega0^2 + Lambda_t (eps_xy + eps_yx): tensor coupling with
    the nine-component convention, so that -d(omega^2)/d eps_xy = -Lambda_t
    and the tensor Grueneisen is gamma_xy = -Lambda_t / (2 omega0^2)."""
    return np.sqrt(OMEGA0**2 + LAMBDA_T * (eps[0, 1] + eps[1, 0]))


def toy_omega_hydro(eps):
    """omega = omega0 (V/V0)^(-gamma_vol) with ln(V/V0) = tr(eps) to first order."""
    lnv = np.log(np.linalg.det(np.eye(3) + eps))
    return OMEGA0 * np.exp(-GAMMA_VOL_T * lnv)


def main() -> None:
    lines = ["quantity,pipeline_value,true_tensor_value,ratio_pipeline_over_true"]

    # ---- shear path through the production cell builder ----
    eps_p = strain_tensor(cell_vectors(A_ANG, "shear", +S), A_ANG)
    eps_m = strain_tensor(cell_vectors(A_ANG, "shear", -S), A_ANG)
    assert np.isclose(eps_p[0, 1], S) and np.isclose(eps_p[1, 0], S), eps_p
    assert np.isclose(eps_m[0, 1], -S) and np.isclose(eps_m[1, 0], -S), eps_m
    print(f"cell_vectors(shear, s={S:+.3f}) -> eps_xy = {eps_p[0,1]:+.4f}, "
          f"eps_yx = {eps_p[1,0]:+.4f}  (both components move: path parameter s)")

    w_p = toy_omega_shear(eps_p)
    w_m = toy_omega_shear(eps_m)
    gamma_true = -LAMBDA_T / (2.0 * OMEGA0**2)

    # (a) omega-space central difference, as latvisc.gruneisen does (Route S formula, Eq. 14)
    gamma_pipe = float(mode_gruneisen_finite_strain(OMEGA0, w_p, w_m, S))
    # (b) eigenvalue-space route used by production: D = (w+^2 - w-^2)/(2s), gamma = -D/(2 w0^2)
    D = (w_p**2 - w_m**2) / (2.0 * S)
    gamma_D = -D / (2.0 * OMEGA0**2)
    lam_route_h = -D          # "Lambda" of Route H, compared with the tensor Lambda_t
    print(f"toy tensor coupling: Lambda_t = {LAMBDA_T:.1f} cm-2, gamma_xy(tensor) = {gamma_true:+.4f}")
    print(f"  pipeline omega-route gamma (mode_gruneisen_finite_strain): {gamma_pipe:+.4f}  "
          f"ratio {gamma_pipe / gamma_true:.4f}")
    print(f"  pipeline D-route gamma (-D/(2 w0^2), compute_eta_SrTiO3): {gamma_D:+.4f}  "
          f"ratio {gamma_D / gamma_true:.4f}")
    print(f"  pipeline Route-H Lambda = -D = {lam_route_h:.1f} vs tensor -Lambda_t = {-LAMBDA_T:.1f}  "
          f"ratio {lam_route_h / (-LAMBDA_T):.4f}")
    print(f"  => eta (which weights gamma^2) from the pipeline gamma is "
          f"{(gamma_D / gamma_true) ** 2:.3f} x the tensor-gamma value")
    lines.append(f"gamma_xy_omega_route,{gamma_pipe:.6f},{gamma_true:.6f},{gamma_pipe / gamma_true:.6f}")
    lines.append(f"gamma_xy_D_route,{gamma_D:.6f},{gamma_true:.6f},{gamma_D / gamma_true:.6f}")
    lines.append(f"Lambda_routeH,{lam_route_h:.6f},{-LAMBDA_T:.6f},{lam_route_h / (-LAMBDA_T):.6f}")
    lines.append(f"eta_factor_from_gamma_squared,{(gamma_D / gamma_true) ** 2:.6f},1,{(gamma_D / gamma_true) ** 2:.6f}")

    # ---- hydrostatic control (no ambiguity: three diagonal components, ln V = tr eps) ----
    eps_hp = strain_tensor(cell_vectors(A_ANG, "hydro", +S), A_ANG)
    eps_hm = strain_tensor(cell_vectors(A_ANG, "hydro", -S), A_ANG)
    vp = np.linalg.det(np.eye(3) + eps_hp)
    vm = np.linalg.det(np.eye(3) + eps_hm)
    g_vol = float(mode_gruneisen_volume(toy_omega_hydro(eps_hp), toy_omega_hydro(eps_hm), vp, vm))
    print(f"hydrostatic control: pipeline gamma_vol = {g_vol:.4f} vs true {GAMMA_VOL_T:.4f}  "
          f"ratio {g_vol / GAMMA_VOL_T:.4f}  (the volumetric check cannot detect the shear factor)")
    lines.append(f"gamma_vol_control,{g_vol:.6f},{GAMMA_VOL_T:.6f},{g_vol / GAMMA_VOL_T:.6f}")

    # ---- elastic-energy analogue: which stress is conjugate to the path s? ----
    C44 = 1.0
    def U(eps):  # U/V = 1/2 sum_ijkl C_ijkl eps_ij eps_kl, only the shear block
        return 0.5 * C44 * (eps[0, 1] ** 2 + eps[0, 1] * eps[1, 0] + eps[1, 0] * eps[0, 1] + eps[1, 0] ** 2)
    sigma_xy_tensor = C44 * (eps_p[0, 1] + eps_p[1, 0])         # sigma_xy = dU/d eps_xy (nine-component)
    dU_ds = (U(eps_p) - U(strain_tensor(cell_vectors(A_ANG, "shear", S - 1e-6), A_ANG))) / 1e-6
    print(f"elastic toy at s={S}: dU/ds = {dU_ds:.5f}, tensor sigma_xy = {sigma_xy_tensor:.5f}  "
          f"ratio {dU_ds / sigma_xy_tensor:.3f}  (the force conjugate to s is 2 sigma_xy)")
    lines.append(f"force_conjugate_to_s_over_sigma_xy,{dU_ds:.6f},{sigma_xy_tensor:.6f},{dU_ds / sigma_xy_tensor:.6f}")

    out = DIAG_DIR / "shear_normalization_toy.csv"
    header = [
        "# shear_normalization_toy.csv - .A analytic check (check_shear_normalization.py)",
        "# One-mode toy omega^2 = omega0^2 + Lambda_t (eps_xy + eps_yx) pushed through the production",
        "# cell builder (eps_xy = eps_yx = s), latvisc.gruneisen.mode_gruneisen_finite_strain and the",
        "# D = d(omega^2)/ds route. ratio 2 for gamma means the pipeline returns the PATH derivative",
        "# -d ln omega/ds = 2 gamma_xy(tensor); eta then carries a factor 4.",
    ]
    out.write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
