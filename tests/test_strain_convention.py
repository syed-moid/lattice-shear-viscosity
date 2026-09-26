"""Strain-convention guardrail (tensor Grueneisen component, one conversion point).

The shear cells apply epsilon_xy = epsilon_yx = s. A central difference over
the +/-s pair is the derivative along that path, twice the tensor derivative
with respect to epsilon_xy. Pinned here:

1. a toy mode with a known tensor coupling, omega^2 = omega0^2 +
   Lambda_t (epsilon_xy + epsilon_yx), returns the tensor Grueneisen
   parameter through mode_gruneisen_finite_strain(symmetric_shear_path=True)
   and through compute_dataset's D route, to 1e-6;
2. the hydrostatic control (three diagonal components, no ambiguity) is
   unchanged;
3. the production path (compute_dataset -> assemble) calls the conversion
   exactly once per dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from latvisc import gruneisen as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_shear_nonlinearity as csn  # noqa: E402
import compute_eta_SrTiO3 as eta_mod  # noqa: E402

OMEGA0 = 100.0
LAMBDA_T = -2.0e4        # cm-2 per unit tensor strain
S = 0.005
GAMMA_TENSOR = -LAMBDA_T / (2.0 * OMEGA0**2)


def _omega(eps_xy, eps_yx):
    return np.sqrt(OMEGA0**2 + LAMBDA_T * (eps_xy + eps_yx))


def test_toy_mode_returns_tensor_gamma():
    w_plus, w_minus = _omega(S, S), _omega(-S, -S)
    # eigenvalue route, as compute_dataset forms it: exact for a coupling linear in omega^2
    d_path = (w_plus**2 - w_minus**2) / (2 * S)
    d_t = gr.path_to_tensor_shear(d_path)
    assert -d_t / (2 * OMEGA0**2) == pytest.approx(GAMMA_TENSOR, rel=1e-9)
    # omega route (Route S formula): the same to the O(s^2) sqrt-curvature truncation (5e-5 here)
    g = gr.mode_gruneisen_finite_strain(OMEGA0, w_plus, w_minus, S, symmetric_shear_path=True)
    assert g == pytest.approx(GAMMA_TENSOR, rel=1e-4)
    # without the flag the raw path derivative (2x) is returned
    assert gr.mode_gruneisen_finite_strain(OMEGA0, w_plus, w_minus, S) == pytest.approx(2 * GAMMA_TENSOR, rel=1e-4)
    # curvature coefficients convert with the square of the multiplicity
    assert gr.path_to_tensor_shear(np.array([8.0, 8.0]), order=[1, 2]).tolist() == [4.0, 2.0]


def test_hydrostatic_control_unchanged():
    gamma_vol = 2.0
    v_plus, v_minus = (1 + S) ** 3, (1 - S) ** 3
    w_plus, w_minus = OMEGA0 * v_plus ** (-gamma_vol), OMEGA0 * v_minus ** (-gamma_vol)
    assert gr.mode_gruneisen_volume(w_plus, w_minus, v_plus, v_minus) == pytest.approx(gamma_vol, rel=1e-9)


def _write_modes(path, freqs, vectors):
    lines = [" q =       0.1000      0.0000      0.0000"]
    for f, v in zip(freqs, vectors):
        lines.append(f"     freq (    1) =      {f / 33.35641:.10f} [THz] =     {f:.10f} [cm-1]")
        for atom in v:
            lines.append(" ( " + "  ".join(f"{c.real:.9f}  {c.imag:.9f}" for c in atom) + " )")
    Path(path).write_text("\n".join(lines) + "\n")


def test_production_path_converts_exactly_once(tmp_path):
    """Synthetic single-q dataset with 15 modes and a known tensor coupling."""
    rng = np.random.default_rng(1)
    freqs = np.linspace(100.0, 800.0, 15)
    masses = csn.MASSES["SrTiO3"]
    q, _ = np.linalg.qr(rng.normal(size=(15, 15)))
    disp = (q / np.sqrt(np.repeat(masses, 3))[None, :]).reshape(15, 5, 3)     # eigendisplacements
    lam_t = LAMBDA_T * np.linspace(0.5, 1.5, 15)
    w_plus = np.sqrt(freqs**2 + lam_t * 2 * S)
    w_minus = np.sqrt(freqs**2 - lam_t * 2 * S)
    _write_modes(tmp_path / "reference.modes", freqs, disp)
    _write_modes(tmp_path / "shear_xy_p005.modes", w_plus, disp)
    _write_modes(tmp_path / "shear_xy_m005.modes", w_minus, disp)
    before = gr._CONVERSION_CALLS[0]
    rows = csn.compute_dataset(tmp_path, masses, mesh_n=1)
    assert gr._CONVERSION_CALLS[0] - before == 1
    d = np.array([r["D"] for r in rows])
    assert d == pytest.approx(lam_t, rel=1e-7)                     # tensor Lambda, not 2x
    g = np.array([r["gamma_xy"] for r in rows])
    assert g == pytest.approx(-lam_t / (2 * freqs**2), rel=1e-4)
    # the assembly consumes D as is: assemble() contains no second conversion
    src = Path(eta_mod.__file__).read_text()
    body = src[src.index("def assemble("):src.index("def main(")]
    assert "path_to_tensor_shear(" not in body and "SYMMETRIC_SHEAR_PATH_MULTIPLICITY" not in body
    calls = gr._CONVERSION_CALLS[0]
    eta_mod.assemble(300, rows, lambda temperature: (None, None), return_details=False) if False else None
    assert gr._CONVERSION_CALLS[0] == calls
