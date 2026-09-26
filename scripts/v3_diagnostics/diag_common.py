"""Shared loaders for the EPJ B revision audit scripts.

Imports the production assembly (scripts/compute_eta_SrTiO3.py) unchanged and
exposes a per-mode table built from its audit-details output, plus the
alternative stress-variable kernel under test. Nothing here writes to
data/processed/; every output of the audit scripts lands under
data/processed/v3_diagnostics/.

Path layout: this file lives in data/processed/v3_diagnostics/scripts/, three
levels below the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
DIAG_DIR = DIAG

from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
from latvisc.viscosity import bose_einstein, tau_two_pole_exact  # noqa: E402

CM1 = eta_mod.CM1            # rad/s per cm-1
V_CELL = eta_mod.V_CELL
N_Q = eta_mod.N_Q
KT_CM1_300K = K_B * 300.0 / (HBAR * CM1)   # k_B T / (hbar c) at 300 K, cm-1


def tau_two_pole_stress(omega, linewidth):
    """Time-integrated correlator of the x^2 (stiffness-conjugate) stress of a
    classical damped oscillator, normalised to gamma^2 (k_B T)^2:

        tau_stress = 1/(2 Gamma) + 2 Gamma / omega^2

    Same unit conventions as latvisc.viscosity.tau_two_pole_exact (omega and
    linewidth in rad/s, HWHM convention for the linewidth). Reduces to
    1/(2 Gamma) for Gamma << omega; deep overdamped it is 4x the energy
    (occupation) kernel tau_two_pole_exact = 1/(2 Gamma) + Gamma/(2 omega^2).
    """
    omega = np.asarray(omega, dtype=float)
    linewidth = np.asarray(linewidth, dtype=float)
    return 1.0 / (2.0 * linewidth) + 2.0 * linewidth / omega**2


_ROWS: dict[str, list] = {}


def load_rows(material: str = "SrTiO3"):
    """Strained-cell dataset rows (one per (q, branch)) exactly as production
    consumes them: omega_ref (bare, cm-1), D = d(omega^2)/ds (cm-2 per unit
    path parameter s), gamma_xy (omega-space central difference), flags."""
    if material not in _ROWS:
        _ROWS[material] = compute_dataset(MODES_DIR / material, MASSES[material], mesh_n=11)
    return _ROWS[material]


def mode_table(temperature: int = 300, extra_gamma_hwhm_cm1=None):
    """Per-mode production details at one temperature, merged with the
    strained-cell row (q vector, D, degeneracy flag).

    Returns (eta_total, sectors, flags, details) where each detail dict has:
      iq, branch (1-based), q (fractional-mesh label as printed by matdyn),
      sector, omega0 (bare, cm-1), omega_r (renormalised, cm-1),
      gamma_hwhm (anharmonic + any extra, cm-1, HWHM), gruneisen (as used in
      production), D (cm-2 per unit s), tau_s (production two-pole exact, s),
      eta_contrib (Pa s).
    """
    rows = load_rows("SrTiO3")
    vogt, _ = eta_mod.load_vogt()
    eta, sec, flags, details = eta_mod.assemble(
        temperature, rows, vogt, extra_gamma_hwhm_cm1=extra_gamma_hwhm_cm1,
        return_details=True)
    by_key = {(r["iq"], r["branch"]): r for r in rows}
    for d in details:
        r = by_key[(d["iq"], d["branch"])]
        d["q"] = r["q"]
        d["D"] = r["D"]
        d["degenerate"] = r["degenerate"]
        d["gamma_xy_omega_space"] = r["gamma_xy"]
    return eta, sec, flags, details


def recompute_eta(details, temperature, kernel=tau_two_pole_exact,
                  gamma_iso_hwhm_cm1=None, gruneisen_scale=1.0):
    """Re-evaluate the production sum from a details list with a swappable
    lifetime kernel and an optional per-mode isotope HWHM (cm-1) array.
    Returns (eta_total, sector dict, per-mode contributions array)."""
    norm = 1.0 / (V_CELL * N_Q * K_B * temperature)
    contribs = np.zeros(len(details))
    sectors: dict[str, float] = {}
    for i, d in enumerate(details):
        w = d["omega_r"] * CM1
        lw_cm1 = d["gamma_hwhm"]
        if gamma_iso_hwhm_cm1 is not None:
            lw_cm1 = lw_cm1 + gamma_iso_hwhm_cm1[i]
        lw = lw_cm1 * CM1
        n = bose_einstein(w, temperature)
        tau = float(kernel(w, lw))
        g = d["gruneisen"] * gruneisen_scale
        c = (HBAR * w) ** 2 * g * g * n * (n + 1.0) * tau * norm
        contribs[i] = c
        sectors[d["sector"]] = sectors.get(d["sector"], 0.0) + c
    return float(contribs.sum()), sectors, contribs
