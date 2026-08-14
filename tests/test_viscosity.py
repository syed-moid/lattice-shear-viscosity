"""Guardrail tests for the lattice shear viscosity kernel.

Three required checks:
1. Dimensional analysis / sanity: representative perovskite inputs must give
   eta in the 1e-5 .. 1e-2 Pa s window (eta(300 K) ~ 1e-4 .. 1e-3 expected).
2. High-T limit: eta -> (k_B T / V) * sum(gruneisen^2 * tau).
3. Kinetic-theory cross-estimate: eta ~ 3 * n_at * k_B * T * <gruneisen^2 * tau>
   must agree with the full formula within a factor of a few.
"""

import numpy as np
import pytest
from scipy.constants import Boltzmann as K_B

from latvisc.viscosity import (
    bose_einstein,
    shear_viscosity,
    tau_effective,
    tau_from_linewidth,
    tau_two_pole_exact,
    thz_to_rad_per_s,
)
from latvisc.validation import kinetic_viscosity_estimate

# Representative SrTiO3-like mode set: 5-atom cell, 15 modes.
# Frequencies span acoustic-to-optic range; gruneisen values are O(1);
# linewidths are a few percent of the frequency (strongly anharmonic).
VOLUME = 61.08e-30  # m^3, relaxed cubic SrTiO3 cell (vc_relax, this repo)
N_ATOMS = 5

rng = np.random.default_rng(42)
OMEGA = thz_to_rad_per_s(np.linspace(1.5, 24.0, 3 * N_ATOMS))  # rad/s
GRUNEISEN = rng.uniform(0.5, 2.5, size=3 * N_ATOMS)  # dimensionless, O(1)
LINEWIDTH = 0.03 * OMEGA  # rad/s, moderately anharmonic


def test_dimensional_sanity_300K():
    """eta at 300 K must land in the physical Pa*s window, not 1e-20."""
    tau = tau_from_linewidth(LINEWIDTH)
    eta = shear_viscosity(
        omega=OMEGA,
        gruneisen=GRUNEISEN,
        tau=tau,
        volume=VOLUME,
        temperature=300.0,
    )
    assert np.isfinite(eta)
    assert eta > 0.0
    assert 1e-5 < eta < 1e-2, f"eta = {eta:.3e} Pa s is outside the sanity window"


def test_high_temperature_limit():
    """For k_B T >> hbar*omega: eta -> (k_B T / V) * sum(gruneisen^2 * tau)."""
    temperature = 20000.0  # far above all mode energies
    tau = tau_from_linewidth(LINEWIDTH)
    eta_full = shear_viscosity(
        omega=OMEGA,
        gruneisen=GRUNEISEN,
        tau=tau,
        volume=VOLUME,
        temperature=temperature,
    )
    eta_classical = (K_B * temperature / VOLUME) * np.sum(GRUNEISEN**2 * tau)
    assert eta_full == pytest.approx(eta_classical, rel=0.02)


def test_kinetic_theory_cross_estimate():
    """eta ~ 3 n_at k_B T <gruneisen^2 tau> within a factor of a few at 300 K."""
    temperature = 300.0
    tau = tau_from_linewidth(LINEWIDTH)
    eta_full = shear_viscosity(
        omega=OMEGA,
        gruneisen=GRUNEISEN,
        tau=tau,
        volume=VOLUME,
        temperature=temperature,
    )
    eta_kinetic = kinetic_viscosity_estimate(
        atom_density=N_ATOMS / VOLUME,
        temperature=temperature,
        mean_gruneisen_sq_tau=np.mean(GRUNEISEN**2 * tau),
    )
    ratio = eta_full / eta_kinetic
    assert 0.2 < ratio < 5.0, f"full/kinetic = {ratio:.2f}, expected within a factor of a few"


def test_bose_einstein_limits():
    """n -> k_B T / (hbar omega) classically; n -> 0 as T -> 0."""
    omega = thz_to_rad_per_s(5.0)
    from scipy.constants import hbar

    n_hot = bose_einstein(omega, 20000.0)
    assert n_hot == pytest.approx(K_B * 20000.0 / (hbar * omega) - 0.5, rel=0.01)
    assert bose_einstein(omega, 1.0) < 1e-100


def test_tau_two_pole_exact_limits_and_crossover():
    # underdamped: reduces to the sharp-resonance 1/(2*Gamma) with a
    # relative correction (Gamma/omega)^2
    omega, linewidth = 1.0e13, 1.0e11
    assert tau_two_pole_exact(omega, linewidth) == pytest.approx(
        tau_from_linewidth(linewidth) * (1.0 + (linewidth / omega) ** 2)
    )
    # deep overdamped: Gamma/(2*omega^2) — half the slow-pole form
    omega, linewidth = 1.0e11, 1.0e13
    assert tau_two_pole_exact(omega, linewidth) == pytest.approx(
        linewidth / (2.0 * omega**2), rel=1e-3
    )
    assert tau_two_pole_exact(omega, linewidth) == pytest.approx(
        0.5 * tau_effective(omega, linewidth), rel=1e-3
    )
    # no regime switch: smooth and finite at critical damping
    omega = linewidth = 1.0e12
    assert tau_two_pole_exact(omega, linewidth) == pytest.approx(1.0 / linewidth)


def test_tau_effective_underdamped_limit():
    """linewidth << omega: tau_eff reduces to 1/(2 linewidth)."""
    omega = thz_to_rad_per_s(5.0)
    linewidth = 0.01 * omega
    assert tau_effective(omega, linewidth) == pytest.approx(
        tau_from_linewidth(linewidth), rel=1e-3
    )


def test_tau_effective_overdamped_limit():
    """linewidth >> omega: tau_eff -> linewidth / omega^2."""
    omega = thz_to_rad_per_s(0.2)
    linewidth = 50.0 * omega
    assert tau_effective(omega, linewidth) == pytest.approx(
        linewidth / omega**2, rel=1e-3
    )
