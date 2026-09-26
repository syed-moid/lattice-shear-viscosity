"""Two-pole lifetime kernels against direct integration of the classical
damped-oscillator correlators.

x'' + 2 Gamma x' + omega0^2 x = xi(t); regression-theorem correlator c(t) =
<x(t)x(0)>/<x^2> obeys the same ODE with c(0) = 1, c'(0) = 0.

  energy variable  E = (v^2 + omega0^2 x^2)/2:
      int_0^inf <dE dE> dt / (kT)^2        = 1/(2G) + G/(2 w0^2)   = tau_two_pole_exact
  stress variable  S = -gamma omega0^2 x^2:
      int_0^inf <dS dS> dt / (gamma^2 (kT)^2) = 1/(2G) + 2G/w0^2   = tau_two_pole_stress

Checked at Gamma/omega0 = 0.1, 1, 5, 10 to 1e-6, plus the limits: both
kernels -> 1/(2 Gamma) underdamped, ratio stress/energy -> 4 deep overdamped.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from latvisc.viscosity import tau_two_pole_exact, tau_two_pole_stress


def _correlators(gamma, omega0=1.0):
    if abs(omega0**2 - gamma**2) < 1e-9 * omega0**2:
        gamma = gamma * (1.0 + 1e-6)
    if gamma < omega0:
        w1 = np.sqrt(omega0**2 - gamma**2)

        def c(t):
            return np.exp(-gamma * t) * (np.cos(w1 * t) + (gamma / w1) * np.sin(w1 * t))

        def cdot(t):
            return -(omega0**2 / w1) * np.exp(-gamma * t) * np.sin(w1 * t)
    else:
        kappa = np.sqrt(gamma**2 - omega0**2)
        ls, lf = gamma - kappa, gamma + kappa

        def c(t):
            return 0.5 * ((1 + gamma / kappa) * np.exp(-ls * t) + (1 - gamma / kappa) * np.exp(-lf * t))

        def cdot(t):
            return -(omega0**2 / (2 * kappa)) * (np.exp(-ls * t) - np.exp(-lf * t))
    return c, cdot


def _integrals(gamma, omega0=1.0):
    c, cdot = _correlators(gamma, omega0)

    def energy(t):
        cc, cd = c(t), cdot(t)
        cvv = (2 * gamma * cd + omega0**2 * cc) / omega0**2
        cvx = cd / omega0
        return 0.5 * (cvv**2 + cc**2 + 2 * cvx**2)

    def stress(t):
        return 2.0 * c(t) ** 2

    tmax = 60.0 / min(gamma, omega0**2 / gamma)
    pts = [1.0 / max(gamma, omega0)]
    return (quad(energy, 0, tmax, limit=4000, points=pts)[0],
            quad(stress, 0, tmax, limit=4000, points=pts)[0])


@pytest.mark.parametrize("ratio", [0.1, 1.0, 5.0, 10.0])
def test_closed_forms_match_direct_integration(ratio):
    ie, is_ = _integrals(ratio)
    assert ie == pytest.approx(float(tau_two_pole_exact(1.0, ratio)), rel=1e-6)
    assert is_ == pytest.approx(float(tau_two_pole_stress(1.0, ratio)), rel=1e-6)


def test_limits():
    omega, lw = 1.0e13, 1.0e10
    assert tau_two_pole_stress(omega, lw) == pytest.approx(1.0 / (2 * lw), rel=1e-5)
    assert tau_two_pole_exact(omega, lw) == pytest.approx(1.0 / (2 * lw), rel=1e-5)
    omega, lw = 1.0e10, 1.0e13
    assert tau_two_pole_stress(omega, lw) / tau_two_pole_exact(omega, lw) == pytest.approx(4.0, rel=1e-5)
    assert tau_two_pole_stress(omega, lw) == pytest.approx(2 * lw / omega**2, rel=1e-5)
