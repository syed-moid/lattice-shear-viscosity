"""Eigenvector-overlap mode matching: the band-crossing guardrail.

Frequency-ordered matching silently corrupts Grueneisen parameters when
branches cross under strain; overlap matching must recover the correct
assignment. Synthetic 3-mode system with distinct eigenvectors.
"""

import numpy as np
import pytest

from latvisc.gruneisen import (
    match_four_strains_by_overlap,
    match_modes_by_overlap,
    match_strain_pair_by_overlap,
    orthonormal_eigenvectors,
)

MASSES = np.array([2.0, 1.0])  # 2 atoms -> 6 dof, we use 3 modes for clarity


def _modes(vectors):
    """Pack rows of shape (nat*3,) into (nmodes, nat, 3) eigendisplacements
    (divide by sqrt(mass) to mimic what QE prints)."""
    arr = np.array(vectors, dtype=complex).reshape(len(vectors), 2, 3)
    return arr / np.sqrt(MASSES)[None, :, None]


E1 = [1, 0, 0, 0, 0, 0]
E2 = [0, 1, 0, 0, 0, 0]
E3 = [0, 0, 1, 0, 0, 0]


def test_band_crossing_recovered():
    """Strained modes 2 and 3 swap frequency order; overlap matching must
    un-swap them, frequency ordering would not."""
    freq_ref = np.array([100.0, 200.0, 210.0])
    vec_ref = _modes([E1, E2, E3])
    # under strain: mode with E2 character hardens past the E3 mode
    freq_str = np.array([101.0, 205.0, 215.0])
    vec_str = _modes([E1, E3, E2])  # sorted by frequency: E3 now below E2
    matched, overlap = match_modes_by_overlap(freq_ref, vec_ref, freq_str, vec_str, MASSES)
    assert matched[0] == pytest.approx(101.0)
    assert matched[1] == pytest.approx(215.0)  # E2 character -> 215, NOT 205
    assert matched[2] == pytest.approx(205.0)  # E3 character -> 205
    assert overlap.max(axis=1).min() > 0.99


def test_degenerate_subspace_invariant_split():
    """A reference doublet splitting under strain is matched as a block and
    reported sorted (the invariant combination), regardless of the strained
    eigenvector mixing within the subspace."""
    freq_ref = np.array([150.0, 150.0, 400.0])
    vec_ref = _modes([E1, E2, E3])
    # strained doublet splits and mixes internally by 45 degrees
    s = 1 / np.sqrt(2)
    mixed_a = [s, s, 0, 0, 0, 0]
    mixed_b = [s, -s, 0, 0, 0, 0]
    freq_str = np.array([145.0, 155.0, 401.0])
    vec_str = _modes([mixed_a, mixed_b, E3])
    matched, _ = match_modes_by_overlap(freq_ref, vec_ref, freq_str, vec_str, MASSES)
    assert matched[0] == pytest.approx(145.0)
    assert matched[1] == pytest.approx(155.0)
    assert matched[2] == pytest.approx(401.0)


def test_orthonormalisation_metric():
    z = orthonormal_eigenvectors(_modes([E1, E2, E3]), MASSES)
    gram = np.abs(z.conj() @ z.T)
    assert np.allclose(gram, np.eye(3), atol=1e-12)


# Shared synthetic system for the strain-sign-flip tests below: a reference
# doublet (modes 0,1 at 150) whose degenerate-perturbation eigenvectors are
# v1=(E1+E2)/sqrt(2) (eigenvalue +1000) and v2=(E1-E2)/sqrt(2) (eigenvalue
# -1000), plus a non-degenerate control mode (E3 at 400, eigenvalue +200).
# Because the two eigenvalues have OPPOSITE SIGN, which branch sits at the
# lower frequency position flips between +eps and -eps -- independently
# sorting each strain against the reference would silently swap v1 and v2
# at that crossover, exactly the bug these matchers exist to avoid.
_S = 1 / np.sqrt(2)
_V1 = [_S, _S, 0, 0, 0, 0]
_V2 = [_S, -_S, 0, 0, 0, 0]
FREQ_REF_DOUBLET = np.array([150.0, 150.0, 400.0])
VEC_REF_DOUBLET = _modes([E1, E2, E3])


def _doublet_strained(eps):
    """omega_v1(eps) = 150 + 1000*eps, omega_v2(eps) = 150 - 1000*eps,
    omega_E3(eps) = 400 + 200*eps. Returns (freq, vec) with modes listed in
    ascending-frequency order, as a real QE-style output would present them."""
    f_v1 = 150.0 + 1000.0 * eps
    f_v2 = 150.0 - 1000.0 * eps
    f_e3 = 400.0 + 200.0 * eps
    if f_v1 <= f_v2:
        freq = np.array([f_v1, f_v2, f_e3])
        vec = _modes([_V1, _V2, E3])
    else:
        freq = np.array([f_v2, f_v1, f_e3])
        vec = _modes([_V2, _V1, E3])
    return freq, vec


def test_strain_pair_survives_degenerate_sign_flip():
    """match_strain_pair_by_overlap must track v1 (slope +1000) and v2
    (slope -1000) as fixed branches across +/-0.005, not swap them at the
    sign-flip crossover that independent per-strain sorting would hit."""
    freq_p, vec_p = _doublet_strained(0.005)
    freq_m, vec_m = _doublet_strained(-0.005)
    matched_p, matched_m = match_strain_pair_by_overlap(
        FREQ_REF_DOUBLET, VEC_REF_DOUBLET, freq_p, vec_p, freq_m, vec_m, MASSES
    )
    # one branch must be (155, 145) [v1: 150+/-5] and the other (145, 155) [v2],
    # matched consistently -- NOT both reading the same (145,145)/(155,155)
    # that a position-based mismatch would produce.
    pairs = sorted(zip(matched_p.tolist(), matched_m.tolist()))
    assert pairs[0] == pytest.approx((145.0, 155.0))  # v2: +eps lower, -eps higher
    assert pairs[1] == pytest.approx((155.0, 145.0))  # v1: +eps higher, -eps lower
    assert matched_p[2] == pytest.approx(401.0)  # non-degenerate control unaffected
    assert matched_m[2] == pytest.approx(399.0)


def test_four_strains_track_linear_branches_through_sign_flip():
    """match_four_strains_by_overlap must recover PERFECTLY LINEAR omega(eps)
    for both v1 (slope +1000) and v2 (slope -1000) across all four strain
    points, despite the ascending-frequency position of each branch flipping
    between negative and positive eps. A naive independent-sort-per-strain
    would instead produce a garbled, non-monotonic sequence at whichever
    position crosses over."""
    freq_m010, vec_m010 = _doublet_strained(-0.010)
    freq_m005, vec_m005 = _doublet_strained(-0.005)
    freq_p005, vec_p005 = _doublet_strained(0.005)
    freq_p010, vec_p010 = _doublet_strained(0.010)

    m010, m005, p005, p010 = match_four_strains_by_overlap(
        FREQ_REF_DOUBLET, VEC_REF_DOUBLET,
        freq_m010, vec_m010, freq_m005, vec_m005,
        freq_p005, vec_p005, freq_p010, vec_p010,
        MASSES,
    )

    # identify which output branch is v1 (positive slope) vs v2 (negative)
    # by its p010 vs m010 ordering, then check exact linearity for each.
    for branch in (0, 1):
        eps = np.array([-0.010, -0.005, 0.005, 0.010])
        vals = np.array([m010[branch], m005[branch], p005[branch], p010[branch]])
        slope = (vals[-1] - vals[0]) / (eps[-1] - eps[0])
        assert abs(slope) == pytest.approx(1000.0, rel=1e-6), (
            f"branch {branch} is not perfectly linear (slope {slope}) -- "
            "the four points were not tracked as the same analytic branch"
        )
        # central value at eps=0 (extrapolated) must match the reference 150
        intercept = vals[1] + (vals[2] - vals[1]) * (0 - eps[1]) / (eps[2] - eps[1])
        assert intercept == pytest.approx(150.0, abs=1e-6)

    # the non-degenerate control branch (index 2) must also be exactly linear
    eps = np.array([-0.010, -0.005, 0.005, 0.010])
    vals = np.array([m010[2], m005[2], p005[2], p010[2]])
    slope_e3 = (vals[-1] - vals[0]) / (eps[-1] - eps[0])
    assert slope_e3 == pytest.approx(200.0, rel=1e-6)
