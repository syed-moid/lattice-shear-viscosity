"""Acoustic limit of the projected coupling on a synthetic elastic-wave set: Lambda ~ q^2 with a
finite Grueneisen parameter as q -> 0 (latvisc.coupling.acoustic_limit_check)."""

import numpy as np

from latvisc.coupling import acoustic_limit_check

# one atom per cell, cubic elastic medium: D(q) = (c44 q^2) 1 + (c11 - c44) q q^T  (cm^-2 units, q reduced)
C11, C44 = 8.0e4, 3.0e4
MASS = np.array([50.0])


def dyn(q, h):
    q = np.asarray(q, dtype=float)
    d = C44 * (q @ q) * np.eye(3) + (C11 - C44) * np.outer(q, q)
    # shear h softens the [1-10]-polarised wave and stiffens the longitudinal one (toy third-order terms)
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    d = d - 2.0 * 3.0 * h * C44 * (q @ q) * np.outer(e1, e1) + 2.0 * 0.5 * h * C11 * np.outer(q, q)
    return d.astype(complex)


def test_lambda_scales_as_q2_with_finite_gamma():
    h = 0.01
    rows = acoustic_limit_check(lambda q: dyn(q, 0.0), lambda q: dyn(q, h), lambda q: dyn(q, -h), h, MASS,
                                direction=(1, 1, 0), magnitudes=(0.05, 0.1, 0.2))
    by = {}
    for r in rows:
        by.setdefault((r["character"], r["mode"]), []).append(r)
    # every branch: Lambda / q^2 constant and gamma finite and q independent
    for key, rs in by.items():
        l2 = [r["lam_over_q2"] for r in rs]
        g = [r["gamma"] for r in rs]
        assert np.allclose(l2, l2[0], rtol=1e-8), key
        assert np.allclose(g, g[0], rtol=1e-8), key
    la = [r for r in rows if r["character"] == "LA"]
    assert all(np.isclose(r["gamma"], -0.5, rtol=1e-8) for r in la)      # omega^2 (1 + 1.0 h) -> gamma = -0.5
    ta_soft = [r for r in rows if r["character"] == "TA" and np.isclose(r["gamma"], 3.0, rtol=1e-8)]
    assert len(ta_soft) == 3                                              # the [1-10] TA branch: gamma = +3
    ta_z = [r for r in rows if r["character"] == "TA" and abs(r["gamma"]) < 1e-8]
    assert len(ta_z) == 3                                                 # the z-polarised TA: no first-order coupling
