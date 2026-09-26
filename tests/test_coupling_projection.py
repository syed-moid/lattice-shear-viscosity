"""Projected coupling construction (latvisc.coupling): Hellmann-Feynman, degenerate-block gauge
invariance, and the engineering-shear identity on a toy model."""

import numpy as np

from latvisc.coupling import (
    ENGINEERING_SHEAR_PER_SYMMETRIC_STRAIN, diagonalise, engineering_shear, gruneisen_from_coupling,
    multiplet_groups, project_coupling, strain_derivative_matrix,
)

rng = np.random.default_rng(7)


def random_hermitian(n):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (a + a.conj().T)


def test_hellmann_feynman_singlets():
    n = 6
    D0 = np.diag(np.array([10.0, 40.0, 90.0, 160.0, 250.0, 360.0]) ** 2)   # distinct omega^2
    U = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    D0 = U @ D0 @ U.conj().T
    K = random_hermitian(n) * 100.0
    h = 1e-4
    w0, E = diagonalise(D0)
    # exact derivative of the eigenvalues at h -> 0 (first order, non-degenerate): e^dagger K e
    exact = np.real(np.einsum("in,ij,jn->n", E.conj(), K, E))
    lam = project_coupling(K, E, w0)["lam"]
    assert np.abs(lam - exact).max() < 1e-10 * np.abs(exact).max()
    # the same through the strained pair (the matrix difference cancels D0 up to rounding of order |D0| eps / h)
    lam_fd = project_coupling(strain_derivative_matrix(D0 + h * K, D0 - h * K, h), E, w0)["lam"]
    assert np.abs(lam_fd - exact).max() < 1e-7 * np.abs(exact).max()
    # finite-difference eigenvalues agree to O(h^2)
    wp, _ = diagonalise(D0 + h * K)
    wm, _ = diagonalise(D0 - h * K)
    fd = (np.sign(wp) * wp ** 2 - np.sign(wm) * wm ** 2) / (2 * h)
    assert np.abs(fd - exact).max() < 1e-6 * np.abs(exact).max()


def test_degenerate_block_gauge_invariance():
    n = 5
    D0 = np.diag([50.0 ** 2, 50.0 ** 2, 50.0 ** 2, 200.0 ** 2, 300.0 ** 2]).astype(complex)
    K = random_hermitian(n) * 1e3
    w0, E = diagonalise(D0)
    # any unitary rotation inside the degenerate triplet is an equally valid eigenbasis
    R = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
    E2 = E.copy()
    E2[:, :3] = E[:, :3] @ R
    p1 = project_coupling(K, E, w0)
    p2 = project_coupling(K, E2, w0)
    assert np.allclose(p1["lam"], p2["lam"], atol=1e-9)
    assert np.allclose(p1["trk2"], p2["trk2"], atol=1e-9)
    assert not np.allclose(p1["diag"][:3], p2["diag"][:3])          # the diagonal elements are gauge dependent
    assert p1["group_size"][:3].tolist() == [3, 3, 3] and p1["group_size"][3] == 1
    # the multiplet eigenvalues are those of the projected block
    block = E[:, :3].conj().T @ K @ E[:, :3]
    assert np.allclose(np.sort(np.linalg.eigvalsh(block)), np.sort(p1["lam"][:3]))
    assert np.isclose(p1["trk2"][0], np.sum(np.abs(block) ** 2))


def test_engineering_shear_identity_toy():
    # toy: omega^2(eps) = w0^2 (1 + c (eps_xy + eps_yx)) -> d omega^2 / d eps_xy = w0^2 c = d omega^2 / dh
    w0, c = 30.0, -4.0
    s = 0.005
    h = engineering_shear(s)
    assert h == 2 * s and ENGINEERING_SHEAR_PER_SYMMETRIC_STRAIN == 2.0
    d_plus = np.array([[w0 ** 2 * (1 + c * h)]])
    d_minus = np.array([[w0 ** 2 * (1 - c * h)]])
    lam = strain_derivative_matrix(d_plus, d_minus, h)[0, 0].real
    assert np.isclose(lam, w0 ** 2 * c)
    assert np.isclose(gruneisen_from_coupling(lam, w0), -c / 2)
    # the historical symmetric-path route (divide the path derivative by two) gives the same number
    d_path = (d_plus - d_minus)[0, 0].real / (2 * s)
    assert np.isclose(d_path / 2, lam)


def test_multiplet_groups():
    assert multiplet_groups(np.array([1.0, 1.2, 5.0, 5.3, 5.6, 20.0]), 0.5) == [[0, 1], [2, 3, 4], [5]]
