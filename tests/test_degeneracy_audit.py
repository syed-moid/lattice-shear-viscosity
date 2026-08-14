"""Regression tests for the degeneracy audit's gauge-check machinery.

Pinned here:

1. reconstruct_dynmat is the exact inverse of eigendecomposition when the
   eigenbasis is complete;
2. gauge_check recovers the exact perturbation eigenvalues for a
   degenerate doublet with a known, analytically solvable strain
   perturbation (Sigma mu^2 = Sigma D_exact^2);
3. gauge invariance: mixing the reference doublet eigenvectors by an
   arbitrary unitary rotation (the gauge freedom the audit exists to
   bound) leaves the projected-block eigenvalue-squared sum unchanged;
4. the projection uses the rows-as-bras convention — the
   conjugate-transposed variant silently agrees on real representations
   and diverges on complex ones, so the doublet here is built complex.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import audit_degeneracy_top20 as audit  # noqa: E402

NAT = 2
DIM = 3 * NAT
MASSES = np.ones(NAT)
RNG = np.random.default_rng(20260814)


def _random_unitary(dim):
    a = RNG.normal(size=(dim, dim)) + 1j * RNG.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    return q


def _modes_entry(dyn):
    """Eigendecompose a Hermitian dynamical matrix into a read_modes-style
    (q, freqs_cm1, vectors) entry (unit masses)."""
    lam, vecs = np.linalg.eigh(dyn)
    freqs = np.sign(lam) * np.sqrt(np.abs(lam))
    order = np.argsort(freqs)
    freqs = freqs[order]
    # rows-as-bras convention used throughout the audit: row s is the
    # conjugate of eigencolumn s, so Dyn = Z^dag Lambda Z exactly
    z = vecs[:, order].T.conj()
    return (np.zeros(3), freqs, z.reshape(DIM, NAT, 3))


def _doublet_system():
    """Dyn0 with an exact doublet (eigenvalue 100^2 twice) in a complex
    representation, plus a perturbation diagonal on the doublet (exact
    first-order eigenvalues d1, d2, no coupling out of the subspace)."""
    u = _random_unitary(DIM)
    lam0 = np.array([100.0**2, 100.0**2, 250.0**2, 300.0**2, 420.0**2, 500.0**2])
    dyn0 = u.conj().T @ np.diag(lam0) @ u
    d_exact = np.array([-3.0e4, 2.4e5])
    pert = (u.conj().T[:, :2] * d_exact) @ u[:2, :]
    return dyn0, pert, u, d_exact


def test_reconstruct_dynmat_roundtrip():
    dyn0, _, _, _ = _doublet_system()
    q, freqs, vecs = _modes_entry(dyn0)
    recon = audit.reconstruct_dynmat(freqs, vecs, MASSES)
    assert np.abs(recon - dyn0).max() < 1e-8 * np.abs(dyn0).max()


def test_gauge_check_recovers_exact_perturbation_eigenvalues():
    dyn0, pert, _, d_exact = _doublet_system()
    h = audit.H
    reference = [_modes_entry(dyn0)]
    plus = [_modes_entry(dyn0 + h * pert)]
    minus = [_modes_entry(dyn0 - h * pert)]
    block_sum, eigvals = audit.gauge_check(0, [1, 2], reference, plus, minus,
                                           MASSES)
    assert block_sum == pytest.approx(float(np.sum(d_exact**2)), rel=1e-6)
    assert sorted(eigvals) == pytest.approx(sorted(d_exact), rel=1e-6)


def test_gauge_check_invariant_under_doublet_rotation():
    dyn0, pert, _, d_exact = _doublet_system()
    h = audit.H
    plus = [_modes_entry(dyn0 + h * pert)]
    minus = [_modes_entry(dyn0 - h * pert)]

    q, freqs, vecs = _modes_entry(dyn0)
    z = vecs.reshape(DIM, DIM)
    mix = _random_unitary(2)
    z_rot = z.copy()
    z_rot[:2] = mix @ z[:2]  # arbitrary gauge inside the doublet
    rotated = [(q, freqs, z_rot.reshape(DIM, NAT, 3))]

    block_sum, _ = audit.gauge_check(0, [1, 2], rotated, plus, minus, MASSES)
    assert block_sum == pytest.approx(float(np.sum(d_exact**2)), rel=1e-6)


def test_group_by_tolerance_chains():
    values = np.array([77.2, 77.9, 99.3, 100.8, 300.0])
    groups = audit.group_by_tolerance(values, 2.0)
    members = sorted(sorted(g) for g in groups)
    assert members == [[0, 1], [2, 3], [4]]
