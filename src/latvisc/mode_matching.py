"""Eigenvector matching between a bare and a renormalised phonon set on one mesh.

Used to assign each bare (q, s) mode of the strained-cell dataset the
renormalised frequency of the physically corresponding mode at the same q,
by the overlap of orthonormal polarisation vectors rather than by frequency
rank. Degenerate multiplets of the renormalised set (|d omega| < tol) are
treated as a subspace: the reported overlap is summed over the multiplet and
the assigned frequency is the multiplet mean, so exact degeneracies do not
read as poor matches.
"""

from __future__ import annotations

import numpy as np

__all__ = ["match_to_renormalised"]


def match_to_renormalised(z_bare, omega_ren, z_ren, degeneracy_tol=0.5):
    """Match bare modes to renormalised modes at one q-point.

    Parameters
    ----------
    z_bare : complex (n_bare, ndof)
        Orthonormal polarisation vectors of the bare modes (rows).
    omega_ren : (n_ren,)
        Renormalised frequencies (any unit; the tolerance is in the same unit).
    z_ren : complex (n_ren, ndof)
        Orthonormal polarisation vectors of the renormalised modes (rows), same
        atom order and phase convention as z_bare.
    degeneracy_tol : float
        Modes of the renormalised set closer than this are one multiplet.

    Returns
    -------
    omega_matched : (n_bare,)  renormalised frequency assigned to each bare mode
    overlap : (n_bare,)        |<z_bare|z_ren>|^2 summed over the matched multiplet
    index : (n_bare,) int      index of the best single renormalised mode
    """
    z_bare = np.asarray(z_bare, dtype=complex)
    z_ren = np.asarray(z_ren, dtype=complex)
    omega_ren = np.asarray(omega_ren, dtype=float)
    ov = np.abs(z_bare.conj() @ z_ren.T) ** 2
    index = ov.argmax(axis=1)
    omega_matched = np.empty(len(z_bare))
    overlap = np.empty(len(z_bare))
    for i, j in enumerate(index):
        grp = np.abs(omega_ren - omega_ren[j]) < degeneracy_tol
        overlap[i] = ov[i, grp].sum()
        omega_matched[i] = omega_ren[grp].mean()
    return omega_matched, overlap, index
