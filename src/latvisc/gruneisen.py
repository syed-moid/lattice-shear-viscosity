"""Finite-strain mode Grueneisen parameters.

The mode Grueneisen tensor component for mode (q, s) and strain epsilon_ij is

    gruneisen_ij(q, s) = - (1 / omega(q, s)) * d omega(q, s) / d epsilon_ij

evaluated here by central finite differences from two strained phonon runs
(+delta and -delta applied to the same strain component).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "mode_gruneisen_finite_strain",
    "mode_gruneisen_volume",
    "orthonormal_eigenvectors",
    "match_modes_by_overlap",
    "match_strain_pair_by_overlap",
    "match_four_strains_by_overlap",
]


def mode_gruneisen_finite_strain(omega_reference, omega_plus, omega_minus, strain_amplitude):
    """Central-difference mode Grueneisen parameter per mode.

    Parameters
    ----------
    omega_reference : array_like
        Unstrained mode frequencies (any consistent unit).
    omega_plus, omega_minus : array_like
        Mode frequencies at strain +delta and -delta. Branches must be
        matched (same mode ordering) across the three runs.
    strain_amplitude : float
        The strain delta (dimensionless).

    Returns
    -------
    ndarray
        Dimensionless gruneisen_ij per mode, expected O(1).
    """
    omega_reference = np.asarray(omega_reference, dtype=float)
    domega = (np.asarray(omega_plus, dtype=float) - np.asarray(omega_minus, dtype=float)) / (
        2.0 * float(strain_amplitude)
    )
    return -domega / omega_reference


def mode_gruneisen_volume(omega_plus, omega_minus, volume_plus, volume_minus):
    """Volumetric (thermodynamic) mode Grueneisen from two volume-scaled runs.

    gruneisen = - d ln omega / d ln V, central difference.
    """
    dlnomega = np.log(np.asarray(omega_plus, dtype=float)) - np.log(np.asarray(omega_minus, dtype=float))
    dlnvolume = np.log(float(volume_plus)) - np.log(float(volume_minus))
    return -dlnomega / dlnvolume


def orthonormal_eigenvectors(vectors, masses):
    """Mass-weight printed eigendisplacements and normalise per mode.

    vectors : complex (nmodes, nat, 3); masses : (nat,).
    Returns complex (nmodes, nat*3) rows of unit norm.
    """
    vectors = np.asarray(vectors, dtype=complex)
    z = vectors * np.sqrt(np.asarray(masses, dtype=float))[None, :, None]
    flat = z.reshape(z.shape[0], -1)
    return flat / np.linalg.norm(flat, axis=1)[:, None]


def _degenerate_groups(freq_reference, degeneracy_tol):
    n = len(freq_reference)
    groups, current = [], [0]
    for i in range(1, n):
        if abs(freq_reference[i] - freq_reference[current[-1]]) < degeneracy_tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def _match_by_overlap_core(freq_reference, z_reference, freq_strained, z_strained, degeneracy_tol):
    """Core of match_modes_by_overlap, operating on already-orthonormalized
    z-vectors (see orthonormal_eigenvectors) instead of raw eigendisplacements
    — shared with match_four_strains_by_overlap's second matching stage,
    which already has z-vectors on hand and must not re-derive them from a
    reordered (and no longer raw-shaped) intermediate array."""
    overlap = np.abs(z_reference.conj() @ z_strained.T)
    n = len(freq_reference)
    groups = _degenerate_groups(freq_reference, degeneracy_tol)

    matched = np.full(n, np.nan)
    taken = np.zeros(n, dtype=bool)
    for group in sorted(groups, key=len, reverse=True):
        candidates = np.where(~taken)[0]
        block = overlap[np.ix_(group, candidates)].sum(axis=0)
        chosen = candidates[np.argsort(block)[::-1][: len(group)]]
        matched[np.array(group)] = np.sort(freq_strained[chosen])
        taken[chosen] = True
    return matched, overlap


def match_modes_by_overlap(freq_reference, vectors_reference, freq_strained,
                           vectors_strained, masses, degeneracy_tol=0.5):
    """Reorder strained frequencies onto reference mode indices.

    Modes are matched by |<z_ref | z_strained>| with the mass-weighted
    metric — never by frequency ordering, which band crossings corrupt
    silently. Degenerate reference subspaces (spacing < degeneracy_tol)
    are matched as a block by total overlap and sorted internally, which
    yields the basis-invariant combinations.

    Suitable for matching a SINGLE strained calculation against the
    reference (e.g. quality/overlap reporting). For a +eps/-eps strain
    PAIR, use match_strain_pair_by_overlap instead: independently sorting
    each strain sign within a degenerate subspace can silently swap which
    physical branch sits at a given output position between +eps and -eps,
    fabricating a spurious curvature signal (see match_strain_pair_by_overlap).

    Returns (matched_frequencies, overlap_matrix).
    """
    z_reference = orthonormal_eigenvectors(vectors_reference, masses)
    z_strained = orthonormal_eigenvectors(vectors_strained, masses)
    freq_reference = np.asarray(freq_reference, dtype=float)
    freq_strained = np.asarray(freq_strained, dtype=float)
    return _match_by_overlap_core(freq_reference, z_reference, freq_strained, z_strained, degeneracy_tol)


def _pair_by_mutual_overlap(freq_reference, z_reference, freq_a, z_a, freq_b, z_b, groups):
    """Shared block-matching core for a strain-sign PAIR (same magnitude,
    opposite sign, or otherwise directly overlap-comparable to each other).

    For each degenerate reference group, candidate branches are selected
    from `a` and `b` independently by total overlap with the reference
    group, then paired bijectively by DIRECT mutual overlap between `a`
    and `b` (the physically correct correspondence within a degenerate
    subspace — see match_strain_pair_by_overlap's docstring). Non-degenerate
    groups (size 1) are matched independently against the reference.

    Returns (chosen_a, chosen_b): int index arrays of length
    len(freq_reference), giving the branch index into freq_a/freq_b for
    each reference-group slot (ordered by ascending freq_a within each
    group — a fixed, deterministic output convention, not itself a claim
    of physical branch identity beyond "the same analytic branch in a
    and b").
    """
    overlap_ref_a = np.abs(z_reference.conj() @ z_a.T)
    overlap_ref_b = np.abs(z_reference.conj() @ z_b.T)
    n = len(freq_reference)
    chosen_a_out = np.full(n, -1, dtype=int)
    chosen_b_out = np.full(n, -1, dtype=int)
    taken_a = np.zeros(n, dtype=bool)
    taken_b = np.zeros(n, dtype=bool)

    for group in sorted(groups, key=len, reverse=True):
        group = np.array(group)
        cand_a = np.where(~taken_a)[0]
        cand_b = np.where(~taken_b)[0]
        block_a = overlap_ref_a[np.ix_(group, cand_a)].sum(axis=0)
        block_b = overlap_ref_b[np.ix_(group, cand_b)].sum(axis=0)
        chosen_a = cand_a[np.argsort(block_a)[::-1][: len(group)]]
        chosen_b = cand_b[np.argsort(block_b)[::-1][: len(group)]]
        taken_a[chosen_a] = True
        taken_b[chosen_b] = True

        if len(group) == 1:
            chosen_a_out[group] = chosen_a
            chosen_b_out[group] = chosen_b
            continue

        cross = np.abs(z_a[chosen_a].conj() @ z_b[chosen_b].T)
        row_ind, col_ind = linear_sum_assignment(-cross)
        paired_a = chosen_a[row_ind]
        paired_b = chosen_b[col_ind]
        order = np.argsort(freq_a[paired_a])
        chosen_a_out[group] = paired_a[order]
        chosen_b_out[group] = paired_b[order]

    return chosen_a_out, chosen_b_out


def match_strain_pair_by_overlap(freq_reference, vectors_reference, freq_plus,
                                 vectors_plus, freq_minus, vectors_minus,
                                 masses, degeneracy_tol=0.5):
    """Pair +eps/-eps strained frequencies onto reference mode indices.

    Non-degenerate reference modes are matched to each strain sign
    independently by |<z_ref | z_strained>| overlap (unambiguous — a
    single mode has nowhere else to go).

    A degenerate reference subspace splits, to leading order in a linear
    perturbation, according to the eigenvectors of the perturbation matrix
    projected onto that subspace. Those eigenvectors are strain-sign
    independent (only their eigenvalues, and hence the frequency shifts,
    flip sign with the strain) — so the physically correct pairing between
    +eps and -eps within a degenerate group is set by DIRECT mutual overlap
    between the two strained calculations, not by independently sorting
    each strain's frequencies against the reference and pairing by output
    position. That independent-sort approach silently swaps which physical
    branch sits at a given position whenever the perturbation eigenvalues
    span both signs (the common case), fabricating a |eps| kink at eps=0
    that a central second difference reads as large spurious curvature.

    Returns (matched_plus, matched_minus): arrays of length
    len(freq_reference); matched_plus[i] and matched_minus[i] are always
    the same analytic branch.
    """
    z_reference = orthonormal_eigenvectors(vectors_reference, masses)
    z_plus = orthonormal_eigenvectors(vectors_plus, masses)
    z_minus = orthonormal_eigenvectors(vectors_minus, masses)

    freq_plus = np.asarray(freq_plus, dtype=float)
    freq_minus = np.asarray(freq_minus, dtype=float)
    freq_reference = np.asarray(freq_reference, dtype=float)
    groups = _degenerate_groups(freq_reference, degeneracy_tol)

    chosen_plus, chosen_minus = _pair_by_mutual_overlap(
        freq_reference, z_reference, freq_plus, z_plus, freq_minus, z_minus, groups
    )
    return freq_plus[chosen_plus], freq_minus[chosen_minus]


def match_four_strains_by_overlap(freq_reference, vectors_reference,
                                  freq_m010, vectors_m010,
                                  freq_m005, vectors_m005,
                                  freq_p005, vectors_p005,
                                  freq_p010, vectors_p010,
                                  masses, degeneracy_tol=0.5):
    """Match four strained calculations at the SAME strain direction, four
    magnitudes (-0.010, -0.005, +0.005, +0.010), onto reference mode
    indices — for a 4- or 5-point polynomial fit of omega(eps)/omega^2(eps)
    per mode (e.g. a symmetric eps05-vs-eps10 Richardson comparison).

    Non-degenerate reference modes are matched to each strain independently
    by overlap (unambiguous).

    A degenerate reference subspace splits, to leading order, according to
    the eigenvectors of the strain-direction perturbation projected onto
    the subspace — those eigenvectors are magnitude- AND sign-independent
    for a FIXED strain direction (only the eigenvalues/frequency shifts
    scale with magnitude and flip sign with strain sign). The canonical
    branch labeling is therefore established ONCE, from the +/-0.005 pair
    (smallest magnitude, most reliable direct mutual overlap — exactly
    match_strain_pair_by_overlap's method), and then PROPAGATED to +0.010
    and -0.010 by matching each against its same-sign 0.005 sibling (the
    smallest strain-magnitude separation available, hence the most
    reliable overlap) rather than independently re-matched against the
    still-degenerate bare reference — that would reintroduce exactly the
    sign-flip mislabeling bug match_strain_pair_by_overlap exists to avoid,
    now with a magnitude mismatch on top. Once the +/-0.005 canonical
    labeling has lifted the reference's degeneracy (real strain generically
    splits a degenerate subspace), matching +/-0.010 against its own-sign
    0.005 sibling is an ordinary, unambiguous overlap match — no further
    degenerate-subspace machinery is needed at that step.

    Returns (matched_m010, matched_m005, matched_p005, matched_p010): four
    arrays of length len(freq_reference); index i is the same analytic
    branch in all four (and corresponds to reference branch i).
    """
    freq_reference = np.asarray(freq_reference, dtype=float)
    freq_m010 = np.asarray(freq_m010, dtype=float)
    freq_m005 = np.asarray(freq_m005, dtype=float)
    freq_p005 = np.asarray(freq_p005, dtype=float)
    freq_p010 = np.asarray(freq_p010, dtype=float)

    z_reference = orthonormal_eigenvectors(vectors_reference, masses)
    z_m010 = orthonormal_eigenvectors(vectors_m010, masses)
    z_m005 = orthonormal_eigenvectors(vectors_m005, masses)
    z_p005 = orthonormal_eigenvectors(vectors_p005, masses)
    z_p010 = orthonormal_eigenvectors(vectors_p010, masses)

    groups = _degenerate_groups(freq_reference, degeneracy_tol)

    # Step 1: canonical labeling from the +/-0.005 pair.
    chosen_p005, chosen_m005 = _pair_by_mutual_overlap(
        freq_reference, z_reference, freq_p005, z_p005, freq_m005, z_m005, groups
    )
    freq_p005_canon = freq_p005[chosen_p005]
    freq_m005_canon = freq_m005[chosen_m005]
    z_p005_canon = z_p005[chosen_p005]
    z_m005_canon = z_m005[chosen_m005]

    # Step 2: propagate to +/-0.010 via its own-sign 0.005 sibling. By this
    # point the canonical 0.005 frequencies are generically non-degenerate
    # (real strain splits the subspace), so ordinary overlap matching
    # applies without further block machinery. z_p005_canon/z_m005_canon
    # are already-orthonormalized z-vectors (reordered from z_p005/z_m005),
    # not raw eigendisplacements, so this goes through the shared core
    # directly rather than match_modes_by_overlap (which expects raw
    # vectors and would re-derive z incorrectly from this reordered array).
    matched_p010, _ = _match_by_overlap_core(
        freq_p005_canon, z_p005_canon, freq_p010, z_p010, degeneracy_tol
    )
    matched_m010, _ = _match_by_overlap_core(
        freq_m005_canon, z_m005_canon, freq_m010, z_m010, degeneracy_tol
    )

    return matched_m010, freq_m005_canon, freq_p005_canon, matched_p010
