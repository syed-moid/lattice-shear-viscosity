#!/usr/bin/env python3
"""Assemble the SrTiO3 lattice shear viscosity eta_xyxy(T).

PRODUCTION MODEL (revision): the hybrid model tut_z_od1 (ALAMODE example harmonic set re-expressed on the
4x4x4 supercell + QE-PBEsol strain perturbation; SCPH SELF_OFFDIAG = 1, correction mesh 2x2x2 / inner mesh
12x12x12), construction C — see the section "PRODUCTION MODEL" below (assemble_construction). Everything between here and that section is the
retained legacy machinery (tutorial surface, rank map, bare-coupling transfer = construction A) kept
runnable for the ledger rows; it no longer defines any production number.

LEGACY DESCRIPTION (Stage C, superseded):

Central formula (manuscript Eq. 5, project guardrails):

    eta = (1 / (V_cell N_q k_B T)) sum_qs (hbar*omega)^2 gamma_xy^2 n(n+1) tau

with the stress-correlator two-pole lifetime tau = 1/(2 Gamma) + 2 Gamma/omega^2
(latvisc.viscosity.tau_two_pole_stress: exact classical time integral of the
stiffness-conjugate stress correlator of an effective damped oscillator with
friction 2 Gamma; reduces to 1/(2 Gamma) underdamped, 2 Gamma/omega^2 deep
overdamped). gamma_xy is the TENSOR Grueneisen component
-d ln omega/d epsilon_xy of the nine-component convention, for which
eta_xyxy is the Newtonian eta_44 of the acoustic conversions; the strained
cells apply epsilon_xy = epsilon_yx = s and the conversion from the path
derivative (/ 2) is made once, in check_shear_nonlinearity.compute_dataset
(latvisc.gruneisen.path_to_tensor_shear), so D and Lambda below are tensor
derivatives. Units come out Pa s.

Partitioned pipeline (Route S / Route H, cutoff omega0 = 175 cm-1 —
Richardson-validated for SrTiO3, see GATE_1p.md):

  Route S (omega0 >= 175 cm-1): gamma_xy = -D/(2*omega0^2) from the
    eps05 central difference on the 11^3 mesh (D = d(omega^2)/d(eps),
    signed eigenvalues; step size validated by the 5-point Richardson
    analysis, richardson_5pt_SrTiO3.csv). Bare omega0 approximates the
    renormalized frequency well in this manifold; the weight factors
    still use the mapped renormalized frequency for consistency.

  Route H (omega0 < 175 cm-1, stable + unstable): gamma =
    Lambda/(2*omega_r^2(T)) with Lambda = D from the SAME strained cells
    and omega_r(T) the SCPH-renormalised frequency of the physically
    corresponding mode, found by EIGENVECTOR MATCHING on the production
    11^3 mesh (OMEGA_R_SOURCE = "direct"): the renormalised harmonic set of
    the production surface (SURFACE_TAG, see SURFACES) is diagonalised by
    anphon on the same 1331 q-points with PRINTEVEC, and each bare QE mode
    takes the frequency of the renormalised mode of maximum
    |<e_bare|e_ren>|^2 (multiplet-summed, latvisc.mode_matching). The
    production surface is the ALAMODE example surface (STO222.xml + the
    example's precomputed SCPH corrections, the surface of the submitted
    manuscript), since the own-surface SCPH did not pass every criterion of
    the production-surface gate; the own-surface variants are available as
    sensitivities. The former per-q rank-pairing eigenvalue map
    (OMEGA_R_SOURCE = "rankmap", build_maps) is kept as a sensitivity.

  Linewidths Gamma(T): per-mode SCPH-coupled Gamma_qs(T) from the own
    anphon MODE = RTA run on the 8^3 mesh of the same surface
    (STO_RTA_scph_<T>K.result for the production surface), carried to the 11^3 mesh by the
    character-aware frequency-class map (GAMMA_SOURCE = "freqmap": binned
    median vs omega_r for stable-character modes, soft-manifold median for
    bare-soft modes; the character split uses the bare RTA file of the same
    surface) or, as a sensitivity, by nearest-q assignment on the 8^3 mesh
    with branch matched by frequency (GAMMA_SOURCE = "nearest_q").

  Gamma-point soft sector: the unstable zone-center TO branches use the
    Vogt (1995) experimental soft-mode frequency and HWHM damping
    (softmode_inputs_SrTiO3.csv) where the digitized series covers T;
    outside that range they fall back to the mapped values and the row
    is flagged.

Reads : data/raw/gruneisen_modes/SrTiO3/*.modes,
        data/raw/alamode_sto/surface_tutorial/mesh{n}_renorm_tutorial_<T>K.npz (+ mesh11_bare.npz),
        data/raw/alamode_sto/STO_RTA_production.result, STO_RTA_scph_<T>K.result   (production)
        data/raw/alamode_sto/own_surface_<tag>/...                                   (sensitivities)
        data/processed/softmode_inputs_SrTiO3.csv
Writes: data/processed/eta_SrTiO3.csv
        data/processed/reports/eta_SrTiO3_stageC.md   (300 K provenance trace)

Usage: uv run python scripts/compute_eta_SrTiO3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.constants import hbar as HBAR
from scipy.constants import speed_of_light

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_shear_nonlinearity import MASSES, MODES_DIR, compute_dataset  # noqa: E402
from crosscheck_alamode_sto_tau import parse_result  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from latvisc.gruneisen import (  # noqa: E402
    mode_gruneisen_finite_strain, mode_gruneisen_volume, orthonormal_eigenvectors, path_to_tensor_shear,
)
from latvisc.mode_matching import match_to_renormalised  # noqa: E402
from latvisc.qe_modes import read_modes  # noqa: E402
from latvisc.viscosity import bose_einstein, tau_two_pole_stress  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ALAMODE_DIR = REPO / "data" / "raw" / "alamode_sto"
# Renormalised surfaces available for the direct (eigenvector-matched) omega_r.
#   tutorial : ALAMODE example harmonic set STO222.xml + the example's precomputed SCPH corrections
#              (the surface of the submitted manuscript); RTA linewidths = the production files
#              STO_RTA_scph_<T>K.result; meshes rebuilt by anphon on the 7.363 bohr cell, eigenvectors
#              permuted to the QE atom order.
#   own_i2s2 / own_i4s4 / own_i4s8 : own-surface SCPH (project's converted QE-PBEsol harmonic set +
#              the example's anharmonic set) at KMESH_INTERPOLATE/KMESH_SCPH = 2x2x2/2x2x2, 4x4x4/4x4x4,
#              4x4x4/8x8x8; own RTA on each. Kept as sensitivities (the 6.0 gate did not pass on all
#              four criteria: kappa(300 K) +13 % for 2x2x2, R-point AFD mode 35 / 18 cm-1 for the 4x4x4 sets).
SURFACES = {
    "tutorial": {"dir": ALAMODE_DIR / "surface_tutorial", "file_tag": "tutorial",
                 "rta_renorm": lambda T: ALAMODE_DIR / f"STO_RTA_scph_{T}K.result",
                 "rta_bare": ALAMODE_DIR / "STO_RTA_production.result"},
    **{f"own_{t}": {"dir": ALAMODE_DIR / f"own_surface_{t}", "file_tag": t,
                    "rta_renorm": (lambda t: (lambda T: ALAMODE_DIR / f"own_surface_{t}" / f"STO_RTA_own_{t}_{T}K.result"))(t),
                    "rta_bare": ALAMODE_DIR / f"own_surface_{t}" / f"STO_RTA_own_{t}_bare.result"}
      for t in ("i2s2", "i4s4", "i4s8")},
}
SURFACE_TAG = "tutorial"       # production surface (6.0 gate outcome)
OMEGA_R_SOURCE = "direct"      # "direct" (eigenvector-matched, production) | "rankmap" (sensitivity)
GAMMA_SOURCE = "freqmap"       # "freqmap" (production) | "nearest_q" (sensitivity)
MATCH_MODE = "one_step"        # "one_step": QE bare -> renormalised; "two_step": QE bare -> anphon bare -> renormalised
DEGEN_TOL = 0.5                # cm-1, multiplet grouping for the eigenvector matching and for construction A
EXACT_DEGEN_TOL = 1e-3         # cm-1, exact degeneracy = SCPH round-off scale (spread histogram empty in 1e-4..1e-2)
COUPLING_DEGEN_TOL = EXACT_DEGEN_TOL  # cm-1, EXACT degeneracy (SCPH round-off scale) for constructions B and C: the restricted
                               # derivative is diagonalised only inside exactly degenerate eigenspaces; every other mode
                               # gets the diagonal projection e^+ K e (first-order eigenvalue derivative). 0.5 = former
                               # grouping of genuinely split modes (sensitivity only).
CM1 = 2.0 * np.pi * speed_of_light * 100.0  # rad/s per cm-1

# PBEsol production cell (Section 3.1: a = 3.8930 Angstrom)
A_PBESOL = 3.8930e-10
V_CELL = A_PBESOL**3          # m^3, 5 atoms
N_Q = 11**3                    # our gamma mesh
CUTOFF_CM1 = 175.0             # Route S / Route H partition (Richardson-validated)
OMEGA_MIN = 5.0                # below this the bare mode is treated as unstable
# Soft-manifold character boundary for the Gamma assignment (audit finding
# A3, eta_SrTiO3_stageC.md): the bare [5,50) cm-1 manifold showed 98-100%
# curvature-flag pathology (it IS the soft manifold), and drawing its
# linewidths from the frequency-binned stable map handed it
# acoustic-contaminated Gamma ~ 0.6 cm-1 (tau ~ 5 ps) — same character-
# blindness bug as the original unstable-sector one, one shell further out.
SOFT_CHAR_CM1 = 50.0
TEMPS = [100, 150, 200, 250, 300, 350, 400]
GAMMA_BINS = 40                # frequency bins for the Gamma(omega) map
MESH_CONVERGENCE_BAND = 0.06   # |eta(11^3)/eta(13^3) - 1| reported against this (Table 2)
PAIR_OVERLAP_THRESHOLD = 0.9   # strain-pair tracking quality reported below this


def sanity_checks_legacy(rows, details, eta300: float) -> dict:
    """Physics checks replacing the former expectation-band gate: numbers,
    not a pass/fail band. Returns the dictionary it prints.

    (i)   convention invariance: a toy mode with a known tensor coupling
          returns the tensor Grueneisen parameter through the same
          functions as production (symmetric-shear path / 2), and the
          hydrostatic control is unchanged;
    (ii)  mesh convergence: 11^3 vs 13^3 from table1_convergence_SrTiO3.csv
          when present, against MESH_CONVERGENCE_BAND;
    (iii) positivity of every mode contribution and of the total;
    (iv)  mode-tracking quality: fraction of Route-H modes (and of their
          eta weight) whose strain-pair eigenvector overlap is below
          PAIR_OVERLAP_THRESHOLD (reported, not asserted);
    (v)   SCPH residual: locally converged SCPH vs the precomputed
          corrections, from reports/scph_local_residual_300K.csv when present;
    (vi)  strain-amplitude stability: eps05 vs eps10 Richardson summary from
          reports/richardson_5pt_SrTiO3.csv when present.
    """
    out = {}
    # (i)  toy: omega^2 = w0^2 + lam_t (eps_xy + eps_yx); the eigenvalue route of
    #      compute_dataset (path derivative / 2) is exact for this coupling
    w0, lam_t, s = 100.0, -2.0e4, 0.005
    wp, wm = np.sqrt(w0**2 + 2 * lam_t * s), np.sqrt(w0**2 - 2 * lam_t * s)
    d_t = float(path_to_tensor_shear((wp**2 - wm**2) / (2 * s)))
    out["toy_gamma_pipeline_over_tensor"] = (-d_t / (2 * w0**2)) / (-lam_t / (2 * w0**2))
    out["toy_gamma_omega_route_over_tensor"] = float(
        mode_gruneisen_finite_strain(w0, wp, wm, s, symmetric_shear_path=True)) / (-lam_t / (2 * w0**2))
    vp, vm = (1 + s) ** 3, (1 - s) ** 3
    out["toy_gamma_vol_pipeline_over_true"] = float(mode_gruneisen_volume(w0 * vp**-2.0, w0 * vm**-2.0, vp, vm)) / 2.0
    # (ii)
    conv = REPO / "data" / "processed" / "table1_convergence_SrTiO3.csv"
    out["mesh_11_vs_13_pct"] = float("nan")
    if conv.exists():
        vals = {}
        for line in conv.read_text().splitlines():
            if line.startswith('"'):
                lab, nq, eta_v = line.split(",")[:3]
                vals[int(nq)] = float(eta_v)
        if 1331 in vals and 2197 in vals:
            out["mesh_11_vs_13_pct"] = 100.0 * (vals[1331] / vals[2197] - 1.0)
    # (iii)
    contribs = np.array([d["eta_contrib"] for d in details])
    out["min_mode_contribution_Pas"] = float(contribs.min())
    out["all_contributions_positive"] = bool((contribs > 0).all() and eta300 > 0)
    # (iv)
    by_key = {(r["iq"], r["branch"]): r.get("pair_overlap", float("nan")) for r in rows}
    h = [d for d in details if d["sector"] != "routeS"]
    ov = np.array([by_key[(d["iq"], d["branch"])] for d in h])
    wt = np.array([d["eta_contrib"] for d in h])
    low = ov < PAIR_OVERLAP_THRESHOLD
    out["routeH_frac_modes_pair_overlap_lt_threshold"] = float(low.mean())
    out["routeH_frac_eta_pair_overlap_lt_threshold"] = float(wt[low].sum() / eta300)
    # (v)
    resid = REPO / "data" / "processed" / "reports" / "scph_local_residual_300K.csv"
    out["scph_local_vs_precomputed_max_dev_cm1"] = float("nan")
    out["scph_local_vs_precomputed_eta_dev_pct"] = float("nan")
    if resid.exists():
        for line in resid.read_text().splitlines():
            if line.startswith("max_abs_dev_cm1"):
                out["scph_local_vs_precomputed_max_dev_cm1"] = float(line.split(",")[1])
            if line.startswith("eta_dev_pct"):
                out["scph_local_vs_precomputed_eta_dev_pct"] = float(line.split(",")[1])
    # (vi)
    rich = REPO / "data" / "processed" / "reports" / "richardson_5pt_SrTiO3.csv"
    out["richardson_flag_above_cutoff_pct"] = float("nan")
    out["richardson_flag_below_cutoff_pct"] = float("nan")
    if rich.exists():
        for line in rich.read_text().splitlines():
            if "flag_above_pct=" in line:
                for tok in line.replace("#", "").split(","):
                    if "flag_above_pct=" in tok:
                        out["richardson_flag_above_cutoff_pct"] = float(tok.split("=")[1])
                    if "flag_below_pct=" in tok:
                        out["richardson_flag_below_cutoff_pct"] = float(tok.split("=")[1])
    print("\nSanity checks (numbers, no band):")
    print(f"  (i)   toy gamma (eigenvalue route)/tensor = {out['toy_gamma_pipeline_over_tensor']:.9f}; "
          f"omega route {out['toy_gamma_omega_route_over_tensor']:.6f} (sqrt-curvature truncation); "
          f"hydrostatic control = {out['toy_gamma_vol_pipeline_over_true']:.6f} (all must be 1)")
    print(f"  (ii)  mesh 11^3 vs 13^3: {out['mesh_11_vs_13_pct']:+.2f} % (reported band {100 * MESH_CONVERGENCE_BAND:.0f} %)")
    print(f"  (iii) all mode contributions positive: {out['all_contributions_positive']} "
          f"(min {out['min_mode_contribution_Pas']:.2e} Pa s)")
    print(f"  (iv)  Route-H strain-pair overlap < {PAIR_OVERLAP_THRESHOLD}: "
          f"{100 * out['routeH_frac_modes_pair_overlap_lt_threshold']:.2f} % of modes, "
          f"{100 * out['routeH_frac_eta_pair_overlap_lt_threshold']:.2f} % of eta")
    print(f"  (v)   SCPH local vs precomputed: max |d omega| {out['scph_local_vs_precomputed_max_dev_cm1']:.4f} cm-1, "
          f"eta {out['scph_local_vs_precomputed_eta_dev_pct']:+.3f} %")
    print(f"  (vi)  eps05 vs Richardson flag rate: above cutoff {out['richardson_flag_above_cutoff_pct']:.2f} %, "
          f"below {out['richardson_flag_below_cutoff_pct']:.2f} %")
    return out


def build_maps(temperature: int):
    """Bare->renormalized eigenvalue map and Gamma(omega_r) maps at one T.

    The Gamma assignment is CHARACTER-AWARE: a purely frequency-binned map
    would mix, in the 50-100 cm-1 window, long-lived acoustic modes
    (tau ~ 17 ps at 300 K) with the heavily damped soft-TO manifold, and
    the binned median would hand acoustic lifetimes to soft modes whose
    gamma is 10-100 — inflating eta by an order of magnitude (found and
    fixed during the first Stage-C assembly, see eta_SrTiO3_stageC.md).
    ALAMODE's formerly-bare-unstable modes therefore provide the
    soft-sector Gamma statistics separately, and are excluded from the
    stable-mode frequency bins.
    """
    freq_bare, _ = parse_result(ALAMODE_DIR / "STO_RTA_production.result",
                                target_temp=temperature)
    freq_ren, gamma_ren = parse_result(
        ALAMODE_DIR / f"STO_RTA_scph_{temperature}K.result", target_temp=temperature)

    # Bare<->renormalized correspondence is by PER-Q RANK PAIRING, not by
    # raw (q,branch) index: each file sorts branches by its OWN frequencies,
    # and at Gamma the bare file ranks the imaginary TO1 (-58.5 cm-1) BELOW
    # the acoustic zeros while the renormalized file ranks the acoustic
    # zeros below the renormalized TO1 (~175 cm-1) — raw index pairing then
    # maps bare TO1 -> 0, poisoning the entire unstable region of the
    # eigenvalue map (this was the root cause of the audit-A3 omega_r =
    # 39.9 cm-1 artifact). The exact-zero acoustic entries are excluded
    # from both sides before rank pairing (they are the only branches that
    # interleave differently between the two sortings).
    ZERO_TOL = 0.5  # cm-1
    by_q_bare: dict[int, list] = {}
    by_q_ren: dict[int, list] = {}
    for (q, b), w in freq_bare.items():
        by_q_bare.setdefault(q, []).append(w)
    for (q, b), w in freq_ren.items():
        by_q_ren.setdefault(q, []).append((w, gamma_ren.get((q, b))))

    lam_bare, lam_ren = [], []
    omegas_r, gammas = [], []           # stable-character modes only
    soft_gammas = []                     # soft-manifold character
    soft_floor_theory = float("nan")
    for q in by_q_bare:
        bare = sorted(w for w in by_q_bare[q] if abs(w) > ZERO_TOL)
        ren = sorted(((w, g) for w, g in by_q_ren.get(q, [])
                      if abs(w) > ZERO_TOL), key=lambda t: t[0])
        if len(bare) != len(ren):
            continue  # unexpected multiplicity mismatch — leave out of the maps
        for wb, (wr, g) in zip(bare, ren):
            if wr <= 0:
                continue
            lam_bare.append(np.sign(wb) * wb * wb)
            lam_ren.append(wr * wr)
            if q == 1 and wb < 0:
                # renormalized Gamma-point soft TO = the branch minimum
                soft_floor_theory = (wr if np.isnan(soft_floor_theory)
                                     else min(soft_floor_theory, wr))
            if g is None or g <= 0:
                continue
            if wb < SOFT_CHAR_CM1:
                soft_gammas.append(g)
            else:
                omegas_r.append(wr)
                gammas.append(g)
    order = np.argsort(lam_bare)
    lam_bare = np.asarray(lam_bare)[order]
    lam_ren = np.asarray(lam_ren)[order]
    soft_gamma_median = float(np.median(soft_gammas)) if soft_gammas else float("nan")

    # stable-character Gamma(omega_r): binned median + linear interpolation
    omegas_r = np.asarray(omegas_r)
    gammas = np.asarray(gammas)
    edges = np.linspace(0, omegas_r.max() * 1.001, GAMMA_BINS + 1)
    centers, medians = [], []
    for i in range(GAMMA_BINS):
        m = (omegas_r >= edges[i]) & (omegas_r < edges[i + 1])
        if m.sum() >= 2:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            medians.append(np.median(gammas[m]))
    centers = np.asarray(centers)
    medians = np.asarray(medians)

    def map_lambda(lam):
        return np.interp(lam, lam_bare, lam_ren)

    def map_gamma(omega_r):
        return np.interp(omega_r, centers, medians)

    return (map_lambda, map_gamma, soft_gamma_median, soft_floor_theory,
            float(lam_bare.min()))


_QE_Z: dict[int, list] = {}
MESH_SUBDIR = {11: "", 4: "ongrid4", 7: "mesh7", 9: "mesh9", 13: "mesh13"}


def _qe_polarisation_vectors(mesh_n: int = 11):
    """Orthonormal polarisation vectors of the bare QE modes, [iq] -> (15, 15) rows."""
    if mesh_n not in _QE_Z:
        directory = MODES_DIR / "SrTiO3" / MESH_SUBDIR[mesh_n] if MESH_SUBDIR[mesh_n] else MODES_DIR / "SrTiO3"
        blocks = read_modes(directory / "reference.modes")
        _QE_Z[mesh_n] = [orthonormal_eigenvectors(vec, MASSES["SrTiO3"]) for _, _, vec in blocks]
    return _QE_Z[mesh_n]


def _cubic_star(q):
    """All images of a fractional q under the 48 cubic operations, folded to [0, 1)."""
    import itertools
    out = set()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            v = tuple(round((signs[i] * q[perm[i]]) % 1.0, 6) for i in range(3))
            out.add(v)
    return out


def load_surface(temperature: int, tag: str = SURFACE_TAG, mesh_n: int = 11, match_mode: str = MATCH_MODE):
    """Own-surface inputs at one temperature on the n^3 production mesh
    (11 in production; 4, 7, 9, 13 for the convergence table, whose QE mode
    files live in data/raw/gruneisen_modes/SrTiO3/{ongrid4,mesh7,mesh9,mesh13}).

    Returns a dict with
      omega_direct[(iq, branch)]  renormalised frequency (cm-1) of the eigenvector-matched
                                  mode on the 11^3 mesh (multiplet mean),
      overlap[(iq, branch)]       |<e_bare|e_ren>|^2 summed over that multiplet,
      evec_index[(iq, branch)]    index of the matched renormalised mode (for the isotope projection),
      map_gamma, soft_gamma_median, soft_floor_theory   (character-aware Gamma map of the 8^3 RTA
                                  result, as build_maps but on the own surface),
      nearest_gamma(q_frac, omega_r) -> Gamma of the branch of the nearest 8^3 mesh point
                                  whose frequency is closest to omega_r (GAMMA_SOURCE = "nearest_q").
    """
    spec = SURFACES[tag]
    d, ftag = spec["dir"], spec["file_tag"]
    mesh = np.load(d / f"mesh{mesh_n}_renorm_{ftag}_{temperature}K.npz")
    om, ev = mesh["omega_cm1"], mesh["evec"]
    zq = _qe_polarisation_vectors(mesh_n)
    omega_direct, overlap, evec_index = {}, {}, {}
    if match_mode == "two_step":
        bare = np.load(d / ("mesh11_bare.npz" if mesh_n == 11 else f"mesh{mesh_n}_bare.npz"))
        omb, evb = bare["omega_cm1"], bare["evec"]
    for iq in range(om.shape[0]):
        if match_mode == "one_step":
            w, o, idx = match_to_renormalised(zq[iq], om[iq], ev[iq], DEGEN_TOL)
        else:
            # QE bare -> anphon bare (same harmonic surface or not), then anphon bare -> renormalised
            _, o1, idx1 = match_to_renormalised(zq[iq], omb[iq], evb[iq], DEGEN_TOL)
            w2, o2, idx2 = match_to_renormalised(evb[iq], om[iq], ev[iq], DEGEN_TOL)
            w, o, idx = w2[idx1], o1 * o2[idx1], idx2[idx1]
        for b in range(om.shape[1]):
            omega_direct[(iq, b + 1)] = float(w[b])
            overlap[(iq, b + 1)] = float(o[b])
            evec_index[(iq, b + 1)] = int(idx[b])
    # Gamma map from the RTA files of the surface (character split from its bare RTA)
    freq_bare, _ = parse_result(spec["rta_bare"], target_temp=300)
    rta_path = spec["rta_renorm"](temperature)
    freq_ren, gamma_ren = parse_result(rta_path, target_temp=temperature)
    map_gamma, soft_gamma_median, soft_floor_theory = _gamma_map_from_results(freq_bare, freq_ren, gamma_ren)
    # nearest-q assignment on the full 8^3 mesh
    klist = _irreducible_kpoints(rta_path)
    full = {}
    for (idx, qa, qb, qc, _w) in klist:
        for img in _cubic_star((qa, qb, qc)):
            full[img] = idx
    full_q = np.array(list(full.keys()))
    full_idx = np.array(list(full.values()))
    by_q = {}
    for (q, b), w in freq_ren.items():
        by_q.setdefault(q, []).append((w, gamma_ren.get((q, b), float("nan"))))

    def nearest_gamma(q_frac, omega_r):
        dq = (full_q - np.asarray(q_frac)[None, :] + 0.5) % 1.0 - 0.5
        j = int(np.argmin((dq**2).sum(axis=1)))
        cands = by_q[int(full_idx[j])]
        w, g = min(cands, key=lambda t: abs(t[0] - omega_r))
        return g

    return {"omega_direct": omega_direct, "overlap": overlap, "evec_index": evec_index,
            "map_gamma": map_gamma, "soft_gamma_median": soft_gamma_median,
            "soft_floor_theory": soft_floor_theory, "nearest_gamma": nearest_gamma, "evec": ev}


def _irreducible_kpoints(result_path):
    import re
    text = Path(result_path).read_text()
    block = re.search(r"#KPOINT\n\d+ \d+ \d+\n(\d+)\n(.*?)#END KPOINT", text, re.S)
    out = []
    for line in block.group(2).strip().splitlines():
        p = line.replace(":", " ").split()
        out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def _gamma_map_from_results(freq_bare, freq_ren, gamma_ren):
    """Character-aware Gamma(omega_r) map (stable-character binned median +
    soft-manifold median) from an 8^3 RTA result, with the character split
    by per-q rank pairing against the bare frequencies of the same surface."""
    ZERO_TOL = 0.5
    by_q_bare, by_q_ren = {}, {}
    for (q, b), w in freq_bare.items():
        by_q_bare.setdefault(q, []).append(w)
    for (q, b), w in freq_ren.items():
        by_q_ren.setdefault(q, []).append((w, gamma_ren.get((q, b))))
    omegas_r, gammas, soft_gammas = [], [], []
    soft_floor_theory = float("nan")
    for q in by_q_bare:
        bare = sorted(w for w in by_q_bare[q] if abs(w) > ZERO_TOL)
        ren = sorted(((w, g) for w, g in by_q_ren.get(q, []) if abs(w) > ZERO_TOL), key=lambda t: t[0])
        if len(bare) != len(ren):
            continue
        for wb, (wr, g) in zip(bare, ren):
            if wr <= 0:
                continue
            if q == 1 and wb < 0:
                soft_floor_theory = wr if np.isnan(soft_floor_theory) else min(soft_floor_theory, wr)
            if g is None or g <= 0:
                continue
            if wb < SOFT_CHAR_CM1:
                soft_gammas.append(g)
            else:
                omegas_r.append(wr)
                gammas.append(g)
    omegas_r, gammas = np.asarray(omegas_r), np.asarray(gammas)
    edges = np.linspace(0, omegas_r.max() * 1.001, GAMMA_BINS + 1)
    centers, medians = [], []
    for i in range(GAMMA_BINS):
        m = (omegas_r >= edges[i]) & (omegas_r < edges[i + 1])
        if m.sum() >= 2:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            medians.append(np.median(gammas[m]))
    centers, medians = np.asarray(centers), np.asarray(medians)

    def map_gamma(omega_r):
        return np.interp(omega_r, centers, medians)

    return map_gamma, (float(np.median(soft_gammas)) if soft_gammas else float("nan")), soft_floor_theory


def load_vogt():
    """Vogt (1995) soft-mode omega_s(T) and Gamma_HWHM(T) interpolants."""
    path = REPO / "data" / "processed" / "softmode_inputs_SrTiO3.csv"
    omega_T, omega_v, gamma_T, gamma_v = [], [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("quantity"):
            continue
        q, t, v = line.split(",")
        if q == "omega_F1u_cm1":
            omega_T.append(float(t)); omega_v.append(float(v))
        elif q == "Gamma_HWHM_cm1":
            gamma_T.append(float(t)); gamma_v.append(float(v))
    oi = np.argsort(omega_T)
    gi = np.argsort(gamma_T)
    omega_T = np.asarray(omega_T)[oi]; omega_v = np.asarray(omega_v)[oi]
    gamma_T = np.asarray(gamma_T)[gi]; gamma_v = np.asarray(gamma_v)[gi]
    rng = (max(omega_T.min(), gamma_T.min()), min(omega_T.max(), gamma_T.max()))

    def vogt(temperature):
        # +/-5 K edge tolerance: the digitized series ends at ~298 K and the
        # smooth soft-mode curves justify clamping over that gap (np.interp
        # clamps at the end points); anything further out is a real
        # extrapolation and returns None instead.
        if not (rng[0] - 5.0 <= temperature <= rng[1] + 5.0):
            return None, None
        return (float(np.interp(temperature, omega_T, omega_v)),
                float(np.interp(temperature, gamma_T, gamma_v)))
    return vogt, rng


def assemble(temperature: int, rows, vogt, extra_gamma_hwhm_cm1=None,
             return_details=False, cutoff_cm1: float = CUTOFF_CM1,
             omega_r_source: str | None = None, gamma_source: str | None = None,
             surface_tag: str | None = None, match_mode: str | None = None):
    """eta_xyxy at one temperature. Returns (eta_total, sector dict, flags)
    — plus a per-mode details list when return_details is True (audit use:
    (iq, branch, sector, omega0, omega_r, gamma_hwhm, gruneisen, tau_s,
    contribution_Pas, overlap)).

    omega_r_source: "direct" — eigenvector-matched renormalised frequency on
    the own surface (production); "rankmap" — the former per-q rank-pairing
    eigenvalue map of the tutorial-surface files (sensitivity).
    gamma_source: "freqmap" (production) or "nearest_q" (sensitivity); only
    used with omega_r_source = "direct".

    extra_gamma_hwhm_cm1: optional additional HWHM (cm-1) added to the
    anharmonic linewidth (Matthiessen) — the Tamura isotope channel. Either
    a dict keyed by (iq, branch) with per-mode values (the projected rate) or
    a callable omega_r_cm1 -> value.

    cutoff_cm1: Route S / Route H partition frequency. The production value
    is CUTOFF_CM1 = 175; scripts/scan_partition_sensitivity.py varies it to
    quantify the sensitivity of the total to the partition choice.
    """
    omega_r_source = OMEGA_R_SOURCE if omega_r_source is None else omega_r_source
    gamma_source = GAMMA_SOURCE if gamma_source is None else gamma_source
    surface_tag = SURFACE_TAG if surface_tag is None else surface_tag
    match_mode = MATCH_MODE if match_mode is None else match_mode
    if omega_r_source == "rankmap":
        (map_lambda, map_gamma, soft_gamma_median, soft_floor_theory,
         lam_bare_min) = build_maps(temperature)
        surface = None
    elif omega_r_source == "direct":
        surface = load_surface(temperature, surface_tag, mesh_n=int(round(N_Q ** (1.0 / 3.0))), match_mode=match_mode)
        map_gamma = surface["map_gamma"]
        soft_gamma_median = surface["soft_gamma_median"]
        soft_floor_theory = surface["soft_floor_theory"]
        lam_bare_min = -np.inf
    else:
        raise ValueError(omega_r_source)
    v_omega, v_gamma = vogt(temperature)
    vogt_used = v_omega is not None

    sectors = {"routeS": 0.0, "routeH_stable": 0.0, "routeH_unstable": 0.0,
               "gamma_sector": 0.0}
    n_extrapolated = 0
    n_skipped = 0
    n_low_overlap = 0
    details = []

    for r in rows:
        if r["acoustic"]:
            continue
        omega0 = r["omega_ref"]
        D = r["D"]
        key = (r["iq"], r["branch"])
        lam0 = np.sign(omega0) * omega0 * omega0
        if surface is None:
            if lam0 < lam_bare_min:
                n_extrapolated += 1
            lam_r = map_lambda(lam0)
            if lam_r <= 0:
                n_skipped += 1
                continue
            omega_r = float(np.sqrt(lam_r))
            overlap = float("nan")
        else:
            omega_r = surface["omega_direct"][key]
            overlap = surface["overlap"][key]
            if overlap < PAIR_OVERLAP_THRESHOLD:
                n_low_overlap += 1
            if omega_r <= 0:
                n_skipped += 1
                continue

        at_gamma_soft = (r["iq"] == 0 and omega0 < OMEGA_MIN)
        if at_gamma_soft and vogt_used:
            omega_r = v_omega
            gamma_hwhm = v_gamma
            sector = "gamma_sector"
        elif omega0 < OMEGA_MIN:
            # bare-unstable manifold: soft-TO character — soft-sector Gamma
            # statistics, and a PHYSICAL floor on omega_r: the TO1 branch
            # has its minimum at Gamma (Vogt omega_s), so any renormalized
            # member of the bare-imaginary manifold must satisfy
            # omega_r >= omega_s(T); the crude eigenvalue map violated this
            # near Gamma (39.9 vs 89.2 cm-1 at 300 K — audit finding A3),
            # inflating gamma = Lambda/(2 omega_r^2) by ~5x there.
            gamma_hwhm = soft_gamma_median
            floor = v_omega if vogt_used else soft_floor_theory
            if np.isfinite(floor):
                omega_r = max(omega_r, floor)
            sector = "gamma_sector" if at_gamma_soft else "routeH_unstable"
        elif omega0 < SOFT_CHAR_CM1:
            # soft-manifold stable modes: soft-character Gamma, never the
            # acoustic-contaminated frequency-binned map (audit finding A3)
            gamma_hwhm = soft_gamma_median
            sector = "routeH_stable"
        else:
            if surface is not None and gamma_source == "nearest_q":
                n = int(round(N_Q ** (1.0 / 3.0)))
                q_frac = ((r["iq"] // (n * n)) / n, ((r["iq"] // n) % n) / n, (r["iq"] % n) / n)
                gamma_hwhm = float(surface["nearest_gamma"](q_frac, omega_r))
            else:
                gamma_hwhm = float(map_gamma(omega_r))
            sector = "routeS" if omega0 >= cutoff_cm1 else "routeH_stable"

        if extra_gamma_hwhm_cm1 is not None:
            if isinstance(extra_gamma_hwhm_cm1, dict):
                gamma_hwhm = gamma_hwhm + float(extra_gamma_hwhm_cm1[(r["iq"], r["branch"])])
            else:
                gamma_hwhm = gamma_hwhm + float(extra_gamma_hwhm_cm1(omega_r))

        if sector == "routeS":
            gru = -D / (2.0 * omega0 * omega0)
        else:
            gru = -D / (2.0 * omega_r * omega_r)

        w = omega_r * CM1                       # rad/s
        lw = gamma_hwhm * CM1                   # rad/s (HWHM)
        occupation = bose_einstein(w, temperature)
        tau = float(tau_two_pole_stress(w, lw))
        contrib = (HBAR * w) ** 2 * gru * gru * occupation * (occupation + 1.0) * tau
        sectors[sector] += contrib
        if return_details:
            details.append({"iq": r["iq"], "branch": r["branch"],
                            "sector": sector, "omega0": omega0,
                            "omega_r": omega_r, "gamma_hwhm": gamma_hwhm,
                            "gruneisen": gru, "tau_s": tau, "overlap": overlap,
                            "contrib_raw": contrib})

    norm = 1.0 / (V_CELL * N_Q * K_B * temperature)
    eta_sectors = {k: v * norm for k, v in sectors.items()}
    eta_total = sum(eta_sectors.values())
    flags = {"vogt_used": vogt_used, "n_extrapolated": n_extrapolated,
             "n_skipped": n_skipped, "n_low_overlap": n_low_overlap,
             "omega_r_source": omega_r_source, "gamma_source": gamma_source, "surface": surface_tag,
             "match_mode": match_mode}
    if return_details:
        for d in details:
            d["eta_contrib"] = d.pop("contrib_raw") * norm
        return eta_total, eta_sectors, flags, details
    return eta_total, eta_sectors, flags



# =============================================================================
# PRODUCTION MODEL (revision): the hybrid model tut_z_od1, SELF_OFFDIAG = 1, construction C
# =============================================================================
# Every production frequency omega_r, eigenvector e_r and coupling Lambda_C comes from ONE SCPH surface
# (the hybrid model): the ALAMODE v1.5.0 example harmonic set re-expressed on the 4x4x4 supercell, strained as
# D(q, +-h) = D_tut(q) +- h K_QE(q) with K_QE from the QE-PBEsol strained pairs (short-range part in real space,
# non-analytic part through the strained Born charges and dielectric tensor), + the external ALAMODE example
# anharmonic set (orders 3-6; cubic terms used by the RTA linewidths, quartic by SCPH; IFC supercell 2x2x2) +
# own SCPH (correction mesh KMESH_INTERPOLATE 2x2x2 / inner mesh KMESH_SCPH 12x12x12, SELF_OFFDIAG = 1) for the
# unstrained cell and the shear-strained cells eps_xy = eps_yx = +-0.005 (engineering shear h = +-0.010), + own
# 8^3 RTA per T. The diagnostic surface (own QE-PBEsol harmonic set, spec OWN_OD1) is selectable through `spec=`.
#   Lambda_C(nu) = e_nu^dagger K_SCPH e_nu,  K_SCPH = [D_SCPH(+h) - D_SCPH(-h)] / (2h),
# with degenerate multiplets by projected-block diagonalisation (gauge-invariant Tr K_sub^2), and
#   gamma_C = -Lambda_C / (2 omega_r^2).
# EXCEPTION (I3): the Gamma-point soft TO1 triplet uses the Vogt (1995) frequency and damping.
# Linewidths: character-aware frequency-class map of the same-T own RTA (nearest-q as a sensitivity);
# Bose weights at omega_r; stress-correlator kernel; no floor; no Route S/H partition.
# Construction B (bare K_HA on the same basis) and A (bare couplings transferred by maximum overlap,
# the previous model) remain runnable through `construction=`; the tutorial-surface / rank-map paths
# above (assemble, load_surface, build_maps) are kept for the ledger rows.

from latvisc.coupling import (  # noqa: E402
    AlamodeSet, QESet, acoustic_limit_check, diagonalise, engineering_shear, gruneisen_from_coupling,
    project_coupling, strain_derivative_matrix, transfer_bare_coupling,
)

CONSTRUCTION = "C"
STRAIN_AMPLITUDE = 0.005                      # eps_xy = eps_yx = s of the strained cells
H_ENG = engineering_shear(STRAIN_AMPLITUDE)   # 0.010, the derivative variable of every Lambda below
STRAIN_SETS = ("reference", "shear_xy_p005", "shear_xy_m005")
QE_DIR = REPO / "dft" / "qe" / "SrTiO3"
QE_FC = {
    "reference": QE_DIR / "dispersion_pbesol" / "SrTiO3_pbesol.444.fc",
    "shear_xy_p005": QE_DIR / "gruneisen_pbesol" / "shear_xy_p005" / "SrTiO3_pbesol_shear_xy_p005.444.fc",
    "shear_xy_m005": QE_DIR / "gruneisen_pbesol" / "shear_xy_m005" / "SrTiO3_pbesol_shear_xy_m005.444.fc",
    "shear_xy_p010": QE_DIR / "gruneisen_pbesol" / "shear_xy_p010" / "SrTiO3_pbesol_shear_xy_p010.444.fc",
    "shear_xy_m010": QE_DIR / "gruneisen_pbesol" / "shear_xy_m010" / "SrTiO3_pbesol_shear_xy_m010.444.fc",
}
OWN_OD1 = {
    "dir": ALAMODE_DIR / "own_od1" / "i2s2",
    "mesh": "i2s2",                      # KMESH_INTERPOLATE 2x2x2 / KMESH_SCPH 2x2x2
    "self_offdiag": 1,
    "rta_bare": ALAMODE_DIR / "own_surface_i2s2" / "STO_RTA_own_i2s2_bare.result",
    # the +-0.010 strained SCPH sets (single-temperature runs, 300 K only) for the step comparison
    "h020_dir": ALAMODE_DIR / "own_surface_6p2",
}
# Production model (revision): the hybrid model "tut_z_od1" = the ALAMODE v1.5.0 example harmonic set re-expressed on
# the 4x4x4 supercell (external), the QE-PBEsol strain perturbation D(q, +-h) = D_tut(q) +- h K_QE(q) (own), SCPH with
# SELF_OFFDIAG = 1 at correction mesh (KMESH_INTERPOLATE) 2x2x2 and inner mesh (KMESH_SCPH) 12x12x12 (the finest completed
# mesh; convergence of the complete coefficient is not established), own 8^3 RTA linewidths. The own-surface model
# ("own_od1", QE-PBEsol harmonic set) is kept as the diagnostic surface for the sensitivity rows.
Z_DIR = ALAMODE_DIR / "z_tut"
Z_FC = {tag: Z_DIR / f"z_{tag}.fc" for tag in QE_FC}          # q2r-format short-range sets of the hybrid model


def _z_label(temperature: int) -> str:
    return "i2s12" if temperature == 300 else f"i2s12T{temperature}"


TUT_Z_OD1 = {
    "dir": lambda T: Z_DIR / ("i2s12" if T == 300 else "i2s12_T"),
    "xml": lambda tag, T: f"renorm_z_{tag}_{_z_label(T)}_od1_{T}K.xml",
    "rta": lambda T: f"STO_RTA_{_z_label(T)}_od1_{T}K.result",
    "mesh": "i2s12",                     # correction mesh 2x2x2 / inner mesh 12x12x12
    "self_offdiag": 1,
    "rta_bare": Z_DIR / "bare" / "STO_RTA_z_bare.result",
    "fc": Z_FC,
    "harmonic": "tut_z",
    "h020": lambda tag, T: Z_DIR / "i2s12" / f"renorm_z_{tag}_i2s12_od1_{T}K.xml",
    "descriptor": "tut_z_od1 (hybrid model: example harmonic set on 4x4x4 + QE strain perturbation; "
                  "correction mesh 2x2x2, inner mesh 12x12x12, SELF_OFFDIAG = 1)",
}
OWN_OD1["fc"] = QE_FC
OWN_OD1["harmonic"] = "own"
OWN_OD1["h020"] = lambda tag, T: OWN_OD1["h020_dir"] / f"renorm_own_{tag}_od1_{T}K.xml"
OWN_OD1["descriptor"] = "own_od1 (diagnostic surface: QE-PBEsol harmonic set; correction mesh 2x2x2, inner mesh 2x2x2)"
SURFACE_SPECS = {"own_od1": OWN_OD1, "tut_z_od1": TUT_Z_OD1}
PRODUCTION_SURFACE = "tut_z_od1"
PROD = SURFACE_SPECS[PRODUCTION_SURFACE]
OMEGA0_BINS = [("imaginary", -np.inf, 0.0), ("[0-50)", 0.0, 50.0), ("[50-100)", 50.0, 100.0),
               ("[100-175)", 100.0, 175.0), ("[175-inf)", 175.0, np.inf)]      # comma-free labels (CSV fields)
SECTOR_KEYS = [b[0] for b in OMEGA0_BINS] + ["gamma_sector"]
COUPLING_KEY = {"A": "lam_A", "B": "lam_B", "C": "lam_C"}
O_SITE_INDEX = [2, 3, 4]
_QE_SETS: dict = {}


def fc_paths(spec: dict | None = None) -> dict:
    """bare harmonic sets (q2r format) belonging to a surface spec; own-surface variants default to the QE sets."""
    spec = PROD if spec is None else spec
    return spec.get("fc", QE_FC)


# Harmonic (unrenormalised) sets used for constructions A and B. "alamode" (production): the ALAMODE FC2 XMLs that the SCPH
# runs start from, evaluated like the SCPH sets (rigid-ion dipole subtracted and restored in real space) - Hermitian, so the
# strain derivative is translationally invariant. "qe": the q2r .fc files with the QE rigid-ion term of each (sheared) cell,
# which is non-Hermitian at ~1 % for a sheared cell with an odd Born-charge part (former production; comparison only).
HARMONIC_CONVENTION = "alamode"
HARMONIC_XML = {
    "tut_z": {"reference": Z_DIR / "i2s12" / "z_reference_full_fc2.xml",
              **{t: Z_DIR / f"z_{t}_full_fc2.xml" for t in ("shear_xy_p005", "shear_xy_m005", "shear_xy_p010", "shear_xy_m010")}},
    "own": {**{t: ALAMODE_DIR / "own_od1" / "i2s8" / f"{t}_full_fc2.xml" for t in ("reference", "shear_xy_p005", "shear_xy_m005")},
            **{t: ALAMODE_DIR / "own_surface_6p2" / f"{t}_full_fc2.xml" for t in ("shear_xy_p010", "shear_xy_m010")}},
}


class HarmonicSet:
    """ALAMODE-convention harmonic set with the QESet interface used here (dynmat(q), masses_amu)."""

    def __init__(self, xml, fc):
        self.s = AlamodeSet(xml, fc)
        self.masses_amu = self.s.masses_amu

    def dynmat(self, q_frac):
        return self.s.dynmat(q_frac, asr_onsite=True)


def qe_set(tag: str, spec: dict | None = None):
    """harmonic set `tag` of a surface spec in the production convention (HARMONIC_CONVENTION)."""
    spec = PROD if spec is None else spec
    path = fc_paths(spec)[tag]
    key = (HARMONIC_CONVENTION, str(path))
    if key not in _QE_SETS:
        fc = fc_paths(spec)
        default = "tut_z" if fc is Z_FC else ("own" if fc is QE_FC else "")   # surface variants inherit their model's sets
        xmls = HARMONIC_XML.get(spec.get("harmonic", default), {})
        if HARMONIC_CONVENTION == "alamode" and tag in xmls:
            _QE_SETS[key] = HarmonicSet(xmls[tag], path)
        else:
            _QE_SETS[key] = QESet(path)
    return _QE_SETS[key]


def _spec_dir(spec: dict, temperature: int) -> Path:
    return spec["dir"](temperature) if callable(spec["dir"]) else spec["dir"]


def own_od1_xml(tag: str, temperature: int, spec: dict | None = None) -> Path:
    """renormalised FC2 XML of a surface spec (name kept for the callers; any surface)."""
    spec = PROD if spec is None else spec
    d = _spec_dir(spec, temperature)
    if "xml" in spec:
        x = spec["xml"]
        return d / (x(tag, temperature) if callable(x) else x.format(tag=tag, T=temperature))
    return d / f"renorm_{tag}_{spec['mesh']}_od{spec['self_offdiag']}_{temperature}K.xml"


def own_od1_rta(temperature: int, spec: dict | None = None) -> Path:
    spec = PROD if spec is None else spec
    d = _spec_dir(spec, temperature)
    if "rta" in spec:
        r = spec["rta"]
        return d / (r(temperature) if callable(r) else r.format(T=temperature))
    return d / f"STO_RTA_{spec['mesh']}_od{spec['self_offdiag']}_{temperature}K.result"


def surface_available(temperature: int, spec: dict | None = None) -> bool:
    spec = PROD if spec is None else spec
    return all(own_od1_xml(t, temperature, spec).exists() for t in STRAIN_SETS) and own_od1_rta(temperature, spec).exists()


def omega0_bin(omega0: float) -> str:
    for name, lo, hi in OMEGA0_BINS:
        if lo <= omega0 < hi:
            return name
    return "[175-inf)"


def mesh_points(n: int):
    return [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)]


AVERAGE_DEGENERATE_LINEWIDTHS = True


def average_degenerate_linewidths(freq: dict, gamma: dict, tol: float = None) -> tuple[dict, int]:
    """Average the RTA linewidths over exactly degenerate sets (same irreducible q, frequencies equal within the
    exact-degeneracy tolerance EXACT_DEGEN_TOL), as ALAMODE requires for results computed with TRISYM = 1 (the
    equivalent of its analysis utility). Returns the averaged dict and the number of sets averaged."""
    tol = EXACT_DEGEN_TOL if tol is None else tol
    by_q = {}
    for (q, b), w in freq.items():
        by_q.setdefault(q, []).append((w, b))
    out, n_sets = dict(gamma), 0
    for q, lst in by_q.items():
        lst.sort()
        i = 0
        while i < len(lst):
            j = i
            while j + 1 < len(lst) and lst[j + 1][0] - lst[j][0] <= tol:
                j += 1
            if j > i:
                keys = [(q, b) for _, b in lst[i:j + 1] if (q, b) in gamma]
                if len(keys) > 1:
                    m = float(np.mean([gamma[k] for k in keys]))
                    for k in keys:
                        out[k] = m
                    n_sets += 1
            i = j + 1
    return out, n_sets


def load_construction_surface(temperature: int, with_h020: bool = False, spec: dict | None = None) -> dict:
    """Own-surface SCPH sets at one temperature (unstrained, +-0.005) and the linewidth maps of
    the same-T own RTA (character split from the bare RTA of the same harmonic set).
    `spec` selects another own-surface variant (internal SCPH mesh, symmetrised inputs) for the
    sensitivity table: dict with dir, and either mesh/self_offdiag or xml/rta name patterns."""
    spec = PROD if spec is None else spec
    fc = fc_paths(spec)
    sets = {tag: AlamodeSet(own_od1_xml(tag, temperature, spec), fc[tag]) for tag in STRAIN_SETS}
    if with_h020 and "h020" in spec:
        for tag in ("shear_xy_p010", "shear_xy_m010"):
            path = spec["h020"](tag, temperature)
            if path.exists():
                sets[tag] = AlamodeSet(path, fc[tag])
    freq_bare, _ = parse_result(spec.get("rta_bare", OWN_OD1["rta_bare"]), target_temp=300)
    rta_path = own_od1_rta(temperature, spec)
    freq_ren, gamma_ren = parse_result(rta_path, target_temp=temperature)
    n_avg = 0
    if AVERAGE_DEGENERATE_LINEWIDTHS:
        gamma_ren, n_avg = average_degenerate_linewidths(freq_ren, gamma_ren)
    map_gamma, soft_gamma_median, soft_floor_theory = _gamma_map_from_results(freq_bare, freq_ren, gamma_ren)
    klist = _irreducible_kpoints(rta_path)
    full = {}
    for (idx, qa, qb, qc, _w) in klist:
        for img in _cubic_star((qa, qb, qc)):
            full[img] = idx
    full_q = np.array(list(full.keys()))
    full_idx = np.array(list(full.values()))
    by_q = {}
    for (q, b), w in freq_ren.items():
        by_q.setdefault(q, []).append((w, gamma_ren.get((q, b), float("nan"))))

    def nearest_gamma(q_frac, omega_r):
        dq = (full_q - np.asarray(q_frac)[None, :] + 0.5) % 1.0 - 0.5
        j = int(np.argmin((dq**2).sum(axis=1)))
        w, g = min(by_q[int(full_idx[j])], key=lambda t: abs(t[0] - omega_r))
        return g

    return {"temperature": temperature, "sets": sets, "spec": spec, "map_gamma": map_gamma,
            "soft_gamma_median": soft_gamma_median, "soft_floor_theory": soft_floor_theory,
            "nearest_gamma": nearest_gamma, "rta_path": rta_path, "n_degenerate_sets_averaged": n_avg}


def construction_modes(temperature: int, mesh_n: int = 11, surface: dict | None = None, qpoints=None):
    """Every renormalised mode (q on the Gamma-centred mesh, branch nu) of the own surface at one
    temperature with its three couplings (A, B, C in cm^-2 per unit engineering shear), the
    max-overlap bare partner (character class), the multiplet information and the O-site components
    of e_r (for the isotope projection)."""
    surface = load_construction_surface(temperature) if surface is None else surface
    sets = surface["sets"]
    bare = {tag: qe_set(tag, surface.get("spec")) for tag in STRAIN_SETS}
    have_h020 = "shear_xy_p010" in sets
    out = []
    for iq, q in enumerate(mesh_points(mesh_n) if qpoints is None else qpoints):
        omega_b, E_b = diagonalise(bare["reference"].dynmat(q))
        omega_r, E_r = diagonalise(sets["reference"].dynmat(q, asr_onsite=True))
        K_HA = strain_derivative_matrix(bare["shear_xy_p005"].dynmat(q), bare["shear_xy_m005"].dynmat(q), H_ENG)
        K_SC = strain_derivative_matrix(sets["shear_xy_p005"].dynmat(q, asr_onsite=True),
                                        sets["shear_xy_m005"].dynmat(q, asr_onsite=True), H_ENG)
        pb = project_coupling(K_HA, E_b, omega_b, DEGEN_TOL)
        pB = project_coupling(K_HA, E_r, omega_r, COUPLING_DEGEN_TOL)
        pC = project_coupling(K_SC, E_r, omega_r, COUPLING_DEGEN_TOL)
        lam_A, mu, ov1, ovm = transfer_bare_coupling(pb["lam"], E_b, E_r, omega_r, DEGEN_TOL)
        if have_h020:
            K2 = strain_derivative_matrix(sets["shear_xy_p010"].dynmat(q, asr_onsite=True),
                                          sets["shear_xy_m010"].dynmat(q, asr_onsite=True), 2.0 * H_ENG)
            pC2 = project_coupling(K2, E_r, omega_r, COUPLING_DEGEN_TOL)
            pCr = project_coupling((4.0 * K_SC - K2) / 3.0, E_r, omega_r, COUPLING_DEGEN_TOL)
        for nu in range(3 * len(bare["reference"].masses_amu)):
            rec = {"iq": iq, "q": q, "nu": nu + 1, "omega_r": float(omega_r[nu]), "omega0": float(omega_b[mu[nu]]),
                   "partner": int(mu[nu]) + 1, "overlap_single": float(ov1[nu]), "overlap_multiplet": float(ovm[nu]),
                   "group_size": int(pC["group_size"][nu]), "lam_A": float(lam_A[nu]), "lam_B": float(pB["lam"][nu]),
                   "lam_C": float(pC["lam"][nu]), "trk2_C": float(pC["trk2"][nu]), "offdiag_C": float(pC["offdiag"][nu]),
                   "evec_O": E_r[:, nu].reshape(-1, 3)[O_SITE_INDEX].copy()}
            if have_h020:
                rec["lam_C_h020"] = float(pC2["lam"][nu])
                rec["lam_C_rich"] = float(pCr["lam"][nu])
            out.append(rec)
    return out


def assemble_construction(temperature: int, construction: str = CONSTRUCTION, mesh_n: int = 11,
                          gamma_source: str = GAMMA_SOURCE, extra_gamma_hwhm_cm1=None, modes=None,
                          surface: dict | None = None, return_details: bool = False, coupling_key: str | None = None,
                          vogt=None):
    """eta_xyxy at one temperature on the own surface (production: construction C).

    Returns (eta_total, sectors by bare-omega_0 bin of the max-overlap partner + 'gamma_sector', flags)
    and the per-mode details when return_details is True. `coupling_key` overrides the construction
    ('lam_C_h020', 'lam_C_rich' for the step sensitivities); `extra_gamma_hwhm_cm1` adds an isotope
    HWHM per (iq, nu) (dict) or as a function of omega_r (callable)."""
    if modes is None:
        modes = construction_modes(temperature, mesh_n, surface)
    if surface is None:
        surface = load_construction_surface(temperature)
    key = coupling_key or COUPLING_KEY[construction]
    if vogt is None:
        vogt, _ = load_vogt()
    v_omega, v_gamma = vogt(temperature)
    vogt_used = v_omega is not None
    n_q = mesh_n ** 3
    sectors = {k: 0.0 for k in SECTOR_KEYS}
    n_skipped = 0
    n_low_overlap = 0
    details = []
    for m in modes:
        omega_r, omega0, iq = m["omega_r"], m["omega0"], m["iq"]
        if iq == 0 and omega_r < OMEGA_MIN:
            continue                                   # acoustic zeros at Gamma
        if omega_r <= 0:
            n_skipped += 1
            continue
        if m["overlap_multiplet"] < PAIR_OVERLAP_THRESHOLD:
            n_low_overlap += 1
        if iq == 0 and omega0 < OMEGA_MIN:
            sector = "gamma_sector"
            omega_use = v_omega if vogt_used else omega_r
            gamma_hwhm = v_gamma if vogt_used else surface["soft_gamma_median"]
        else:
            sector = omega0_bin(omega0)
            omega_use = omega_r
            if omega0 < SOFT_CHAR_CM1:
                gamma_hwhm = surface["soft_gamma_median"]
            elif gamma_source == "nearest_q":
                gamma_hwhm = float(surface["nearest_gamma"](m["q"], omega_r))
            else:
                gamma_hwhm = float(surface["map_gamma"](omega_r))
        if extra_gamma_hwhm_cm1 is not None:
            if isinstance(extra_gamma_hwhm_cm1, dict):
                gamma_hwhm = gamma_hwhm + float(extra_gamma_hwhm_cm1.get((iq, m["nu"]), 0.0))
            else:
                gamma_hwhm = gamma_hwhm + float(extra_gamma_hwhm_cm1(omega_r))
        lam = m[key]
        gru = float(gruneisen_from_coupling(lam, omega_use))
        w = omega_use * CM1
        lw = gamma_hwhm * CM1
        occupation = bose_einstein(w, temperature)
        tau = float(tau_two_pole_stress(w, lw))
        contrib = (HBAR * w) ** 2 * gru * gru * occupation * (occupation + 1.0) * tau
        sectors[sector] += contrib
        if return_details:
            details.append({**{k: v for k, v in m.items() if k != "evec_O"}, "sector": sector, "omega_use": omega_use,
                            "gamma_hwhm": gamma_hwhm, "gruneisen": gru, "tau_s": tau, "n_n_plus_1": occupation * (occupation + 1.0),
                            "contrib_raw": contrib, "coupling": lam})
    norm = 1.0 / (V_CELL * n_q * K_B * temperature)
    eta_sectors = {k: v * norm for k, v in sectors.items()}
    eta_total = sum(eta_sectors.values())
    flags = {"vogt_used": vogt_used, "n_skipped": n_skipped, "n_low_overlap": n_low_overlap,
             "surface": PRODUCTION_SURFACE, "construction": construction, "coupling_key": key,
             "gamma_source": gamma_source, "mesh_n": mesh_n, "h_engineering": H_ENG}
    if return_details:
        for d in details:
            d["eta_contrib"] = d.pop("contrib_raw") * norm
        return eta_total, eta_sectors, flags, details
    return eta_total, eta_sectors, flags


def scph_run_statistics() -> list:
    """Iterations and residuals of the production SCPH runs (runs.csv written by the driver) and the
    SELF_OFFDIAG value of every production SCPH deck."""
    out = []
    pdir = _spec_dir(PROD, 300)
    runs = pdir / "runs.csv"
    if runs.exists():
        for line in runs.read_text().splitlines()[1:]:
            p = line.split(",")
            if p[4] == "scph":
                out.append({"set": p[0], "mesh": p[1], "self_offdiag": int(p[2]), "wall_s": float(p[5]),
                            "iterations": p[6], "final_DIFF": p[7]})
    decks = sorted(pdir.glob("scph_*.in"))
    for d in decks:
        txt = d.read_text()
        for r in out:
            if r["set"] in d.name:
                r["deck_self_offdiag"] = int("SELF_OFFDIAG = 1" in txt)
    return out


def sanity_checks(details, mesh_etas=None, surface=None, modes=None) -> dict:
    """Physics checks of the production model (numbers, no pass/fail band):
    (i)   engineering-shear identity on a toy coupling through latvisc.coupling;
    (ii)  mesh convergence of construction C (9^3/11^3/13^3/15^3, `mesh_etas` = {n: eta});
    (iii) positivity of every contribution;
    (iv)  acoustic limit on the production surface at 300 K: Lambda_C/q^2 finite and gamma of the
          TA/LA branches along Gamma-X and Gamma-M at |q| = 0.05, 0.1 (reported);
    (v)   SCPH iterations / residual per temperature and SELF_OFFDIAG = 1 on every production deck;
    (vi)  0.005 vs 0.010 strain step for Lambda_C (zone-wide, 300 K; reported).
    """
    out = {}
    # (i)
    w0, lam_t, s = 100.0, -2.0e4, 0.005
    h = engineering_shear(s)
    d_plus = np.array([[w0**2 + lam_t * h]]); d_minus = np.array([[w0**2 - lam_t * h]])
    lam = strain_derivative_matrix(d_plus, d_minus, h)[0, 0].real
    out["toy_gamma_pipeline_over_tensor"] = float(gruneisen_from_coupling(lam, w0) / (-lam_t / (2 * w0**2)))
    out["toy_engineering_over_symmetric_path"] = float(lam / ((d_plus - d_minus)[0, 0].real / (2 * s)))   # 1/2
    # (ii)
    out["mesh_etas"] = dict(mesh_etas or {})
    if mesh_etas and 11 in mesh_etas and 13 in mesh_etas:
        out["mesh_11_vs_13_pct"] = 100.0 * (mesh_etas[11] / mesh_etas[13] - 1.0)
    if mesh_etas and len(mesh_etas) >= 2:
        vals = np.array(list(mesh_etas.values()))
        out["mesh_spread_pct"] = 100.0 * (vals.max() - vals.min()) / vals[-1]
    # (iii)
    contribs = np.array([d["eta_contrib"] for d in details])
    out["min_mode_contribution_Pas"] = float(contribs.min())
    out["all_contributions_positive"] = bool((contribs >= 0).all())
    # (iv)
    if surface is not None:
        sets = surface["sets"]
        masses = qe_set("reference", surface.get("spec")).masses_amu
        ac = []
        for name, d in (("GX", (1, 0, 0)), ("GM", (1, 1, 0))):
            rows = acoustic_limit_check(lambda q: sets["reference"].dynmat(q, asr_onsite=True),
                                        lambda q: sets["shear_xy_p005"].dynmat(q, asr_onsite=True),
                                        lambda q: sets["shear_xy_m005"].dynmat(q, asr_onsite=True), H_ENG, masses, d)
            for r in rows:
                r["line"] = name
            ac += rows
        out["acoustic_limit"] = ac
    # (v)
    out["scph_runs"] = scph_run_statistics()
    out["all_decks_self_offdiag_1"] = bool(out["scph_runs"]) and all(r.get("deck_self_offdiag", 0) == 1 for r in out["scph_runs"])
    # (vi)
    if modes is not None and modes and "lam_C_h020" in modes[0]:
        by = {(d["iq"], d["nu"]): d for d in details}
        num = den = 0.0
        for m in modes:
            d = by.get((m["iq"], m["nu"]))
            if d is None or m["lam_C"] == 0:
                continue
            pref = d["eta_contrib"] / (m["lam_C"] ** 2)
            num += pref * m["lam_C_h020"] ** 2
            den += pref * m["lam_C"] ** 2
        out["eta_C_h020_over_h010"] = num / den if den else float("nan")
    print("\nSanity checks (numbers, no band):")
    print(f"  (i)   toy gamma / tensor = {out['toy_gamma_pipeline_over_tensor']:.9f}; engineering-shear derivative / "
          f"symmetric-path derivative = {out['toy_engineering_over_symmetric_path']:.6f} (must be 0.5)")
    if "mesh_spread_pct" in out:
        print(f"  (ii)  mesh convergence of C: {', '.join(f'{n}^3 {e:.4e}' for n, e in out['mesh_etas'].items())}; "
              f"spread {out['mesh_spread_pct']:.2f} % of the densest; 11^3 vs 13^3 {out.get('mesh_11_vs_13_pct', float('nan')):+.2f} %")
    print(f"  (iii) all mode contributions non-negative: {out['all_contributions_positive']} (min {out['min_mode_contribution_Pas']:.2e} Pa s)")
    if "acoustic_limit" in out:
        for r in out["acoustic_limit"]:
            print(f"  (iv)  {r['line']} |q| = {r['q_mag']:.2f} {r['character']}: omega_r {r['omega_r']:6.1f} cm-1, "
                  f"Lambda_C/q^2 = {r['lam_over_q2']:11.4g} cm-2, gamma_C = {r['gamma']:+.3f}")
    for r in out["scph_runs"]:
        print(f"  (v)   SCPH {r['set']:14s} {r['mesh']} SELF_OFFDIAG {r['self_offdiag']} (deck {r.get('deck_self_offdiag', '?')}): "
              f"{r['iterations']} iterations, final residual {r['final_DIFF']}, {r['wall_s']:.0f} s")
    print(f"        every production deck SELF_OFFDIAG = 1: {out['all_decks_self_offdiag_1']}")
    if "eta_C_h020_over_h010" in out:
        print(f"  (vi)  eta_C with the 0.010-amplitude pair (h = 0.020) / production (h = 0.010): {out['eta_C_h020_over_h010']:.4f}")
    return out


def export_mode_table(details, modes, path: Path, eta_total: float, temperature: int, extra_gamma=None) -> None:
    """mode table v6: one row per renormalised mode slot (production surface, construction C)."""
    n = int(round(len(modes) ** (1.0 / 3.0) / 15 ** (1.0 / 3.0))) if False else None  # noqa: F841 (mesh known from details)
    by_mode = {(m["iq"], m["nu"]): m for m in modes}
    rows = sorted(details, key=lambda d: -d["eta_contrib"])
    cum = np.cumsum([d["eta_contrib"] for d in rows]) / eta_total
    rank = {(d["iq"], d["nu"]): (k + 1, cum[k]) for k, d in enumerate(rows)}
    cols = ["iq", "qa", "qb", "qc", "nu", "omega_r_cm1", "omega0_partner_cm1", "partner_mu", "overlap_single",
            "overlap_multiplet", "multiplet_size", "Lambda_C_cm2", "Lambda_B_cm2", "Lambda_A_cm2", "TrK2_C_cm4",
            "Lambda_C_h020_cm2", "gamma_C", "gamma_B", "gamma_A", "sector", "omega_used_cm1", "Gamma_hwhm_cm1",
            "Gamma_iso_f015_cm1", "n_n_plus_1", "tau_stress_ps", "eta_contrib_Pas", "rank", "cum_frac"]
    lines = [
        f"# mode_table_SrTiO3_{temperature}K_v6.csv - production model {PROD['descriptor']}, construction C;",
        f"# eta_xyxy({temperature} K) = {eta_total:.6e} Pa s; Lambda in cm^-2 per unit engineering shear h = 2 s (d/dh = d/d eps_xy);",
        "# gamma = -Lambda/(2 omega_used^2); the Gamma-point TO1 triplet uses the Vogt frequency/damping (I3 exception);",
        "# sectors = bare-omega_0 bin of the max-overlap bare partner; multiplet couplings = projected-block eigenvalues;",
        "# Gamma_iso_f015: projected Tamura HWHM at f = 0.15 with the production-surface eigenvectors (when available).",
        ",".join(cols),
    ]
    for d in sorted(details, key=lambda d: (d["iq"], d["nu"])):
        m = by_mode[(d["iq"], d["nu"])]
        q = d["q"]
        r, c = rank[(d["iq"], d["nu"])]
        giso = float(extra_gamma.get((d["iq"], d["nu"]), float("nan"))) if extra_gamma else float("nan")
        vals = [d["iq"], f"{q[0]:.6g}", f"{q[1]:.6g}", f"{q[2]:.6g}", d["nu"], f"{d['omega_r']:.6g}", f"{d['omega0']:.6g}",
                d["partner"], f"{d['overlap_single']:.6g}", f"{d['overlap_multiplet']:.6g}", d["group_size"],
                f"{d['lam_C']:.6g}", f"{d['lam_B']:.6g}", f"{d['lam_A']:.6g}", f"{d['trk2_C']:.6g}",
                f"{m.get('lam_C_h020', float('nan')):.6g}", f"{d['gruneisen']:.6g}",
                f"{gruneisen_from_coupling(d['lam_B'], d['omega_use']):.6g}",
                f"{gruneisen_from_coupling(d['lam_A'], d['omega_use']):.6g}", d["sector"], f"{d['omega_use']:.6g}",
                f"{d['gamma_hwhm']:.6g}", f"{giso:.6g}", f"{d['n_n_plus_1']:.6g}", f"{d['tau_s'] * 1e12:.6g}",
                f"{d['eta_contrib']:.6g}", r, f"{c:.6g}"]
        lines.append(",".join(str(v) for v in vals))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(path)


def append_stage_c_note(text: str) -> None:
    path = REPO / "data" / "processed" / "reports" / "eta_SrTiO3_stageC.md"
    with open(path, "a") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def main() -> None:
    vogt, vogt_range = load_vogt()
    temps = [T for T in (200, 250, 300, 350, 400) if surface_available(T)]
    missing = [T for T in (200, 250, 300, 350, 400) if T not in temps]
    print(f"production model: {PROD['descriptor']}, construction {CONSTRUCTION}, h = {H_ENG}; "
          f"Vogt series covers T in [{vogt_range[0]:.0f}, {vogt_range[1]:.0f}] K")
    print(f"temperatures with converged strained SCPH sets and RTA: {temps}; not available (not run or not converged): {missing}")
    header = ("T_K,eta_total_Pas,eta_imaginary_Pas,eta_0_50_Pas,eta_50_100_Pas,eta_100_175_Pas,eta_175_inf_Pas,"
              "eta_gamma_sector_Pas,frac_below_100_cm1,vogt_gamma_sector,n_skipped,provisional_method_flag")
    out_rows = [header]
    results = {}
    modes300 = surface300 = None
    for T in temps:
        surface = load_construction_surface(T, with_h020=(T == 300))
        modes = construction_modes(T, 11, surface)
        eta, sec, flags, details = assemble_construction(T, modes=modes, surface=surface, return_details=True, vogt=vogt)
        low = sec["imaginary"] + sec["[0-50)"] + sec["[50-100)"] + sec["gamma_sector"]
        provisional = "no" if T == 300 else "linewidths_validated_at_300K_only"
        results[T] = (eta, sec, flags, details)
        if T == 300:
            modes300, surface300 = modes, surface
        print(f"T={T:3d} K: eta = {eta:.4e} Pa s  [imag {sec['imaginary']:.2e} | [0,50) {sec['[0-50)']:.2e} | "
              f"[50,100) {sec['[50-100)']:.2e} | [100,175) {sec['[100-175)']:.2e} | >=175 {sec['[175-inf)']:.2e} | "
              f"Gamma {sec['gamma_sector']:.2e}]  frac<100 {low / eta:.3f}  vogt={'y' if flags['vogt_used'] else 'FALLBACK'} "
              f"skipped={flags['n_skipped']} {provisional}")
        out_rows.append(f"{T},{eta:.6e},{sec['imaginary']:.6e},{sec['[0-50)']:.6e},{sec['[50-100)']:.6e},"
                        f"{sec['[100-175)']:.6e},{sec['[175-inf)']:.6e},{sec['gamma_sector']:.6e},{low / eta:.4f},"
                        f"{int(flags['vogt_used'])},{flags['n_skipped']},{provisional}")
    eta300, _, _, details300 = results[300]
    print(f"\neta(300 K) = {eta300:.4e} Pa s ({PRODUCTION_SURFACE}, construction C, h = {H_ENG}; finite-mesh result, convergence unresolved)")
    for cons in ("B", "A"):
        e, _, _ = assemble_construction(300, construction=cons, modes=modes300, surface=surface300, vogt=vogt)
        print(f"  construction {cons} on the same surface: {e:.4e} Pa s")
    mesh_etas = {}
    for n in (9, 11, 13, 15):
        if n == 11:
            mesh_etas[n] = eta300
            continue
        e, _, _ = assemble_construction(300, mesh_n=n, surface=surface300, vogt=vogt)
        mesh_etas[n] = e
        print(f"  mesh {n}^3: eta_C(300 K) = {e:.4e} Pa s")
    checks = sanity_checks(details300, mesh_etas, surface300, modes300)
    export_mode_table(details300, modes300, REPO / "data" / "processed" / "v3_diagnostics" / "mode_table_SrTiO3_300K_v6.csv",
                      eta300, 300)

    out = REPO / "data" / "processed" / "eta_SrTiO3.csv"
    head = [
        "# eta_SrTiO3.csv - produced by scripts/compute_eta_SrTiO3.py (hybrid model tut_z_od1)",
        "# eta_xyxy(T), Pa s. Hybrid model: ALAMODE v1.5.0 example harmonic set re-expressed on the 4x4x4 supercell +",
        "# QE-PBEsol strain perturbation K_QE; example anharmonic set (IFC supercell 2x2x2); SCPH with SELF_OFFDIAG = 1,",
        "# correction mesh 2x2x2 / inner mesh 12x12x12 (finest completed mesh; convergence of eta not established).",
        "# Only temperatures with converged unstrained and eps_xy = +-0.005 SCPH solutions and an own RTA are listed.",
        "# Coupling Lambda_C = e_r K_SCPH e_r with K_SCPH = [D(+h)-D(-h)]/(2h),",
        "# h = 0.010 engineering shear; gamma = -Lambda_C/(2 omega_r^2); diagonal projection for distinct modes, block",
        "# diagonalisation only inside exactly degenerate eigenspaces (1e-3 cm^-1); lifetime = stress-correlator kernel",
        "# 1/(2G) + 2G/w^2; linewidths from the same-T own 8^3 RTA, averaged over exactly degenerate sets, through the",
        "# character-aware frequency-class map; Gamma-point TO1 triplet from Vogt 1995",
        "# (the one exception to the single-surface rule). Sectors = bare-omega_0 bin of the max-overlap bare partner.",
        "# Harmonic strained sets (constructions A, B) in the ALAMODE convention (translationally invariant strain derivative).",
        "# See data/processed/reports/eta_SrTiO3_stageC.md.",
    ]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(head) + "\n" + "\n".join(out_rows) + "\n")
    tmp.rename(out)
    print(f"-> {out.relative_to(REPO)}")
    append_stage_c_note(
        f"## Production model {PRODUCTION_SURFACE} (hybrid model, construction C)\n\n"
        f"eta(300 K) = {eta300:.4e} Pa s; eta(T) for T = {', '.join(str(t) for t in temps)} K: "
        + ", ".join(f"{results[t][0]:.3e}" for t in temps) + " Pa s. Mesh 9/11/13/15: "
        + ", ".join(f"{mesh_etas[n]:.3e}" for n in sorted(mesh_etas)) + " Pa s (spread "
        f"{checks.get('mesh_spread_pct', float('nan')):.2f} %). Numbers only; provenance in the script header.")


if __name__ == "__main__":
    main()
