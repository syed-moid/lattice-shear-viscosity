#!/usr/bin/env python3
"""coupling_kernel.py - shared machinery for Steps 6.1b, 6.3 and 6.4 .

Strain variable (H3): engineering shear h = eps_xy + eps_yx = 2 s; the QE cells at s = +-0.005
carry h = +-0.010, those at s = +-0.010 carry h = +-0.020.  d/dh equals the nine-component
tensor derivative d/d eps_xy, so every Lambda below is dD/dh in cm^-2 per unit engineering
shear and gamma = -Lambda / (2 omega_r^2) is the tensor mode Grueneisen parameter of the
production convention (D1).  The identity is stated here once; nothing is divided elsewhere.

Sets (H5): harmonic force constants from the QE 4x4x4 q2r grid (own PBEsol set); quartic force
constants from the ALAMODE example (IFC supercell 2x2x2, fixed); SCPH on KMESH_INTERPOLATE
2x2x2 / KMESH_SCPH 2x2x2, 300 K, SELF_OFFDIAG = 0 ("od0") or 1 ("od1").

Constructions at one q, in the OWN-surface unstrained renormalised basis {e_r, omega_r}:
  A : Lambda_A(nu) = Lambda_bare(mu*) of the bare mode mu* with the largest overlap with nu
      (the Route-H basis transfer of the production model);
  B : e_nu^dagger K_HA e_nu, K_HA = [D_HA(+h) - D_HA(-h)] / (2h) from the QE strained sets;
  C : e_nu^dagger K_SCPH e_nu, K_SCPH from the strained own-surface SCPH renormalised sets.
Degenerate multiplets (|d omega| < 0.5 cm^-1): the projected block K_sub is diagonalised; its
eigenvalues are the multiplet couplings (basis independent, sum of squares = Tr K_sub^2, the
gauge-invariant quantity entering eta).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from dynmat import AlamodeSet, QESet, diagonalise, multiplet_groups  # noqa: E402
from latvisc.gruneisen import match_strain_pair_by_overlap  # noqa: E402

QE_DIR = REPO / "dft" / "qe" / "SrTiO3"
OWN6P2 = REPO / "data" / "raw" / "alamode_sto" / "own_surface_6p2"
FC = {
    "reference": QE_DIR / "dispersion_pbesol" / "SrTiO3_pbesol.444.fc",
    "shear_xy_p005": QE_DIR / "gruneisen_pbesol" / "shear_xy_p005" / "SrTiO3_pbesol_shear_xy_p005.444.fc",
    "shear_xy_m005": QE_DIR / "gruneisen_pbesol" / "shear_xy_m005" / "SrTiO3_pbesol_shear_xy_m005.444.fc",
    "shear_xy_p010": QE_DIR / "gruneisen_pbesol" / "shear_xy_p010" / "SrTiO3_pbesol_shear_xy_p010.444.fc",
    "shear_xy_m010": QE_DIR / "gruneisen_pbesol" / "shear_xy_m010" / "SrTiO3_pbesol_shear_xy_m010.444.fc",
}
H_STEP = {"005": 0.010, "010": 0.020}     # engineering shear h = 2 s of the strained cells
DEGEN_TOL = 0.5                           # cm-1, multiplet grouping
COUPLING_TOL = 1e-3                       # cm-1, exact degeneracy: block treatment for B and C only inside exact eigenspaces
NEAR_TOL = 5.0                            # cm-1, near-degenerate pairs reported via |K_nu mu|
VARIANTS = ("od0", "od1")


def hermitian(m):
    return 0.5 * (m + m.conj().T)


HARMONIC_CONVENTION = "alamode"
BARE_XML = {**{t: RAW / "own_od1" / "i2s8" / f"{t}_full_fc2.xml" for t in ("reference", "shear_xy_p005", "shear_xy_m005")},
            **{t: RAW / "own_surface_6p2" / f"{t}_full_fc2.xml" for t in ("shear_xy_p010", "shear_xy_m010")}}


class _HarmonicSet:
    def __init__(self, xml, fc):
        self.s = AlamodeSet(xml, fc)
        self.masses_amu = self.s.masses_amu

    def dynmat(self, q):
        return self.s.dynmat(q, asr_onsite=True)


class Sets:
    """All force-constant sets needed for the three constructions (one SCPH variant)."""

    def __init__(self, variant="od0", asr_onsite_scph=True, strained_scph=True, surface_dir=None, xml_pattern=None):
        """surface_dir: directory of the renormalised XMLs; xml_pattern: format with {tag} and {variant}
        (default 'renorm_own_{tag}_{variant}_300K.xml' in own_surface_6p2)."""
        self.variant = variant
        surface_dir = OWN6P2 if surface_dir is None else Path(surface_dir)
        xml_pattern = xml_pattern or "renorm_own_{tag}_{variant}_300K.xml"
        self.asr_onsite_scph = asr_onsite_scph
        tags = ["reference", "shear_xy_p005", "shear_xy_m005", "shear_xy_p010", "shear_xy_m010"]
        # harmonic sets in the production (ALAMODE, translationally invariant) convention; BARE_XML maps tag -> FC2 XML
        self.bare = {t: (_HarmonicSet(BARE_XML[t], FC[t]) if HARMONIC_CONVENTION == "alamode" and t in BARE_XML else QESet(FC[t]))
                     for t in tags}
        scph_tags = tags if strained_scph else ["reference"]
        self.scph = {t: AlamodeSet(surface_dir / xml_pattern.format(tag=t, variant=variant), FC[t]) for t in scph_tags
                     if (surface_dir / xml_pattern.format(tag=t, variant=variant)).exists()}
        if "reference" not in self.scph:
            raise FileNotFoundError(surface_dir / xml_pattern.format(tag="reference", variant=variant))
        self.masses = self.bare["reference"].masses_amu

    def D_HA(self, q, tag="reference"):
        return self.bare[tag].dynmat(q)

    def D_SCPH(self, q, tag="reference", asr_onsite=None):
        asr = self.asr_onsite_scph if asr_onsite is None else asr_onsite
        return self.scph[tag].dynmat(q, asr_onsite=asr)

    def K_HA(self, q, step="005"):
        h = H_STEP[step]
        return hermitian((self.D_HA(q, f"shear_xy_p{step}") - self.D_HA(q, f"shear_xy_m{step}")) / (2.0 * h))

    def K_SCPH(self, q, step="005", asr_onsite=None):
        h = H_STEP[step]
        return hermitian((self.D_SCPH(q, f"shear_xy_p{step}", asr_onsite) -
                          self.D_SCPH(q, f"shear_xy_m{step}", asr_onsite)) / (2.0 * h))


def project(K, omega, E, tol=DEGEN_TOL, near=NEAR_TOL):
    """Project K onto the eigenbasis E (columns).  Returns dict of per-mode arrays:
    diag   e_nu^dagger K e_nu;
    eig    multiplet-block eigenvalues (ascending within each multiplet; equal to diag for singlets);
    trk2   Tr(K_sub^2) of the mode's multiplet;
    gid, gsize   multiplet id and size;
    offdiag  max |K_nu mu| over other modes mu within `near` cm-1 but outside the multiplet."""
    Kb = hermitian(E.conj().T @ K @ E)
    n = len(omega)
    diag = np.real(np.diag(Kb))
    eig = np.zeros(n)
    trk2 = np.zeros(n)
    gid = np.zeros(n, dtype=int)
    gsize = np.zeros(n, dtype=int)
    for k, g in enumerate(multiplet_groups(omega, tol)):
        lam = np.linalg.eigvalsh(Kb[np.ix_(g, g)])
        eig[g] = lam
        trk2[g] = float((lam ** 2).sum())
        gid[g] = k
        gsize[g] = len(g)
    offdiag = np.zeros(n)
    for i in range(n):
        m = (np.abs(omega - omega[i]) < near) & (gid != gid[i])
        offdiag[i] = np.abs(Kb[i, m]).max() if m.any() else 0.0
    return {"diag": diag, "eig": eig, "trk2": trk2, "gid": gid, "gsize": gsize, "offdiag": offdiag}


def transfer_A(z_bare, lam_bare, z_ren, omega_ren, tol=DEGEN_TOL):
    """Route-H basis transfer.  z arrays hold modes as rows.
    Returns Lambda_A per renormalised mode, the bare partner index, its single overlap and the
    overlap summed over the renormalised multiplet, plus the production map bare -> ren
    (index of the best renormalised mode for each bare mode, multiplet-summed overlap)."""
    ov = np.abs(z_bare.conj() @ z_ren.T) ** 2            # (n_bare, n_ren)
    mu_star = ov.argmax(axis=0)
    lam_A = lam_bare[mu_star]
    single = ov[mu_star, np.arange(ov.shape[1])]
    summed = np.zeros(ov.shape[1])
    for g in multiplet_groups(omega_ren, tol):
        for nu in g:
            summed[nu] = ov[mu_star[nu], g].sum()
    prod_index = ov.argmax(axis=1)
    prod_overlap = np.zeros(ov.shape[0])
    for mu in range(ov.shape[0]):
        grp = np.abs(omega_ren - omega_ren[prod_index[mu]]) < tol
        prod_overlap[mu] = ov[mu, grp].sum()
    return lam_A, mu_star, single, summed, prod_index, prod_overlap


def signed_sq(w):
    return np.sign(w) * w * w


def evaluate_q(sets: Sets, q, steps=("005", "010"), matched_C=True):
    """Everything at one q for one SCPH variant."""
    q = np.asarray(q, dtype=float)
    wb, Eb = diagonalise(sets.D_HA(q))
    wr, Er = diagonalise(sets.D_SCPH(q))
    out = {"q": q, "omega_b": wb, "omega_r": wr, "E_b": Eb, "E_r": Er}
    K_HA = {s: sets.K_HA(q, s) for s in steps}
    K_SC = {s: sets.K_SCPH(q, s) for s in steps if f"shear_xy_p{s}" in sets.scph}
    if "005" in steps and "010" in steps:
        K_HA["rich"] = (4.0 * K_HA["005"] - K_HA["010"]) / 3.0
        if "005" in K_SC and "010" in K_SC:
            K_SC["rich"] = (4.0 * K_SC["005"] - K_SC["010"]) / 3.0
    out["bare"] = {s: project(K_HA[s], wb, Eb) for s in K_HA}
    out["B"] = {s: project(K_HA[s], wr, Er, tol=COUPLING_TOL) for s in K_HA}
    out["C"] = {s: project(K_SC[s], wr, Er, tol=COUPLING_TOL) for s in K_SC}
    if "shear_xy_p005" in sets.scph:
        out["C_noasr"] = project(sets.K_SCPH(q, "005", asr_onsite=False), wr, Er, tol=COUPLING_TOL)
    lam_A, mu_star, single, summed, prod_index, prod_overlap = transfer_A(Eb.T, out["bare"]["005"]["eig"], Er.T, wr)
    out["A"] = {"lam": lam_A, "mu_star": mu_star, "overlap_single": single, "overlap_summed": summed,
                "prod_index": prod_index, "prod_overlap": prod_overlap}
    if matched_C and "shear_xy_p005" in sets.scph:
        nat = len(sets.masses)
        vec_ref = (Er.T / np.sqrt(np.repeat(sets.masses, 3))[None, :]).reshape(-1, nat, 3)
        out["C_matched"] = {}
        for s in steps:
            if f"shear_xy_p{s}" not in sets.scph:
                continue
            wp, Ep = diagonalise(sets.D_SCPH(q, f"shear_xy_p{s}"))
            wm, Em = diagonalise(sets.D_SCPH(q, f"shear_xy_m{s}"))
            vp = (Ep.T / np.sqrt(np.repeat(sets.masses, 3))[None, :]).reshape(-1, nat, 3)
            vm = (Em.T / np.sqrt(np.repeat(sets.masses, 3))[None, :]).reshape(-1, nat, 3)
            mp, mm = match_strain_pair_by_overlap(wr, vec_ref, wp, vp, wm, vm, sets.masses, DEGEN_TOL)
            out["C_matched"][s] = (signed_sq(mp) - signed_sq(mm)) / (2.0 * H_STEP[s])
    return out


def gruneisen(lam, omega):
    """gamma = -Lambda / (2 omega^2) (tensor convention, engineering-shear derivative)."""
    return -lam / (2.0 * omega * omega)


def acoustic_character(E, masses, q_dir):
    """LA/TA label of each mode (columns of E) from the centre-of-mass displacement direction:
    u_cm ~ sum_a sqrt(m_a) e_a.  Returns (label list, |u_cm . q_hat|^2 / |u_cm|^2, |u_cm|^2 fraction)."""
    nat = len(masses)
    qhat = np.asarray(q_dir, dtype=float)
    qhat = qhat / np.linalg.norm(qhat)
    labels, proj, frac = [], [], []
    for nu in range(E.shape[1]):
        e = E[:, nu].reshape(nat, 3)
        u = (np.sqrt(masses)[:, None] * e).sum(axis=0)
        norm = np.vdot(u, u).real
        p = abs(np.vdot(qhat, u)) ** 2 / norm if norm > 1e-12 else 0.0
        f = norm / masses.sum()          # 1 for a pure translation
        proj.append(p)
        frac.append(f)
        labels.append("LA" if p > 0.5 else "TA")
    return labels, np.array(proj), np.array(frac)
