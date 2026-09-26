#!/usr/bin/env python3
"""dynmat.py - one mass-scaled Cartesian dynamical-matrix builder for two force-constant
formats, with one set of conventions .

(i)  QESet     : Quantum ESPRESSO q2r ``.fc`` file, matdyn algorithm
                 (Wigner-Seitz-weighted short-range Fourier sum, ``frc_blk``)
                 + rigid-ion non-analytic term at the given q (``rgd_blk`` port,
                 converter).  The distributed minimal-norm acoustic sum
                 rule of the converter stands in for matdyn's asr='crystal'.
(ii) AlamodeSet: ALAMODE FC2 XML (bare converted set or SCPH-renormalised dfc2 output)
                 summed exactly as anphon ``Dynamical::calc_analytic_k`` (minimum-image
                 cells with 1/multiplicity weights, phase exp(+i q.R) with R the cell of
                 the second atom), minus the rigid-ion dipole force constants that the
                 converter restored on the q2r grid, plus the same ``rgd_blk`` term at q.

Conventions (identical for both builders):
  * atom order  : QE order (Sr, Ti, O1, O2, O3); a permutation can be given for
                  XML files in the ALAMODE tutorial order;
  * masses      : those of the QE ``.fc`` file (Sr 87.62, Ti 47.867, O 15.999 amu) unless
                  ``masses_amu`` is given (anphon's table has O = 15.9994);
  * phases      : lattice-vector-only phases, exp(-i q.R) with R the cell of the FIRST
                  atom in the QE storage, which equals anphon's exp(+i q.R) with R the
                  cell of the SECOND atom ;
  * q           : fractional (crystal) reciprocal coordinates of the set's OWN primitive
                  cell; the Cartesian 2 pi/alat vector for the non-analytic term is
                  bg @ q_frac with the set's own reciprocal vectors;
  * unit of D   : cm^-2 (eigenvalues are omega^2 in cm^-2, negative for imaginary modes);
  * NA term     : each set's own dielectric tensor and Born charges.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
from q2r_to_alamode_fc2 import (  # noqa: E402
    add_dipole_real_space, build_supercell, image_shifts, impose_asr_distributed,
    minimum_image_cells, read_qe_fc, rigid_ion_dynmat,
)

RY_TO_CMM1 = 109737.31568          # QE constants module (1 Ry in cm^-1)
AMU_RY = 911.44424310              # QE constants module (amu in Rydberg mass units)
WS_EPS = 1.0e-6                    # QE ws_base eps (alat^2 units)
WS_NX = 2                          # QE ws_base wsinit shell count
NAT = 5
_FLOAT = re.compile(r"[-+]?\d+\.\d+[eE][-+]?\d+|[-+]?\d+\.?\d*")


def _floats(text):
    """dfc2 writes negative exponents fused to the previous number; split on the float pattern."""
    return [float(x) for x in _FLOAT.findall(text)]


# ----------------------------------------------------------------------------- helpers
def signed_sqrt_cm1(eigenvalues_cm2):
    return np.sign(eigenvalues_cm2) * np.sqrt(np.abs(eigenvalues_cm2))


def diagonalise(dmat_cm2):
    """Eigen-decomposition of a Hermitian D (cm^-2): (omega_cm1 sorted, evec[:, mode])."""
    dh = 0.5 * (dmat_cm2 + dmat_cm2.conj().T)
    lam, vec = np.linalg.eigh(dh)
    return signed_sqrt_cm1(lam), vec


def mass_scale_matrix(masses_amu_per_atom):
    m = np.repeat(np.asarray(masses_amu_per_atom, dtype=float), 3)
    return 1.0 / np.sqrt(np.outer(m, m))


def ry_per_bohr2_to_cm2(dyn_ry_bohr2, masses_amu_per_atom):
    """Mass-scale a force-constant-unit matrix (Ry/bohr^2, index 3*atom+xyz) into cm^-2."""
    return dyn_ry_bohr2 * mass_scale_matrix(masses_amu_per_atom) / AMU_RY * RY_TO_CMM1 ** 2


def ws_init(atws_columns):
    """QE ws_base::wsinit - the lattice vectors defining the Wigner-Seitz cell."""
    rws, half = [], []
    for ir in range(-WS_NX, WS_NX + 1):
        for jr in range(-WS_NX, WS_NX + 1):
            for kr in range(-WS_NX, WS_NX + 1):
                r = ir * atws_columns[:, 0] + jr * atws_columns[:, 1] + kr * atws_columns[:, 2]
                r0 = 0.5 * float(r @ r)
                if r0 > WS_EPS:
                    rws.append(r)
                    half.append(r0)
    return np.array(rws), np.array(half)


def ws_weights(rvec, rws, half):
    """QE ws_base::wsweight vectorised over rows of rvec."""
    ck = rvec @ rws.T - half[None, :]
    outside = (ck > WS_EPS).any(axis=1)
    nreq = 1 + (np.abs(ck) < WS_EPS).sum(axis=1)
    return np.where(outside, 0.0, 1.0 / nreq)


# ----------------------------------------------------------------------------- (i) QE .fc
class QESet:
    """matdyn-algorithm dynamical matrix from a q2r .fc file."""

    def __init__(self, fc_path, asr="distributed", masses_amu=None):
        data = read_qe_fc(Path(fc_path))
        nat, nq = data["nat"], data["nq"]
        if asr == "distributed":
            fc, self.asr_before, self.asr_after, self.asr_iterations = impose_asr_distributed(data["fc"], nat, nq)
        elif asr == "none":
            fc = data["fc"]
        else:
            raise ValueError(asr)
        self.data = dict(data, fc=fc)
        self.data_raw = data
        self.nat, self.nq = nat, list(nq)
        self.alat = data["celldm"][0]
        self.at = data["lavec"] / self.alat                   # columns, alat units
        self.bg = np.linalg.inv(self.at).T                    # columns, 2 pi/alat
        self.tau = data["x_alat"]
        self.masses_amu = (np.array(data["masses_ry"])[data["kd"] - 1] / AMU_RY
                           if masses_amu is None else np.asarray(masses_amu, dtype=float))
        ncell = nq[0] * nq[1] * nq[2]
        # fcb[na, nb, icell, i, j] = frc(m; i, j, na, nb) of matdyn
        self.fcb = fc.reshape(nat, 3, nat, 3, ncell).transpose(2, 0, 4, 3, 1)
        rws, half = ws_init(self.at * np.array(nq)[None, :])
        n1 = np.arange(-2 * nq[0], 2 * nq[0] + 1)
        n2 = np.arange(-2 * nq[1], 2 * nq[1] + 1)
        n3 = np.arange(-2 * nq[2], 2 * nq[2] + 1)
        grid = np.array(np.meshgrid(n1, n2, n3, indexing="ij")).reshape(3, -1).T   # (N, 3)
        rvec = grid @ self.at.T                                                   # (N, 3) alat units
        m = np.mod(grid, np.array(nq)[None, :])
        icell = m[:, 0] + nq[0] * (m[:, 1] + nq[1] * m[:, 2])
        self.entries = {}
        for na in range(nat):
            for nb in range(nat):
                w = ws_weights(rvec + self.tau[na] - self.tau[nb], rws, half)
                keep = w > 0.0
                if abs(w.sum() - ncell) > 1e-8:
                    raise RuntimeError(f"wrong total WS weight for pair ({na},{nb}): {w.sum()}")
                self.entries[(na, nb)] = (rvec[keep], w[keep], icell[keep])

    def q_cart(self, q_frac):
        return self.bg @ np.asarray(q_frac, dtype=float)

    def short_range(self, q_frac):
        """Wigner-Seitz Fourier sum, Ry/bohr^2, index 3*atom+xyz."""
        q = self.q_cart(q_frac)
        dyn = np.zeros((self.nat, 3, self.nat, 3), dtype=complex)
        for (na, nb), (rvec, w, icell) in self.entries.items():
            phase = w * np.exp(-2j * np.pi * (rvec @ q))
            dyn[na, :, nb, :] = np.einsum("e,eij->ij", phase, self.fcb[na, nb, icell])
        return dyn.reshape(3 * self.nat, 3 * self.nat)

    def nonanalytic(self, q_frac):
        if self.data["dielectric"] is None:
            return np.zeros((3 * self.nat, 3 * self.nat), dtype=complex)
        return rigid_ion_dynmat(self.data, self.q_cart(q_frac)).reshape(3 * self.nat, 3 * self.nat)

    def dynmat(self, q_frac):
        """Mass-scaled dynamical matrix in cm^-2 at fractional q."""
        return ry_per_bohr2_to_cm2(self.short_range(q_frac) + self.nonanalytic(q_frac), self.masses_amu)


# ----------------------------------------------------------------------------- (ii) ALAMODE XML
class AlamodeSet:
    """anphon-algorithm dynamical matrix from an ALAMODE FC2 XML.

    fc_source: the q2r .fc file the XML descends from (its dielectric block, Born
    charges, primitive cell and grid are needed to remove the restored dipole
    force constants and to evaluate the non-analytic term at q).
    perm: primitive-atom permutation mapping XML atom order -> QE order
    (qe_atom = perm.index(xml_atom)); None for converter-generated XMLs.
    """

    def __init__(self, xml_path, fc_source, masses_amu=None, perm=None, subtract_dipole=True):
        root = ET.parse(Path(xml_path)).getroot()
        st = root.find("Structure")
        nat_s = int(st.find("NumberOfAtoms").text)
        lv = st.find("LatticeVector")
        self.lavec_s = np.array([_floats(lv.find(f"a{i}").text) for i in (1, 2, 3)]).T
        self.x_s = np.zeros((nat_s, 3))
        self.elem_s = []
        for pos in st.find("Position"):
            self.x_s[int(pos.get("index")) - 1] = _floats(pos.text)
            self.elem_s.append(pos.get("element"))
        trans = root.find("Symmetry").find("Translations")
        ntran = int(root.find("Symmetry").find("NumberOfTranslations").text)
        nat = nat_s // ntran
        self.map_p2s = np.zeros((ntran, nat), dtype=int)
        for mp in trans:
            self.map_p2s[int(mp.get("tran")) - 1, int(mp.get("atom")) - 1] = int(mp.text) - 1
        self.map_s2p = np.zeros(nat_s, dtype=int)
        for t in range(ntran):
            for j in range(nat):
                self.map_s2p[self.map_p2s[t, j]] = j
        self.nat = nat
        self.qe = QESet(fc_source, asr="none", masses_amu=masses_amu)
        self.alat, self.at, self.bg, self.nq = self.qe.alat, self.qe.at, self.qe.bg, self.qe.nq
        lavec_p_expected = self.qe.data["lavec"] * np.array(self.nq)[None, :]
        if not np.allclose(self.lavec_s, lavec_p_expected, atol=1e-6):
            raise ValueError("XML supercell does not match the .fc primitive cell times the q2r grid")
        self.perm = list(perm) if perm is not None else None
        # XML atom -> QE atom
        self.to_qe = np.arange(nat) if perm is None else np.array([self.perm.index(a) for a in range(nat)])
        if masses_amu is None:
            self.masses_amu = self.qe.masses_amu
        else:
            self.masses_amu = np.asarray(masses_amu, dtype=float)
        shifts = image_shifts()
        a1, x1, a2s, x2, cell, val = [], [], [], [], [], []
        for fc2 in root.find("ForceConstants").find("HARMONIC"):
            p1 = fc2.get("pair1").split()
            p2 = fc2.get("pair2").split()
            a1.append(int(p1[0]) - 1); x1.append(int(p1[1]) - 1)
            a2s.append(int(p2[0]) - 1); x2.append(int(p2[1]) - 1); cell.append(int(p2[2]) - 1)
            val.append(float(fc2.text))
        a1, x1, a2s, x2, cell = (np.array(v) for v in (a1, x1, a2s, x2, cell))
        self.val = np.array(val)
        a2p = self.map_s2p[a2s]
        vec_s = self.x_s[a2s] + shifts[cell] - self.x_s[self.map_p2s[0, a2p]]     # supercell fractional
        self.rfrac_p = vec_s * np.array(self.nq)[None, :]                          # primitive fractional
        self.row = 3 * self.to_qe[a1] + x1
        self.col = 3 * self.to_qe[a2p] + x2
        self.n_entries = len(self.val)
        # dipole force constants restored on the grid, in the same (row, col, image) layout
        self.subtract_dipole = subtract_dipole and self.qe.data["dielectric"] is not None
        if self.subtract_dipole:
            zero = dict(self.qe.data_raw, fc=np.zeros_like(self.qe.data_raw["fc"]))
            fc_dip = add_dipole_real_space(zero)["fc"]
            x_super, map_p2s, lavec_s = build_supercell(self.qe.data_raw)
            if not (np.allclose(x_super, self.x_s, atol=1e-9) and np.array_equal(map_p2s, self.map_p2s)):
                raise ValueError("XML supercell atom layout differs from the converter's enumeration")
            mind = minimum_image_cells(x_super, map_p2s, lavec_s, nat)
            ncell = self.nq[0] * self.nq[1] * self.nq[2]
            drow, dcol, dr, dv = [], [], [], []
            for iat in range(nat):
                for icrd in range(3):
                    for icell in range(ncell):
                        for jat in range(nat):
                            kat = map_p2s[icell, jat]
                            cells = mind[(iat, kat)]
                            for jcrd in range(3):
                                v = fc_dip[3 * iat + icrd, 3 * jat + jcrd, icell]
                                for c in cells:
                                    drow.append(3 * iat + icrd); dcol.append(3 * jat + jcrd)
                                    dr.append((x_super[kat] + shifts[c] - x_super[map_p2s[0, jat]]) * np.array(self.nq))
                                    dv.append(v / len(cells))
            self.dip_row, self.dip_col = np.array(drow), np.array(dcol)
            self.dip_rfrac, self.dip_val = np.array(dr), np.array(dv)

    @staticmethod
    def _fourier(row, col, rfrac, val, q_frac, n):
        phase = np.exp(2j * np.pi * (rfrac @ np.asarray(q_frac, dtype=float)))
        dyn = np.zeros((n, n), dtype=complex)
        np.add.at(dyn, (row, col), val * phase)
        return dyn

    def xml_sum(self, q_frac):
        """Plain anphon sum of the XML force constants (Ry/bohr^2)."""
        return self._fourier(self.row, self.col, self.rfrac_p, self.val, q_frac, 3 * self.nat)

    def dipole_sum(self, q_frac):
        if not self.subtract_dipole:
            return np.zeros((3 * self.nat, 3 * self.nat), dtype=complex)
        return self._fourier(self.dip_row, self.dip_col, self.dip_rfrac, self.dip_val, q_frac, 3 * self.nat)

    def onsite_asr_correction(self):
        """On-site acoustic-sum-rule correction (Ry/bohr^2, block-diagonal): for each atom a and
        Cartesian pair (i, j), sum over b of the analytic D(q = 0)[a i, b j]; subtracting it from
        D[a i, a j] makes the uniform translation an exact null vector.  Used for the SCPH-
        renormalised sets, whose correction is not projected onto the sum rule by anphon."""
        if not hasattr(self, "_asr_corr"):
            d0 = self.xml_sum((0.0, 0.0, 0.0)) - self.dipole_sum((0.0, 0.0, 0.0))
            corr = np.zeros_like(d0)
            for a in range(self.nat):
                for b in range(self.nat):
                    corr[3 * a:3 * a + 3, 3 * a:3 * a + 3] += d0[3 * a:3 * a + 3, 3 * b:3 * b + 3]
            self._asr_corr = corr
        return self._asr_corr

    def dynmat(self, q_frac, nonanalytic=True, asr_onsite=False):
        """Mass-scaled dynamical matrix in cm^-2 at fractional q."""
        dyn = self.xml_sum(q_frac)
        if self.subtract_dipole and nonanalytic:
            dyn = dyn - self.dipole_sum(q_frac) + self.qe.nonanalytic(q_frac)
        if asr_onsite:
            dyn = dyn - self.onsite_asr_correction()
        return ry_per_bohr2_to_cm2(dyn, self.masses_amu)


# ----------------------------------------------------------------------------- eigenvector utilities
def modes_to_polarisation(vectors, masses_amu):
    """matdyn .modes displacement patterns (nmodes, nat, 3) -> normalised mass-scaled
    polarisation vectors z[mode, 3*atom+xyz]."""
    z = (vectors * np.sqrt(np.asarray(masses_amu))[None, :, None]).reshape(vectors.shape[0], -1)
    return z / np.linalg.norm(z, axis=1)[:, None]


def multiplet_groups(omega_cm1, tol=0.5):
    groups, start = [], 0
    for i in range(1, len(omega_cm1) + 1):
        if i == len(omega_cm1) or omega_cm1[i] - omega_cm1[i - 1] > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def multiplet_overlaps(omega_ref, z_ref, z_test, tol=0.5):
    """Per-multiplet subspace overlap ||Z_ref^H Z_test||_F^2 / d in [0, 1]; z[mode, comp]."""
    out = []
    for g in multiplet_groups(omega_ref, tol):
        a, b = z_ref[g], z_test[g]
        out.append(float(np.linalg.norm(a.conj() @ b.T) ** 2 / len(g)))
    return out


# ----------------------------------------------------------------------------- fast rigid-ion term
class RigidIon:
    """Vectorised port of rigid_ion_dynmat (same G sum, same alpha = 1, gmax = 14, e2 = 2).
    The q-independent term is precomputed; dynmat(q_2pi_alat) returns dyn[3*na+i, 3*nb+j] in Ry/bohr^2."""

    def __init__(self, data):
        nat = data["nat"]
        alat = data["celldm"][0]
        at = data["lavec"] / alat
        bg = np.linalg.inv(at).T
        omega = abs(np.linalg.det(data["lavec"]))
        self.nat, self.tau, self.eps, self.zeu = nat, data["x_alat"], data["dielectric"], data["born"]
        nr = data["nq"]
        self.gmax, self.alph = 14.0, 1.0
        geg0 = self.gmax * self.alph * 4.0
        nrx = [0 if nr[p] == 1 else int(np.sqrt(geg0) / np.linalg.norm(bg[:, p])) + 1 for p in range(3)]
        m = np.array(np.meshgrid(*[np.arange(-n, n + 1) for n in nrx], indexing="ij")).reshape(3, -1).T
        self.g = m @ bg.T                                   # (Ng, 3)
        self.fac = 2.0 * 4.0 * np.pi / omega
        self.dtau = self.tau[:, None, :] - self.tau[None, :, :]   # (nat, nat, 3)
        # q-independent part
        g = self.g
        geg = np.einsum("gi,ij,gj->g", g, self.eps, g)
        keep = (geg > 0.0) & (geg / self.alph / 4.0 < self.gmax)
        g, geg = g[keep], geg[keep]
        facgd = self.fac * np.exp(-geg / self.alph / 4.0) / geg
        zg = np.einsum("gi,aij->gaj", g, self.zeu)          # (Ng, nat, 3)
        cosarg = np.cos(2.0 * np.pi * np.einsum("gk,abk->gab", g, self.dtau))
        fnat = np.einsum("gbj,gab->gaj", zg, cosarg)        # (Ng, nat, 3)
        self.static = -np.einsum("g,gai,gaj->aij", facgd, zg, fnat)   # (nat, 3, 3)

    def dynmat(self, q_2pi_alat):
        nat = self.nat
        gq = self.g + np.asarray(q_2pi_alat, dtype=float)[None, :]
        geg = np.einsum("gi,ij,gj->g", gq, self.eps, gq)
        keep = (geg > 0.0) & (geg / self.alph / 4.0 < self.gmax)
        gq, geg = gq[keep], geg[keep]
        facgd = self.fac * np.exp(-geg / self.alph / 4.0) / geg
        zg = np.einsum("gi,aij->gaj", gq, self.zeu)
        phase = np.exp(2j * np.pi * np.einsum("gk,abk->gab", gq, self.dtau))
        dyn = np.einsum("g,gab,gai,gbj->aibj", facgd, phase, zg, zg)
        for a in range(nat):
            dyn[a, :, a, :] += self.static[a]
        return dyn.reshape(3 * nat, 3 * nat)


def _qeset_nonanalytic_fast(self, q_frac):
    if self.data["dielectric"] is None:
        return np.zeros((3 * self.nat, 3 * self.nat), dtype=complex)
    if not hasattr(self, "_rigid"):
        self._rigid = RigidIon(self.data)
    return self._rigid.dynmat(self.q_cart(q_frac))


QESet.nonanalytic_loop = QESet.nonanalytic
QESet.nonanalytic = _qeset_nonanalytic_fast
