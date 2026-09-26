"""Strain-coupling constructions on one SCPH surface (production coupling of the revised model).

Strain variable
---------------
The strained cells apply the symmetric shear ε_xy = ε_yx = s. The ENGINEERING shear h = ε_xy + ε_yx = 2s
is the derivative variable used here; the cells at s = ±0.005 carry h = ±0.010. Because the nine-component
tensor derivative d/dε_xy of a quantity that depends on the strain through the symmetric combination
ε_xy + ε_yx equals d/dh, every Λ = dω²/dh returned by this module is already the tensor-convention
coupling of the manuscript, and γ = −Λ/(2ω_r²) is the tensor mode Grüneisen parameter. This identity is
stated here once; no factor of two is applied anywhere else in the production path. (The historical
``path_to_tensor_shear`` of ``latvisc.gruneisen`` implements the same identity for the frequency-difference
route and is kept for the legacy bare-coupling comparison only.)

Constructions
-------------
Given the mass-scaled dynamical matrices D(q; +h) and D(q; −h) of two strained sets,
``strain_derivative_matrix`` forms K(q) = [D(+h) − D(−h)]/(2h) (Hermitian, cm⁻² per unit h), and
``project_coupling`` projects it on the eigenbasis {e_ν} of the unstrained reference: singlets give
Λ_ν = e_ν† K e_ν; degenerate multiplets (|Δω| < tol) are treated as a subspace, the projected block is
diagonalised and its eigenvalues are the multiplet couplings, whose sum of squares Tr K_sub² is the
gauge-invariant quantity that enters η. Applied to the SCPH-renormalised sets this is construction C:
the eigenvalue derivative of the differentiable SCPH dynamical matrix of the specified effective model
(fixed external anharmonic force constants); applied to the bare sets on the renormalised basis it is
construction B; the bare couplings transferred by maximum overlap (``transfer_bare_coupling``) are the
retired construction A.

Dynamical matrices
------------------
``QESet`` builds D(q) from a Quantum ESPRESSO q2r ``.fc`` file with the matdyn algorithm (Wigner–Seitz
weighted short-range sum plus the rigid-ion non-analytic term of ``rgd_blk``); ``AlamodeSet`` builds D(q)
from an ALAMODE FC2 XML (bare converted or SCPH-renormalised) with anphon's minimum-image sum, the
restored grid dipole force constants removed and the same non-analytic term added. Both use the QE atom
order, the masses of the ``.fc`` file, fractional q of the set's own cell, each set's own dielectric data,
and return D in cm⁻² (eigenvalues ω², negative for imaginary modes).
"""

from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

__all__ = [
    "RY_TO_CMM1", "AMU_RY", "ENGINEERING_SHEAR_PER_SYMMETRIC_STRAIN", "engineering_shear",
    "read_qe_fc", "impose_asr_distributed", "add_dipole_real_space", "QESet", "AlamodeSet",
    "diagonalise", "multiplet_groups", "strain_derivative_matrix", "project_coupling",
    "transfer_bare_coupling", "gruneisen_from_coupling", "acoustic_character", "acoustic_limit_check",
]

RY_TO_CMM1 = 109737.31568          # 1 Ry in cm^-1 (QE constants)
AMU_RY = 911.44424310              # atomic mass unit in Rydberg mass units (QE constants)
WS_EPS = 1.0e-6                    # QE ws_base tolerance (alat^2)
WS_NX = 2
EPS_FC = 1e-15
DIST_TOL = 1e-3                    # bohr, minimum-image tie tolerance (anphon convention)
ENGINEERING_SHEAR_PER_SYMMETRIC_STRAIN = 2.0
_FLOAT = re.compile(r"[-+]?\d+\.\d+[eE][-+]?\d+|[-+]?\d+\.?\d*")


def engineering_shear(symmetric_strain_amplitude):
    """h = 2 s for a cell strained by ε_xy = ε_yx = s (see the module docstring)."""
    return ENGINEERING_SHEAR_PER_SYMMETRIC_STRAIN * float(symmetric_strain_amplitude)


# ----------------------------------------------------------------------------- QE q2r .fc
def _tokens(path):
    for line in Path(path).read_text().splitlines():
        for tok in line.split():
            yield tok


def read_qe_fc(path):
    """Parse a q2r ``.fc`` file. Returns a dict with the primitive cell (columns = lattice vectors, bohr),
    species, positions, dielectric data, the grid and the force constants in the transposed storage
    fc[3*jat+jcrd, 3*iat+icrd, cell] = C(icrd, jcrd, iat, jat; cell) (Ry/bohr^2, m1 fastest cell order)."""
    it = _tokens(path)
    nkd, nat, ibrav = int(next(it)), int(next(it)), int(next(it))
    celldm = [float(next(it)) for _ in range(6)]
    if ibrav == 0:
        lavec = np.zeros((3, 3))
        for i in range(3):
            for c in range(3):
                lavec[c, i] = float(next(it))
        lavec *= celldm[0]
    elif ibrav == 1:
        lavec = celldm[0] * np.eye(3)
    else:
        raise NotImplementedError(f"ibrav = {ibrav} not supported (only 0 and 1)")
    symbols, masses = [], []
    for _ in range(nkd):
        next(it)
        sym = next(it)[1:]
        next(it)
        masses.append(float(next(it)))
        symbols.append(sym)
    kd = np.zeros(nat, dtype=int)
    x_alat = np.zeros((nat, 3))
    for i in range(nat):
        next(it)
        kd[i] = int(next(it))
        x_alat[i] = [float(next(it)) for _ in range(3)]
    x_frac = (np.linalg.inv(lavec) @ (x_alat * celldm[0]).T).T
    flag = next(it)
    dielectric, born = None, None
    if flag[0] == "T":
        dielectric = np.array([[float(next(it)) for _ in range(3)] for _ in range(3)])
        born = np.zeros((nat, 3, 3))
        for i in range(nat):
            next(it)
            born[i] = np.array([[float(next(it)) for _ in range(3)] for _ in range(3)])
    nq = [int(next(it)) for _ in range(3)]
    ncell = nq[0] * nq[1] * nq[2]
    fc = np.zeros((3 * nat, 3 * nat, ncell))
    for icrd in range(3):
        for jcrd in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    for _ in range(4):
                        next(it)
                    icell = 0
                    for _m3 in range(nq[2]):
                        for _m2 in range(nq[1]):
                            for _m1 in range(nq[0]):
                                next(it); next(it); next(it)
                                fc[3 * jat + jcrd, 3 * iat + icrd, icell] = float(next(it))
                                icell += 1
    return {"nkd": nkd, "nat": nat, "ibrav": ibrav, "celldm": celldm, "lavec": lavec, "symbols": symbols,
            "masses_ry": masses, "kd": kd, "x_frac": x_frac, "x_alat": x_alat, "dielectric": dielectric,
            "born": born, "nq": nq, "fc": fc}


def impose_asr_distributed(fc, nat, nq, tol=1e-12, max_iter=200):
    """Minimal-norm acoustic sum rule with pair symmetry (row sums over (jat, cell) driven to zero by an
    equal correction of every entry of the row, the pair symmetry re-imposed, alternated to tol).
    Returns (fc_new, residual_before, residual_after, n_iter)."""
    ncell = nq[0] * nq[1] * nq[2]
    c = np.zeros((3, 3, nat, nat, ncell))
    for iat in range(nat):
        for jat in range(nat):
            for icrd in range(3):
                for jcrd in range(3):
                    c[icrd, jcrd, iat, jat, :] = fc[3 * jat + jcrd, 3 * iat + icrd, :]
    cells = [(m1, m2, m3) for m3 in range(nq[2]) for m2 in range(nq[1]) for m1 in range(nq[0])]
    lookup = {m: k for k, m in enumerate(cells)}
    neg = np.array([lookup[((-m1) % nq[0], (-m2) % nq[1], (-m3) % nq[2])] for (m1, m2, m3) in cells])

    def residual(arr):
        return np.abs(arr.sum(axis=(3, 4))).max()

    before = residual(c)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        c = c - c.sum(axis=(3, 4), keepdims=True) / (nat * ncell)
        c = 0.5 * (c + np.transpose(c, (1, 0, 3, 2, 4))[..., neg])
        if residual(c) < tol:
            break
    after = residual(c)
    out = fc.copy()
    for iat in range(nat):
        for jat in range(nat):
            for icrd in range(3):
                for jcrd in range(3):
                    out[3 * jat + jcrd, 3 * iat + icrd, :] = c[icrd, jcrd, iat, jat, :]
    return out, before, after, n_iter


class RigidIon:
    """Vectorised port of QE ``rigid.f90::rgd_blk`` (3D, sign +1, alpha = 1, gmax = 14, e2 = 2).
    ``dynmat(q_2pi_alat)`` returns the dipole term dyn[3*na+i, 3*nb+j] in Ry/bohr^2."""

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
        self.g = m @ bg.T
        self.fac = 2.0 * 4.0 * np.pi / omega
        self.dtau = self.tau[:, None, :] - self.tau[None, :, :]
        g = self.g
        geg = np.einsum("gi,ij,gj->g", g, self.eps, g)
        keep = (geg > 0.0) & (geg / self.alph / 4.0 < self.gmax)
        g, geg = g[keep], geg[keep]
        facgd = self.fac * np.exp(-geg / self.alph / 4.0) / geg
        zg = np.einsum("gi,aij->gaj", g, self.zeu)
        cosarg = np.cos(2.0 * np.pi * np.einsum("gk,abk->gab", g, self.dtau))
        fnat = np.einsum("gbj,gab->gaj", zg, cosarg)
        self.static = -np.einsum("g,gai,gaj->aij", facgd, zg, fnat)

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


def add_dipole_real_space(data):
    """Copy of ``data`` whose force constants are the full real-space constants (short-range + rigid-ion
    dipole term) on the q2r grid."""
    if data["dielectric"] is None:
        raise ValueError("the .fc file carries no dielectric block")
    nat, nq = data["nat"], data["nq"]
    alat = data["celldm"][0]
    bg = np.linalg.inv(data["lavec"] / alat).T
    ncell = nq[0] * nq[1] * nq[2]
    cells = [(m1, m2, m3) for m3 in range(nq[2]) for m2 in range(nq[1]) for m1 in range(nq[0])]
    kpts = [(k1, k2, k3) for k3 in range(nq[2]) for k2 in range(nq[1]) for k1 in range(nq[0])]
    rigid = RigidIon(data)
    d_dip = np.array([rigid.dynmat(sum((k[p] / nq[p]) * bg[:, p] for p in range(3))).reshape(nat, 3, nat, 3) for k in kpts])
    phase = np.array([[np.exp(2j * np.pi * sum(k[p] * m[p] / nq[p] for p in range(3))) for m in cells] for k in kpts])
    delta = np.einsum("km,kaibj->maibj", phase, d_dip) / ncell
    if np.abs(delta.imag).max() > 1e-8:
        raise RuntimeError("dipole real-space term not real")
    delta = delta.real
    fc = data["fc"].copy()
    for icell in range(ncell):
        for iat in range(nat):
            for jat in range(nat):
                fc[3 * jat:3 * jat + 3, 3 * iat:3 * iat + 3, icell] += delta[icell, iat, :, jat, :].T
    out = dict(data)
    out["fc"] = fc
    out["dipole_added"] = True
    return out


def image_shifts():
    shifts = [(0, 0, 0)]
    for ix, iy, iz in itertools.product((-1, 0, 1), repeat=3):
        if (ix, iy, iz) != (0, 0, 0):
            shifts.append((ix, iy, iz))
    return np.array(shifts, dtype=float)


def build_supercell(data):
    nat, nq = data["nat"], data["nq"]
    ncell = nq[0] * nq[1] * nq[2]
    x_super = np.zeros((nat * ncell, 3))
    map_p2s = np.zeros((ncell, nat), dtype=int)
    icount = icell = 0
    for m3 in range(nq[2]):
        for m2 in range(nq[1]):
            for m1 in range(nq[0]):
                for i in range(nat):
                    x_super[icount] = (data["x_frac"][i] + np.array([m1, m2, m3])) / np.array(nq)
                    map_p2s[icell, i] = icount
                    icount += 1
                icell += 1
    return x_super, map_p2s, data["lavec"] * np.array(nq)[None, :]


def minimum_image_cells(x_super, map_p2s, lavec_s, nat):
    shifts = image_shifts()
    x_cart = np.einsum("ab,ncb->nca", lavec_s, x_super[None, :, :] + shifts[:, None, :])
    out = {}
    for i in range(nat):
        origin = x_cart[0, map_p2s[0, i]]
        d = np.linalg.norm(x_cart - origin[None, None, :], axis=2)
        dmin = d.min(axis=0)
        for j in range(x_super.shape[0]):
            out[(i, j)] = [c for c in range(27) if abs(d[c, j] - dmin[j]) < DIST_TOL]
    return out


# ----------------------------------------------------------------------------- helpers
def signed_sqrt_cm1(eigenvalues_cm2):
    return np.sign(eigenvalues_cm2) * np.sqrt(np.abs(eigenvalues_cm2))


def hermitian(m):
    return 0.5 * (m + m.conj().T)


def diagonalise(dmat_cm2):
    """(omega_cm1 sorted ascending, signed for imaginary modes; eigenvectors as columns)."""
    lam, vec = np.linalg.eigh(hermitian(dmat_cm2))
    return signed_sqrt_cm1(lam), vec


def mass_scale_matrix(masses_amu_per_atom):
    m = np.repeat(np.asarray(masses_amu_per_atom, dtype=float), 3)
    return 1.0 / np.sqrt(np.outer(m, m))


def ry_per_bohr2_to_cm2(dyn, masses_amu_per_atom):
    return dyn * mass_scale_matrix(masses_amu_per_atom) / AMU_RY * RY_TO_CMM1 ** 2


def ws_init(atws_columns):
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
    ck = rvec @ rws.T - half[None, :]
    outside = (ck > WS_EPS).any(axis=1)
    nreq = 1 + (np.abs(ck) < WS_EPS).sum(axis=1)
    return np.where(outside, 0.0, 1.0 / nreq)


def multiplet_groups(omega_cm1, tol=0.5):
    groups, start = [], 0
    n = len(omega_cm1)
    for i in range(1, n + 1):
        if i == n or omega_cm1[i] - omega_cm1[i - 1] > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


# ----------------------------------------------------------------------------- builders
class QESet:
    """matdyn-algorithm dynamical matrix from a q2r .fc file (see module docstring)."""

    def __init__(self, fc_path, asr="distributed", masses_amu=None):
        data = read_qe_fc(fc_path)
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
        self.at = data["lavec"] / self.alat
        self.bg = np.linalg.inv(self.at).T
        self.tau = data["x_alat"]
        self.masses_amu = (np.array(data["masses_ry"])[data["kd"] - 1] / AMU_RY
                           if masses_amu is None else np.asarray(masses_amu, dtype=float))
        ncell = nq[0] * nq[1] * nq[2]
        self.fcb = fc.reshape(nat, 3, nat, 3, ncell).transpose(2, 0, 4, 3, 1)
        rws, half = ws_init(self.at * np.array(nq)[None, :])
        axes = [np.arange(-2 * nq[p], 2 * nq[p] + 1) for p in range(3)]
        grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        rvec = grid @ self.at.T
        m = np.mod(grid, np.array(nq)[None, :])
        icell = m[:, 0] + nq[0] * (m[:, 1] + nq[1] * m[:, 2])
        self.entries = {}
        for na in range(nat):
            for nb in range(nat):
                w = ws_weights(rvec + self.tau[na] - self.tau[nb], rws, half)
                if abs(w.sum() - ncell) > 1e-8:
                    raise RuntimeError(f"wrong total Wigner-Seitz weight for pair ({na},{nb})")
                keep = w > 0.0
                self.entries[(na, nb)] = (rvec[keep], w[keep], icell[keep])
        self._rigid = RigidIon(self.data) if self.data["dielectric"] is not None else None

    def q_cart(self, q_frac):
        return self.bg @ np.asarray(q_frac, dtype=float)

    def short_range(self, q_frac):
        q = self.q_cart(q_frac)
        dyn = np.zeros((self.nat, 3, self.nat, 3), dtype=complex)
        for (na, nb), (rvec, w, icell) in self.entries.items():
            phase = w * np.exp(-2j * np.pi * (rvec @ q))
            dyn[na, :, nb, :] = np.einsum("e,eij->ij", phase, self.fcb[na, nb, icell])
        return dyn.reshape(3 * self.nat, 3 * self.nat)

    def nonanalytic(self, q_frac):
        if self._rigid is None:
            return np.zeros((3 * self.nat, 3 * self.nat), dtype=complex)
        return self._rigid.dynmat(self.q_cart(q_frac))

    def dynmat(self, q_frac):
        """Mass-scaled dynamical matrix in cm^-2 at fractional q."""
        return ry_per_bohr2_to_cm2(self.short_range(q_frac) + self.nonanalytic(q_frac), self.masses_amu)


def _floats(text):
    return [float(x) for x in _FLOAT.findall(text)]


class AlamodeSet:
    """anphon-algorithm dynamical matrix from an ALAMODE FC2 XML descending from ``fc_source``
    (a q2r .fc file: primitive cell, grid, dielectric data). ``asr_onsite`` projects the on-site
    sum rule out of SCPH-renormalised sets (whose correction anphon leaves unprojected)."""

    def __init__(self, xml_path, fc_source, masses_amu=None, perm=None, subtract_dipole=True):
        root = ET.parse(Path(xml_path)).getroot()
        st = root.find("Structure")
        nat_s = int(st.find("NumberOfAtoms").text)
        lv = st.find("LatticeVector")
        self.lavec_s = np.array([_floats(lv.find(f"a{i}").text) for i in (1, 2, 3)]).T
        self.x_s = np.zeros((nat_s, 3))
        for pos in st.find("Position"):
            self.x_s[int(pos.get("index")) - 1] = _floats(pos.text)
        sym = root.find("Symmetry")
        ntran = int(sym.find("NumberOfTranslations").text)
        nat = nat_s // ntran
        self.map_p2s = np.zeros((ntran, nat), dtype=int)
        for mp in sym.find("Translations"):
            self.map_p2s[int(mp.get("tran")) - 1, int(mp.get("atom")) - 1] = int(mp.text) - 1
        self.map_s2p = np.zeros(nat_s, dtype=int)
        for t in range(ntran):
            for j in range(nat):
                self.map_s2p[self.map_p2s[t, j]] = j
        self.nat = nat
        self.qe = QESet(fc_source, asr="none", masses_amu=masses_amu)
        self.nq = self.qe.nq
        if not np.allclose(self.lavec_s, self.qe.data["lavec"] * np.array(self.nq)[None, :], atol=1e-6):
            raise ValueError("XML supercell does not match the .fc primitive cell times the q2r grid")
        self.to_qe = np.arange(nat) if perm is None else np.array([list(perm).index(a) for a in range(nat)])
        self.masses_amu = self.qe.masses_amu if masses_amu is None else np.asarray(masses_amu, dtype=float)
        shifts = image_shifts()
        a1, x1, a2s, x2, cell, val = [], [], [], [], [], []
        for fc2 in root.find("ForceConstants").find("HARMONIC"):
            p1, p2 = fc2.get("pair1").split(), fc2.get("pair2").split()
            a1.append(int(p1[0]) - 1); x1.append(int(p1[1]) - 1)
            a2s.append(int(p2[0]) - 1); x2.append(int(p2[1]) - 1); cell.append(int(p2[2]) - 1)
            val.append(float(fc2.text))
        a1, x1, a2s, x2, cell = (np.array(v) for v in (a1, x1, a2s, x2, cell))
        self.val = np.array(val)
        a2p = self.map_s2p[a2s]
        self.rfrac_p = (self.x_s[a2s] + shifts[cell] - self.x_s[self.map_p2s[0, a2p]]) * np.array(self.nq)[None, :]
        self.row = 3 * self.to_qe[a1] + x1
        self.col = 3 * self.to_qe[a2p] + x2
        self.n_entries = len(self.val)
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
        return self._fourier(self.row, self.col, self.rfrac_p, self.val, q_frac, 3 * self.nat)

    def dipole_sum(self, q_frac):
        if not self.subtract_dipole:
            return np.zeros((3 * self.nat, 3 * self.nat), dtype=complex)
        return self._fourier(self.dip_row, self.dip_col, self.dip_rfrac, self.dip_val, q_frac, 3 * self.nat)

    def onsite_asr_correction(self):
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


# ----------------------------------------------------------------------------- constructions
def strain_derivative_matrix(dyn_plus, dyn_minus, h):
    """K = [D(+h) − D(−h)] / (2h), Hermitian; h is the ENGINEERING shear of the strained pair."""
    return hermitian((np.asarray(dyn_plus) - np.asarray(dyn_minus)) / (2.0 * float(h)))


def project_coupling(K, eigvecs, freqs, tol=0.5, near=5.0):
    """Project K on the eigenbasis (columns of ``eigvecs``, frequencies ``freqs`` in cm^-1).

    Returns a dict of per-mode arrays: ``lam`` (multiplet-block eigenvalues, ascending inside each
    multiplet; the diagonal element for singlets), ``diag`` (e† K e), ``trk2`` (Tr K_sub² of the mode's
    multiplet, gauge invariant), ``group`` / ``group_size`` and ``offdiag`` (largest |K_νμ| to a mode within
    ``near`` cm^-1 outside the multiplet)."""
    Kb = hermitian(eigvecs.conj().T @ K @ eigvecs)
    n = len(freqs)
    lam = np.zeros(n); trk2 = np.zeros(n); gid = np.zeros(n, dtype=int); gsize = np.zeros(n, dtype=int)
    for k, g in enumerate(multiplet_groups(np.asarray(freqs), tol)):
        ev = np.linalg.eigvalsh(Kb[np.ix_(g, g)])
        lam[g] = ev
        trk2[g] = float((ev ** 2).sum())
        gid[g] = k
        gsize[g] = len(g)
    offdiag = np.zeros(n)
    for i in range(n):
        m = (np.abs(np.asarray(freqs) - freqs[i]) < near) & (gid != gid[i])
        offdiag[i] = np.abs(Kb[i, m]).max() if m.any() else 0.0
    return {"lam": lam, "diag": np.real(np.diag(Kb)), "trk2": trk2, "group": gid, "group_size": gsize, "offdiag": offdiag}


def transfer_bare_coupling(lam_bare, eigvecs_bare, eigvecs_ren, freqs_ren, tol=0.5):
    """Retired construction A: each renormalised mode ν receives the bare coupling of the bare mode of
    largest overlap. Returns (lam_A, partner index, single overlap, multiplet-summed overlap)."""
    ov = np.abs(eigvecs_bare.conj().T @ eigvecs_ren) ** 2
    mu = ov.argmax(axis=0)
    single = ov[mu, np.arange(ov.shape[1])]
    summed = np.zeros(ov.shape[1])
    for g in multiplet_groups(np.asarray(freqs_ren), tol):
        for nu in g:
            summed[nu] = ov[mu[nu], g].sum()
    return np.asarray(lam_bare)[mu], mu, single, summed


def gruneisen_from_coupling(lam, omega_r):
    """γ = −Λ/(2 ω_r²) (tensor convention, engineering-shear derivative)."""
    return -np.asarray(lam) / (2.0 * np.asarray(omega_r) ** 2)


def acoustic_character(eigvecs, masses_amu, q_direction):
    """LA/TA labels from the centre-of-mass displacement u_cm ∝ Σ_a sqrt(m_a) e_a: returns
    (labels, |û_cm·q̂|², translation fraction)."""
    nat = len(masses_amu)
    qhat = np.asarray(q_direction, dtype=float)
    qhat = qhat / np.linalg.norm(qhat)
    labels, proj, frac = [], [], []
    for nu in range(eigvecs.shape[1]):
        e = eigvecs[:, nu].reshape(nat, 3)
        u = (np.sqrt(masses_amu)[:, None] * e).sum(axis=0)
        norm = np.vdot(u, u).real
        p = abs(np.vdot(qhat, u)) ** 2 / norm if norm > 1e-12 else 0.0
        proj.append(p)
        frac.append(norm / np.sum(masses_amu))
        labels.append("LA" if p > 0.5 else "TA")
    return labels, np.array(proj), np.array(frac)


def acoustic_limit_check(dynmat_reference, dynmat_plus, dynmat_minus, h, masses_amu, direction,
                         magnitudes=(0.05, 0.1), tol=0.5):
    """Λ/q² and γ of the three lowest (acoustic) modes along ``direction`` (fractional, cubic cell) at the
    reduced |q| in ``magnitudes``. The callables take a fractional q and return D in cm^-2. Returns a list
    of dicts (q_mag, mode, character, omega_r, lam, lam_over_q2, gamma)."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    out = []
    for s in magnitudes:
        q = s * d
        w, E = diagonalise(dynmat_reference(q))
        K = strain_derivative_matrix(dynmat_plus(q), dynmat_minus(q), h)
        pr = project_coupling(K, E, w, tol)
        labels, _, _ = acoustic_character(E, masses_amu, d)
        for nu in range(3):
            out.append({"q_mag": s, "mode": nu + 1, "character": labels[nu], "omega_r": float(w[nu]),
                        "lam": float(pr["lam"][nu]), "lam_over_q2": float(pr["lam"][nu] / s ** 2),
                        "gamma": float(gruneisen_from_coupling(pr["lam"][nu], w[nu]))})
    return out
