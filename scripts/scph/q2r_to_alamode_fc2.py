#!/usr/bin/env python3
"""Convert a Quantum ESPRESSO q2r force-constant file to an ALAMODE harmonic
force-constant XML (FC2XML) and, optionally, an ALAMODE BORN file.

The conversion follows the algorithm of ALAMODE's own tools/qe2alm.cpp
(T. Tadano, MIT licence), reimplemented here so that the strained cells
(ibrav = 0, sheared lattice) and the Born-charge block are handled in one
place and the result can be cross-checked against the compiled tool:

  * the primitive cell, atomic positions (alat Cartesian -> crystal) and the
    nq1 x nq2 x nq3 real-space force constants are read from the .fc file;
  * the supercell is nq1 a1, nq2 a2, nq3 a3 with atoms ordered cell-major
    (m1 fastest inside m2 inside m3), primitive atom minor, exactly as
    qe2alm does, so that the Translations map is tran = cell, atom = primitive index;
  * every (primitive atom, supercell atom) pair is assigned to the supercell
    image(s) of minimum distance among the 27 neighbouring supercell shifts
    (cell index 1 = no shift, then ix, iy, iz in -1..1 skipping the origin,
    the enumeration anphon uses for xshift_s); a force constant shared by
    n equidistant images is written n times with value/n;
  * the QE block value C(i, j, na, nb; R) is stored transposed (pair1 = nb,
    j ; pair2 = na in cell R, i), as qe2alm does;
  * the acoustic sum rule (--asr): "none" (raw q2r values), "onsite"
    (qe2alm-style correction of the on-site term only; shifts soft-mode
    frequencies by up to 8 cm^-1 and is not recommended) or "distributed"
    (DEFAULT; minimal-norm correction: the residual of each row
    sum_{j,R} Phi(i alpha, j beta; R) is spread evenly over all n_at x n_cell
    entries of that row, then the pair symmetry Phi(i a, j b; R) =
    Phi(j b, i a; -R) is restored, iterated to 1e-12 Ry/bohr^2). The q2r
    force constants of this project violate the sum rule at the 4e-3
    Ry/bohr^2 level; matdyn imposes its 'crystal' rule at run time, anphon
    imposes nothing, and without an exact rule the Gamma-point acoustic
    frequencies of the SCPH-renormalised set come out at +-0.01 cm^-1 rather
    than 0, which anphon's RTA then treats as real phonons with enormous
    Bose factors (three-phonon linewidths 100x too large). The distributed
    rule changes every entry by ~1e-5 Ry/bohr^2 (frequencies at the DFPT
    grid points by < 0.05 cm^-1) and makes the acoustic frequencies at
    Gamma vanish exactly.

Units are those of the .fc file (Ry, bohr), which are also ALAMODE's.

Long-range (dipole) part. When the .fc file carries a dielectric block,
q2r has SUBTRACTED the rigid-ion dipole-dipole term (QE routine rgd_blk,
sign -1) from every dynamical matrix before the inverse Fourier transform,
so the file holds short-range force constants only, and matdyn re-adds the
dipole term at each q. anphon's Ewald treatment (NONANALYTIC = 3) instead
expects FULL force constants: it subtracts its own real-space dipole part
and adds the reciprocal-space Ewald term at each q. With --add-dipole the
converter therefore restores the full real-space force constants on the
nq1 x nq2 x nq3 grid, Phi_full(R) = Phi_SR(R) + (1/N) sum_k D_dip(q_k)
exp(+i q_k.R), with D_dip(q_k) the QE rigid-ion term (a literal port of
rgd_blk, sign +1, Ewald parameter alpha = 1, gmax = 14, G-space only) at
the N commensurate grid points; at those points the result is identical to
the QE dynamical matrices. The dielectric tensor and Born charges are then
written to a separate ALAMODE BORN file (--born) for NONANALYTIC = 3.

Usage:
  q2r_to_alamode_fc2.py FILE.fc OUT.xml [--asr none|onsite|distributed] [--born BORN] [--add-dipole]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

EPS_FC = 1e-15
DIST_TOL = 1e-3


def _tokens(path: Path):
    for line in path.read_text().splitlines():
        for tok in line.split():
            yield tok


def read_qe_fc(path: Path):
    """Parse a q2r .fc file. Returns a dict with the primitive cell (columns =
    lattice vectors, bohr), species, crystal positions, optional dielectric
    data, the mesh and the force constants fc[3*nat, 3*nat, ncell] in the
    transposed storage of qe2alm (fc[3*jat+jcrd, 3*iat+icrd, cell] = C(icrd, jcrd, iat, jat; cell))."""
    it = _tokens(path)
    nkd, nat, ibrav = int(next(it)), int(next(it)), int(next(it))
    celldm = [float(next(it)) for _ in range(6)]
    if ibrav == 0:
        lavec = np.zeros((3, 3))
        for i in range(3):                       # line i = lattice vector i (alat units)
            for c in range(3):
                lavec[c, i] = float(next(it))
        lavec *= celldm[0]
    elif ibrav == 1:
        lavec = celldm[0] * np.eye(3)
    else:
        raise NotImplementedError(f"ibrav = {ibrav} not supported (only 0 and 1)")
    symbols, masses = [], []
    for _ in range(nkd):
        next(it)                                 # index
        sym = next(it)[1:]                       # strip the opening quote
        next(it)                                 # closing quote token
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
    return {"nkd": nkd, "nat": nat, "ibrav": ibrav, "celldm": celldm, "lavec": lavec,
            "symbols": symbols, "masses_ry": masses, "kd": kd, "x_frac": x_frac, "x_alat": x_alat,
            "dielectric": dielectric, "born": born, "nq": nq, "fc": fc}


def image_shifts():
    """The 27 supercell shifts in anphon's enumeration (index 0 = origin)."""
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
    lavec_s = data["lavec"] * np.array(nq)[None, :]      # columns scaled
    return x_super, map_p2s, lavec_s


def minimum_image_cells(x_super, map_p2s, lavec_s, nat):
    """For each primitive atom i and supercell atom j: list of image-cell
    indices (0..26) realising the minimum distance within DIST_TOL bohr."""
    shifts = image_shifts()
    x_cart_images = np.einsum("ab,ncb->nca", lavec_s, x_super[None, :, :] + shifts[:, None, :])  # (27, nat_s, 3)
    out = {}
    for i in range(nat):
        iat = map_p2s[0, i]
        origin = x_cart_images[0, iat]
        d = np.linalg.norm(x_cart_images - origin[None, None, :], axis=2)   # (27, nat_s)
        dmin = d.min(axis=0)
        for j in range(x_super.shape[0]):
            out[(i, j)] = [c for c in range(27) if abs(d[c, j] - dmin[j]) < DIST_TOL]
    return out


def rigid_ion_dynmat(data, q_2pi_alat):
    """Port of QE PHonon/PH/rigid.f90::rgd_blk (sign = +1, 3D case).

    Returns the dipole-dipole contribution to the dynamical matrix in
    Ry/bohr^2 as dyn[na, i, nb, j] (QE index order dyn(i, j, na, nb)) for q
    given in 2 pi/alat Cartesian units. tau in alat units, bg in 2 pi/alat,
    omega in bohr^3, e2 = 2 (Rydberg units)."""
    nat = data["nat"]
    alat = data["celldm"][0]
    at = data["lavec"] / alat                       # columns = lattice vectors, alat units
    bg = np.linalg.inv(at).T                        # columns = reciprocal vectors, 2 pi/alat
    omega = abs(np.linalg.det(data["lavec"]))
    tau = data["x_alat"]
    eps = data["dielectric"]
    zeu = data["born"]                              # zeu[na][i, j]
    nr = data["nq"]
    gmax, alph = 14.0, 1.0
    geg0 = gmax * alph * 4.0
    nrx = [0 if nr[p] == 1 else int(np.sqrt(geg0) / np.linalg.norm(bg[:, p])) + 1 for p in range(3)]
    fac = 2.0 * 4.0 * np.pi / omega
    dyn = np.zeros((nat, 3, nat, 3), dtype=complex)
    q = np.asarray(q_2pi_alat, dtype=float)
    for m1 in range(-nrx[0], nrx[0] + 1):
        for m2 in range(-nrx[1], nrx[1] + 1):
            for m3 in range(-nrx[2], nrx[2] + 1):
                g = m1 * bg[:, 0] + m2 * bg[:, 1] + m3 * bg[:, 2]
                geg = g @ eps @ g
                if geg > 0.0 and geg / alph / 4.0 < gmax:
                    facgd = fac * np.exp(-geg / alph / 4.0) / geg
                    zg = np.array([g @ zeu[na] for na in range(nat)])          # zag[na, :]
                    for na in range(nat):
                        fnat = np.zeros(3)
                        for nb in range(nat):
                            arg = 2.0 * np.pi * (g @ (tau[na] - tau[nb]))
                            fnat += zg[nb] * np.cos(arg)
                        dyn[na, :, na, :] -= facgd * np.outer(zg[na], fnat)
                gq = g + q
                geg = gq @ eps @ gq
                if geg > 0.0 and geg / alph / 4.0 < gmax:
                    facgd = fac * np.exp(-geg / alph / 4.0) / geg
                    zg = np.array([gq @ zeu[na] for na in range(nat)])
                    for nb in range(nat):
                        for na in range(nat):
                            arg = 2.0 * np.pi * (gq @ (tau[na] - tau[nb]))
                            dyn[na, :, nb, :] += facgd * np.exp(1j * arg) * np.outer(zg[na], zg[nb])
    return dyn


def add_dipole_real_space(data):
    """Return a copy of data whose force constants are the FULL real-space
    force constants Phi_SR + Phi_dip on the q2r grid (see module docstring)."""
    if data["dielectric"] is None:
        raise ValueError("the .fc file carries no dielectric block; nothing to add")
    nat, nq = data["nat"], data["nq"]
    alat = data["celldm"][0]
    at = data["lavec"] / alat
    bg = np.linalg.inv(at).T
    ncell = nq[0] * nq[1] * nq[2]
    cells = [(m1, m2, m3) for m3 in range(nq[2]) for m2 in range(nq[1]) for m1 in range(nq[0])]
    kpts = [(k1, k2, k3) for k3 in range(nq[2]) for k2 in range(nq[1]) for k1 in range(nq[0])]
    d_dip = []
    for k in kpts:
        q = (k[0] / nq[0]) * bg[:, 0] + (k[1] / nq[1]) * bg[:, 1] + (k[2] / nq[2]) * bg[:, 2]
        d_dip.append(rigid_ion_dynmat(data, q))
    d_dip = np.array(d_dip)                                   # (ncell, nat, 3, nat, 3)
    phase = np.array([[np.exp(2j * np.pi * sum(k[p] * m[p] / nq[p] for p in range(3)))
                       for m in cells] for k in kpts])         # (k, m)
    delta = np.einsum("km,kaibj->maibj", phase, d_dip) / ncell   # (ncell, na, i, nb, j)
    if np.abs(delta.imag).max() > 1e-8:
        raise RuntimeError(f"dipole real-space term not real: max imag {np.abs(delta.imag).max():.2e}")
    delta = delta.real
    out = dict(data)
    fc = data["fc"].copy()
    for icell in range(ncell):
        for iat in range(nat):
            for jat in range(nat):
                for icrd in range(3):
                    for jcrd in range(3):
                        # transposed storage: fc[3*jat+jcrd, 3*iat+icrd, cell] = C(icrd, jcrd, iat, jat; cell)
                        fc[3 * jat + jcrd, 3 * iat + icrd, icell] += delta[icell, iat, icrd, jat, jcrd]
    out["fc"] = fc
    out["dipole_added"] = True
    return out


def dynamical_matrix_grid(data, k):
    """Full dynamical matrix (Ry/bohr^2, dyn[na,i,nb,j]) at the commensurate grid
    point k = (k1, k2, k3) by the plain Fourier sum over the stored cells."""
    nat, nq = data["nat"], data["nq"]
    fc = data["fc"]
    dyn = np.zeros((nat, 3, nat, 3), dtype=complex)
    icell = 0
    for m3 in range(nq[2]):
        for m2 in range(nq[1]):
            for m1 in range(nq[0]):
                ph = np.exp(-2j * np.pi * (k[0] * m1 / nq[0] + k[1] * m2 / nq[1] + k[2] * m3 / nq[2]))
                for iat in range(nat):
                    for jat in range(nat):
                        for icrd in range(3):
                            for jcrd in range(3):
                                dyn[iat, icrd, jat, jcrd] += fc[3 * jat + jcrd, 3 * iat + icrd, icell] * ph
                icell += 1
    return dyn


def impose_asr(fc, nat, ncell):
    """qe2alm-style on-site correction (kept for comparison; not recommended)."""
    fc = fc.copy()
    for icrd in range(3):
        for jcrd in range(3):
            for iat in range(nat):
                s = 0.0
                for jat in range(nat):
                    s += fc[3 * iat + icrd, 3 * jat + jcrd, :].sum()
                fc[3 * iat + icrd, 3 * iat + jcrd, 0] -= s
    return fc


def impose_asr_distributed(fc, nat, nq, tol=1e-12, max_iter=200):
    """Minimal-norm acoustic sum rule with pair symmetry.

    Storage: fc[3*jat+jcrd, 3*iat+icrd, cell] = C(icrd, jcrd, iat, jat; cell),
    C being the force constant between atom iat (origin cell) and atom jat in
    cell R (m1 fastest). Row sums over (jat, cell) are driven to zero by
    subtracting the residual / (nat * ncell) from every entry of the row; the
    symmetry C(i, j, na, nb; R) = C(j, i, nb, na; -R) is then re-imposed;
    the two steps are alternated until both residuals fall below tol.
    Returns (fc_new, residual_before, residual_after, n_iter).
    """
    ncell = nq[0] * nq[1] * nq[2]
    c = np.zeros((3, 3, nat, nat, ncell))
    for iat in range(nat):
        for jat in range(nat):
            for icrd in range(3):
                for jcrd in range(3):
                    c[icrd, jcrd, iat, jat, :] = fc[3 * jat + jcrd, 3 * iat + icrd, :]
    # index of the cell -R for every cell R (m1 fastest)
    cells = [(m1, m2, m3) for m3 in range(nq[2]) for m2 in range(nq[1]) for m1 in range(nq[0])]
    lookup = {m: k for k, m in enumerate(cells)}
    neg = np.array([lookup[((-m1) % nq[0], (-m2) % nq[1], (-m3) % nq[2])] for (m1, m2, m3) in cells])

    def residual(arr):
        return np.abs(arr.sum(axis=(3, 4))).max()

    before = residual(c)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        res = c.sum(axis=(3, 4), keepdims=True) / (nat * ncell)
        c = c - res
        ct = np.transpose(c, (1, 0, 3, 2, 4))[..., neg]
        c = 0.5 * (c + ct)
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


def fmt(x: float) -> str:
    return f"{x: .15e}"


def write_alamode_xml(data, out: Path, asr: str = "distributed", source: str = "") -> int:
    nat, nq = data["nat"], data["nq"]
    ncell = nq[0] * nq[1] * nq[2]
    if asr == "onsite":
        fc = impose_asr(data["fc"], nat, ncell)
    elif asr == "distributed":
        fc, before, after, n_iter = impose_asr_distributed(data["fc"], nat, nq)
        print(f"  distributed ASR: max row residual {before:.3e} -> {after:.3e} Ry/bohr^2 in {n_iter} iterations; "
              f"max |change| {np.abs(fc - data['fc']).max():.3e}")
    elif asr == "none":
        fc = data["fc"]
    else:
        raise ValueError(asr)
    x_super, map_p2s, lavec_s = build_supercell(data)
    mind = minimum_image_cells(x_super, map_p2s, lavec_s, nat)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<Data>",
             f"  <FilenameOfQE>{source}</FilenameOfQE>",
             "  <Structure>",
             f"    <NumberOfAtoms>{nat * ncell}</NumberOfAtoms>",
             f"    <NumberOfElements>{data['nkd']}</NumberOfElements>",
             "    <AtomicElements>"]
    for k, sym in enumerate(data["symbols"]):
        lines.append(f'      <element number="{k + 1}">{sym}</element>')
    lines += ["    </AtomicElements>", "    <LatticeVector>"]
    for i in range(3):
        lines.append(f"      <a{i + 1}>{''.join(fmt(lavec_s[c, i]) for c in range(3))}</a{i + 1}>")
    lines += ["    </LatticeVector>", "    <Periodicity>1 1 1</Periodicity>", "    <Position>"]
    icount = 0
    for icell in range(ncell):
        for j in range(nat):
            idx = map_p2s[icell, j]
            lines.append(f'      <pos index="{icount + 1}" element="{data["symbols"][data["kd"][j] - 1]}">'
                         f"{''.join(fmt(v) for v in x_super[idx])}</pos>")
            icount += 1
    lines += ["    </Position>", "  </Structure>", "  <Symmetry>",
              f"    <NumberOfTranslations>{ncell}</NumberOfTranslations>", "    <Translations>"]
    for icell in range(ncell):
        for j in range(nat):
            lines.append(f'      <map tran="{icell + 1}" atom="{j + 1}">{map_p2s[icell, j] + 1}</map>')
    lines += ["    </Translations>", "  </Symmetry>", "  <ForceConstants>", "    <HARMONIC>"]
    n_written = 0
    for iat in range(nat):
        for icrd in range(3):
            for icell in range(ncell):
                for jat in range(nat):
                    kat = map_p2s[icell, jat]
                    cells = mind[(iat, kat)]
                    nmulti = len(cells)
                    for jcrd in range(3):
                        val = fc[3 * iat + icrd, 3 * jat + jcrd, icell]
                        if abs(val) < EPS_FC:
                            continue
                        for c in cells:
                            lines.append(f'      <FC2 pair1="{iat + 1} {icrd + 1}" '
                                         f'pair2="{kat + 1} {jcrd + 1} {c + 1}">{val / nmulti:.15e}</FC2>')
                            n_written += 1
    lines += ["    </HARMONIC>", "  </ForceConstants>", "</Data>", ""]
    out.write_text("\n".join(lines))
    return n_written


def write_born(data, out: Path) -> None:
    """ALAMODE BORN file: dielectric tensor (3 lines) then Z* (3 lines per atom)."""
    if data["dielectric"] is None:
        raise ValueError("the .fc file carries no dielectric block")
    rows = [" ".join(f"{v:16.8f}" for v in r) for r in data["dielectric"]]
    for z in data["born"]:
        rows += [" ".join(f"{v:16.8f}" for v in r) for r in z]
    out.write_text("\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fc", type=Path)
    ap.add_argument("xml", type=Path)
    ap.add_argument("--asr", choices=("none", "onsite", "distributed"), default="distributed",
                    help="acoustic sum rule treatment (default: distributed, minimal-norm)")
    ap.add_argument("--born", type=Path, default=None, help="also write an ALAMODE BORN file")
    ap.add_argument("--add-dipole", action="store_true",
                    help="restore the full force constants (short-range + rigid-ion dipole term) for NONANALYTIC = 3")
    args = ap.parse_args()
    data = read_qe_fc(args.fc)
    if args.add_dipole:
        data = add_dipole_real_space(data)
    n = write_alamode_xml(data, args.xml, asr=args.asr, source=args.fc.name)
    lv = data["lavec"]
    print(f"{args.fc.name}: nat = {data['nat']}, species {data['symbols']}, mesh {data['nq']}, "
          f"ibrav {data['ibrav']}, a1 = {np.round(lv[:, 0], 6)} bohr; wrote {n} FC2 entries -> {args.xml}")
    if args.born is not None:
        write_born(data, args.born)
        print(f"  BORN -> {args.born} (eps_inf diag {np.round(np.diag(data['dielectric']), 4)})")


if __name__ == "__main__":
    main()
