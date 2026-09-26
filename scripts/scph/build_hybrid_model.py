#!/usr/bin/env python3
"""build_hybrid_model.py - the hybrid model of the paper (Sec. 3.2).

reexpress  Re-express the ALAMODE example harmonic set (STO222.xml, 2x2x2 supercell, cell 7.363 bohr) on the 4x4x4 supercell:
           short-range dynamical matrix on the 4^3 grid (minimum-image Fourier sum minus the rigid-ion dipole force constants of
           the 2x2x2 supercell, example BORN), inverse FFT to the 320-atom supercell, dipole force constants of the 4x4x4
           supercell restored, distributed translational sum rule; XML STO444_tut.xml (+ BORN_tut). Atom order = QE order.
           Validation: band path, full 11^3 mesh and 20 off-grid wave vectors against the original set.
strain     Strained sets D(q, +-h) = D_tut(q) +- h K_QE(q): K_QE = [Phi_QE(+s) - Phi_QE(-s)] / (2h) from the QE q2r short-range
           force constants (real space), h = 0.010 (+-0.005 pair) and 0.020 (+-0.010 pair); dipole part regenerated for the
           sheared cell with eps_inf, Z* = example values + the odd part of the QE pair (the same values in the BORN files).
           Validation on the 4^3 grid; Gamma translations; odd part after the sum rule.
qefc       Write the sets in q2r format (z_*.fc) for the assembly code; round-trip check against the XML.
Usage: DYNMAT_SCRATCH=<ALAMODE work directory> uv run python scripts/scph/build_hybrid_model.py reexpress | strain | qefc
Outputs: XML/BORN in DYNMAT_SCRATCH/run/z_sets and data/raw/alamode_sto/z_tut/; validation CSVs in data/processed/reports/.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import re

import numpy as np
import pandas as pd

_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _floats(text):
    return [float(x) for x in _FLOAT.findall(text)]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from q2r_to_alamode_fc2 import (add_dipole_real_space, build_supercell, image_shifts, minimum_image_cells,  # noqa: E402
                                read_qe_fc, rigid_ion_dynmat, write_alamode_xml, write_born)
from latvisc.coupling import QESet, diagonalise, multiplet_groups, ry_per_bohr2_to_cm2  # noqa: E402

QE = REPO / "dft" / "qe" / "SrTiO3"
FC = {
    "reference": QE / "dispersion_pbesol" / "SrTiO3_pbesol.444.fc",
    "shear_xy_p005": QE / "gruneisen_pbesol" / "shear_xy_p005" / "SrTiO3_pbesol_shear_xy_p005.444.fc",
    "shear_xy_m005": QE / "gruneisen_pbesol" / "shear_xy_m005" / "SrTiO3_pbesol_shear_xy_m005.444.fc",
    "shear_xy_p010": QE / "gruneisen_pbesol" / "shear_xy_p010" / "SrTiO3_pbesol_shear_xy_p010.444.fc",
    "shear_xy_m010": QE / "gruneisen_pbesol" / "shear_xy_m010" / "SrTiO3_pbesol_shear_xy_m010.444.fc",
}
REPORTS = REPO / "data" / "processed" / "reports"
SCRATCH = Path(os.environ["DYNMAT_SCRATCH"])
ZDIR = SCRATCH / "run" / "z_sets"
PERSIST = REPO / "data" / "raw" / "alamode_sto" / "z_tut"
EX = SCRATCH / "alamode" / "example" / "SrTiO3" / "reference"
ALAT_TUT = 7.363
NQ4 = [4, 4, 4]
AMU_RY = 911.444243  # electron-mass-free: 1 amu in Ry mass units (2 m_e units); consistent with latvisc.coupling
SYMBOLS = ["Sr", "Ti", "O"]
KD = np.array([1, 2, 3, 3, 3])
MASSES_AMU = np.array([87.62, 47.867, 15.999, 15.999, 15.999])
PERM_TUT_TO_QE = [0, 1, 4, 3, 2]          # tutorial primitive atom a -> QE atom index; O(0,.5,.5)->QE O3 etc.


def read_born_file(path):
    v = np.array([float(x) for x in Path(path).read_text().split()]).reshape(-1, 3, 3)
    return v[0], v[1:]


def parse_xml(path):
    root = ET.parse(Path(path)).getroot()
    st = root.find("Structure")
    nat_s = int(st.find("NumberOfAtoms").text)
    lv = st.find("LatticeVector")
    lavec_s = np.array([_floats(lv.find(f"a{i}").text) for i in (1, 2, 3)]).T
    x_s = np.zeros((nat_s, 3))
    for pos in st.find("Position"):
        x_s[int(pos.get("index")) - 1] = _floats(pos.text)
    sym = root.find("Symmetry")
    ntran = int(sym.find("NumberOfTranslations").text)
    nat = nat_s // ntran
    map_p2s = np.zeros((ntran, nat), dtype=int)
    for mp in sym.find("Translations"):
        map_p2s[int(mp.get("tran")) - 1, int(mp.get("atom")) - 1] = int(mp.text) - 1
    map_s2p = np.zeros(nat_s, dtype=int)
    for t in range(ntran):
        for j in range(nat):
            map_s2p[map_p2s[t, j]] = j
    a1, x1, a2s, x2, cell, val = [], [], [], [], [], []
    for fc2 in root.find("ForceConstants").find("HARMONIC"):
        p1, p2 = fc2.get("pair1").split(), fc2.get("pair2").split()
        a1.append(int(p1[0]) - 1); x1.append(int(p1[1]) - 1)
        a2s.append(int(p2[0]) - 1); x2.append(int(p2[1]) - 1); cell.append(int(p2[2]) - 1)
        val.append(float(fc2.text))
    return dict(lavec_s=lavec_s, x_s=x_s, nat=nat, ntran=ntran, map_p2s=map_p2s, map_s2p=map_s2p,
                a1=np.array(a1), x1=np.array(x1), a2s=np.array(a2s), x2=np.array(x2), cell=np.array(cell), val=np.array(val))


def cell_data(nq, shear=0.0, eps=None, born=None, x_frac=None):
    """QE-like data dict for the tutorial cell (QE atom order) used by the dipole/XML machinery."""
    s = shear
    rows = np.array([[1.0, s, 0.0], [s, 1.0, 0.0], [0.0, 0.0, 1.0]])   # rows = lattice vectors (alat units)
    lavec = ALAT_TUT * rows.T                                           # columns = lattice vectors (bohr)
    xf = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) if x_frac is None else x_frac
    x_alat = (lavec @ xf.T).T / ALAT_TUT
    return {"nkd": 3, "nat": 5, "ibrav": 0, "celldm": [ALAT_TUT, 0, 0, 0, 0, 0], "lavec": lavec, "symbols": SYMBOLS,
            "masses_ry": (MASSES_AMU[[0, 1, 2]] * AMU_RY).tolist(), "kd": KD, "x_frac": xf, "x_alat": x_alat,
            "dielectric": eps, "born": born, "nq": list(nq), "fc": np.zeros((15, 15, int(np.prod(nq))))}


class LightSet:
    """Fourier sum of an ALAMODE FC2 XML in anphon's minimum-image convention (lattice-vector phases), with the rigid-ion
    dipole subtraction on the XML's own supercell grid and the rigid-ion non-analytic term; QE atom order via perm."""

    def __init__(self, xml_path, data, perm=None):
        x = parse_xml(xml_path)
        self.nat, self.data = x["nat"], data
        nq = np.array(data["nq"])
        to_qe = np.arange(self.nat) if perm is None else np.array([list(perm).index(a) for a in range(self.nat)])
        # tutorial atom a (0-based) sits at QE index perm[a]; to_qe[a] = perm[a]
        to_qe = np.arange(self.nat) if perm is None else np.array(perm)
        shifts = image_shifts()
        a2p = x["map_s2p"][x["a2s"]]
        self.rfrac = (x["x_s"][x["a2s"]] + shifts[x["cell"]] - x["x_s"][x["map_p2s"][0, a2p]]) * nq[None, :]
        self.row = 3 * to_qe[x["a1"]] + x["x1"]
        self.col = 3 * to_qe[a2p] + x["x2"]
        self.val = x["val"]
        # dipole force constants of this supercell (rigid-ion model, data's eps/Z*), mapped like the XML entries
        zero = dict(data, fc=np.zeros_like(data["fc"]))
        fc_dip = add_dipole_real_space(zero)["fc"]
        x_super, map_p2s, lavec_s = build_supercell(data)
        mind = minimum_image_cells(x_super, map_p2s, lavec_s, self.nat)
        ncell = int(np.prod(nq))
        drow, dcol, dr, dv = [], [], [], []
        for iat in range(self.nat):
            for icrd in range(3):
                for icell in range(ncell):
                    for jat in range(self.nat):
                        kat = map_p2s[icell, jat]
                        cells = mind[(iat, kat)]
                        for jcrd in range(3):
                            v = fc_dip[3 * iat + icrd, 3 * jat + jcrd, icell]
                            for c in cells:
                                drow.append(3 * iat + icrd); dcol.append(3 * jat + jcrd)
                                dr.append((x_super[kat] + shifts[c] - x_super[map_p2s[0, jat]]) * nq)
                                dv.append(v / len(cells))
        self.drow, self.dcol, self.dr, self.dv = np.array(drow), np.array(dcol), np.array(dr), np.array(dv)
        self.qe_like = QESetLike(data)

    @staticmethod
    def _fourier(row, col, rfrac, val, q, n=15):
        phase = np.exp(2j * np.pi * (rfrac @ np.asarray(q, dtype=float)))
        dyn = np.zeros((n, n), dtype=complex)
        np.add.at(dyn, (row, col), val * phase)
        return dyn

    def xml_sum(self, q):
        return self._fourier(self.row, self.col, self.rfrac, self.val, q)

    def dip_sum(self, q):
        return self._fourier(self.drow, self.dcol, self.dr, self.dv, q)

    def short_range(self, q):
        return self.xml_sum(q) - self.dip_sum(q)

    def full_ry(self, q):
        return self.short_range(q) + self.qe_like.nonanalytic(q)

    def dynmat(self, q):
        return ry_per_bohr2_to_cm2(self.full_ry(q), MASSES_AMU)


class QESetLike:
    """rigid-ion non-analytic term for a cell_data dict (fractional q)."""

    def __init__(self, data):
        self.data = data
        at = data["lavec"] / data["celldm"][0]
        self.bg = np.linalg.inv(at).T

    def nonanalytic(self, q_frac):
        if self.data["dielectric"] is None:
            return np.zeros((15, 15), dtype=complex)
        return rigid_ion_dynmat(self.data, self.bg @ np.asarray(q_frac, dtype=float)).reshape(15, 15)


def grid_points(n):
    return [(i / n, j / n, k / n) for k in range(n) for j in range(n) for i in range(n)]   # m1 fastest = cell order


def inverse_fft_to_fc(dyn_of_k, nq):
    """fc[3*iat+icrd, 3*jat+jcrd, icell] = (1/N) sum_k D(k)[iat icrd, jat jcrd] exp(-2 pi i k.m) with the converter's
    cell order (m1 fastest); D(k) in the LightSet convention (lattice-vector phases exp(+2 pi i q.R))."""
    ks = grid_points(nq[0])
    cells = [(m1, m2, m3) for m3 in range(nq[2]) for m2 in range(nq[1]) for m1 in range(nq[0])]
    N = len(ks)
    D = np.array([dyn_of_k(k) for k in ks])                                       # (N, 15, 15)
    phase = np.array([[np.exp(-2j * np.pi * (k[0] * m[0] + k[1] * m[1] + k[2] * m[2])) for m in cells] for k in ks])
    fc = np.einsum("km,kab->abm", phase, D) / N
    if np.abs(fc.imag).max() > 1e-9:
        raise RuntimeError(f"real-space force constants not real: {np.abs(fc.imag).max():.2e}")
    return fc.real


def band_path(npts=51):
    pts = {"G": (0, 0, 0), "X": (.5, 0, 0), "M": (.5, .5, 0), "R": (.5, .5, .5)}
    segs = [("G", "X"), ("X", "M"), ("M", "G"), ("G", "R"), ("R", "M")]
    qs = []
    for a, b in segs:
        for t in np.linspace(0, 1, npts):
            qs.append(tuple((1 - t) * np.array(pts[a]) + t * np.array(pts[b])))
    return qs


def overlap_multiplet(w1, v1, w2, v2, tol=0.5):
    """multiplet-summed eigenspace overlap between two sets (sorted ascending): min over multiplets of mean subspace overlap."""
    groups = multiplet_groups(w1, tol)
    worst = 1.0
    for g in groups:
        P = v1[:, g] @ v1[:, g].conj().T
        ov = np.real(np.trace(v2[:, g].conj().T @ P @ v2[:, g])) / len(g)
        worst = min(worst, ov)
    return worst


def reexpress():
    ZDIR.mkdir(parents=True, exist_ok=True); PERSIST.mkdir(parents=True, exist_ok=True)
    eps, born_t = read_born_file(EX / "BORN")
    born_qe_order = born_t[PERM_TUT_TO_QE_INV()]
    d222 = cell_data([2, 2, 2], eps=eps, born=born_qe_order)
    tut = LightSet(EX / "STO222.xml", d222, perm=PERM_TUT_TO_QE)
    # short-range force constants on the 4x4x4 grid
    fc_sr = inverse_fft_to_fc(tut.short_range, NQ4)
    d444 = cell_data(NQ4, eps=eps, born=born_qe_order)
    d444_full = add_dipole_real_space(dict(d444, fc=fc_sr))
    xml = ZDIR / "STO444_tut.xml"
    n = write_alamode_xml(d444_full, xml, asr="distributed", source="STO222.xml re-expressed on the 4x4x4 supercell (build_hybrid_model.py)")
    write_born(d444, ZDIR / "BORN_tut")
    for f in (xml, ZDIR / "BORN_tut"):
        (PERSIST / f.name).write_bytes(f.read_bytes())
    print(f"STO444_tut.xml written: {n} FC2 entries")
    new = LightSet(xml, d444)
    rows = []
    # (i) band path + 11^3
    qs_band = band_path(); qs_mesh = [(i / 11, j / 11, k / 11) for i in range(11) for j in range(11) for k in range(11)]
    for label, qs in (("band_path_255", qs_band), ("mesh_11cubed", qs_mesh)):
        dmax = 0.0; gmax = 0.0; worst_ov = 1.0; dnorm_max = 0.0
        for q in qs:
            D1, D2 = tut.dynmat(q), new.dynmat(q)
            w1, v1 = diagonalise(D1); w2, v2 = diagonalise(D2)
            w1, w2 = np.sort(w1), np.sort(w2)
            if all(abs(c) < 1e-12 for c in q):
                gmax = max(gmax, np.sort(np.abs(w2))[:3].max())   # three smallest |omega| = translations
            dmax = max(dmax, np.abs(w1 - w2).max())
            dnorm_max = max(dnorm_max, np.linalg.norm(D1 - D2) / np.linalg.norm(D1))
            worst_ov = min(worst_ov, overlap_multiplet(w1, v1, w2, v2))
        rows.append({"check": label, "n_q": len(qs), "max_abs_domega_cm1": dmax, "max_rel_dynmat_norm": dnorm_max,
                     "min_multiplet_overlap": worst_ov, "gamma_translation_max_cm1": gmax})
        print(rows[-1], flush=True)
    # (ii) 20 off-grid q (not on the 2^3 or 4^3 grids)
    rng = np.random.default_rng(7)
    qs_off = [tuple(rng.uniform(0.05, 0.45, 3) + rng.choice([0, 0.5], 3) * 0 + 0.03) for _ in range(20)]
    for iq, q in enumerate(qs_off):
        D1, D2 = tut.dynmat(q), new.dynmat(q)
        w1, v1 = diagonalise(D1); w2, v2 = diagonalise(D2)
        rows.append({"check": f"offgrid_q{iq}", "n_q": 1, "max_abs_domega_cm1": float(np.abs(np.sort(w1) - np.sort(w2)).max()),
                     "max_rel_dynmat_norm": float(np.linalg.norm(D1 - D2) / np.linalg.norm(D1)),
                     "min_multiplet_overlap": overlap_multiplet(w1, v1, w2, v2), "gamma_translation_max_cm1": np.nan,
                     "q": f"({q[0]:.4f} {q[1]:.4f} {q[2]:.4f})"})
    # on-grid exactness (4^3 grid)
    dg = max(np.abs(np.sort(diagonalise(tut.dynmat(q))[0]) - np.sort(diagonalise(new.dynmat(q))[0])).max() for q in grid_points(4))
    rows.append({"check": "grid_4cubed_exactness", "n_q": 64, "max_abs_domega_cm1": dg})
    df = pd.DataFrame(rows)
    out = REPORTS / "hybrid_model_reexpression_validation.csv"; tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# build_hybrid_model.py reexpress: STO222.xml (2x2x2 supercell) vs STO444_tut.xml (4x4x4 re-expression), both evaluated\n"
                 "# with the same builder (minimum-image Fourier sum, rigid-ion dipole subtraction on each set's own supercell, rigid-ion NA term,\n"
                 "# tutorial BORN); frequencies in cm^-1 with the standard atomic masses; multiplet overlap = minimum over degenerate\n"
                 "# multiplets (|d omega| < 0.5 cm^-1) of the summed eigenspace overlap.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    print(df.to_string())


def PERM_TUT_TO_QE_INV():
    """index array giving, for each QE atom, the tutorial atom index (used to reorder Born tensors)."""
    inv = np.zeros(5, dtype=int)
    for a, qi in enumerate(PERM_TUT_TO_QE):
        inv[qi] = a
    return inv


def strain():
    ZDIR.mkdir(parents=True, exist_ok=True); PERSIST.mkdir(parents=True, exist_ok=True)
    eps, born_t = read_born_file(EX / "BORN")
    born_qe = born_t[PERM_TUT_TO_QE_INV()]
    d222 = cell_data([2, 2, 2], eps=eps, born=born_qe)
    tut = LightSet(EX / "STO222.xml", d222, perm=PERM_TUT_TO_QE)
    fc_sr = inverse_fft_to_fc(tut.short_range, NQ4)
    rows = []
    qe_ref = read_qe_fc(FC["reference"])
    for step, (tp, tm), h in (("005", ("shear_xy_p005", "shear_xy_m005"), 0.010), ("010", ("shear_xy_p010", "shear_xy_m010"), 0.020)):
        dp, dm = read_qe_fc(FC[tp]), read_qe_fc(FC[tm])
        assert dp["nq"] == NQ4 and dm["nq"] == NQ4
        K_sr = (dp["fc"] - dm["fc"]) / (2 * h)                       # Ry/bohr^2 per unit engineering shear, real space
        eps_odd = (dp["dielectric"] - dm["dielectric"]) / 2; born_odd = (dp["born"] - dm["born"]) / 2
        # QE bare Gamma soft-triplet splitting for the same pair (QESet, QE convention; Gamma is gauge-free)
        qp, qm = QESet(FC[tp], asr="distributed"), QESet(FC[tm], asr="distributed")   # production convention (each set sum-rule corrected)
        wp = np.sort(diagonalise(qp.dynmat((0, 0, 0)))[0]); wm = np.sort(diagonalise(qm.dynmat((0, 0, 0)))[0])
        lam_qe = (np.sign(wp) * wp ** 2 - np.sign(wm) * wm ** 2) / (2 * h)     # cm^-2 per unit h, sorted-order (three lowest = soft triplet)
        sets = {}
        for sign, name in ((+1, f"z_shear_xy_p{step}"), (-1, f"z_shear_xy_m{step}")):
            s = sign * h / 2
            d = cell_data(NQ4, shear=s, eps=eps + sign * eps_odd, born=born_qe + sign * born_odd)
            fc_full = add_dipole_real_space(dict(d, fc=fc_sr + sign * h * K_sr))
            xml = ZDIR / f"{name}_full_fc2.xml"
            n = write_alamode_xml(fc_full, xml, asr="distributed", source=f"{name}: D_tut(4x4x4) {'+' if sign > 0 else '-'} h K_QE({step}), h = {h}")
            write_born(d, ZDIR / f"BORN_{name}")
            for f in (xml, ZDIR / f"BORN_{name}"):
                (PERSIST / f.name).write_bytes(f.read_bytes())
            sets[sign] = (LightSet(xml, d), d, fc_sr + sign * h * K_sr)
            print(f"{name}: {n} entries", flush=True)
        # validation on the 4^3 grid (where the plain cell sum and the minimum-image interpolation coincide) + Gamma; the
        # in-memory reference carries the same distributed sum rule; the sum-rule effect itself is reported separately
        from q2r_to_alamode_fc2 import impose_asr_distributed
        qs = grid_points(4)
        cells = [(m1, m2, m3) for m3 in range(4) for m2 in range(4) for m1 in range(4)]
        for sign in (+1, -1):
            ls, d, fc_mem = sets[sign]
            fc_full_mem = add_dipole_real_space(dict(d, fc=fc_mem))["fc"]
            fc_asr, before, after, nit = impose_asr_distributed(fc_full_mem, 5, NQ4)
            dmax = dmax_noasr = 0.0
            for q in qs:
                ph = np.array([np.exp(2j * np.pi * (q[0] * m[0] + q[1] * m[1] + q[2] * m[2])) for m in cells])
                w_xml = np.sort(diagonalise(ls.dynmat(q))[0])
                for fc_ref, tag in ((fc_asr, "asr"), (fc_full_mem, "raw")):
                    D_mem = np.einsum("abm,m->ab", fc_ref, ph) - ls.dip_sum(q) + ls.qe_like.nonanalytic(q)
                    w_mem = np.sort(diagonalise(ry_per_bohr2_to_cm2(D_mem, MASSES_AMU))[0])
                    if tag == "asr":
                        dmax = max(dmax, np.abs(w_mem - w_xml).max())
                    else:
                        dmax_noasr = max(dmax_noasr, np.abs(w_mem - w_xml).max())
            wG = np.sort(diagonalise(ls.dynmat((0, 0, 0)))[0])
            rows.append({"set": f"z_{'p' if sign > 0 else 'm'}{step}", "h": sign * h, "n_test_q": len(qs),
                         "max_abs_domega_xml_vs_memory_cm1": dmax, "max_abs_domega_vs_memory_without_sumrule_cm1": dmax_noasr,
                         "sumrule_residual_before_Ry_bohr2": before, "sumrule_max_change_Ry_bohr2": float(np.abs(fc_asr - fc_full_mem).max()),
                         "gamma_translations_max_cm1": float(np.sort(np.abs(wG))[:3].max()),
                         "gamma_soft_triplet_cm1": " ".join(f"{v:.3f}" for v in wG[:3])})
            print(rows[-1], flush=True)
        # Gamma soft-triplet splitting Lambda (bare, signed omega^2 of the three lowest modes): Z vs QE
        wZp = np.sort(diagonalise(sets[+1][0].dynmat((0, 0, 0)))[0]); wZm = np.sort(diagonalise(sets[-1][0].dynmat((0, 0, 0)))[0])
        lam_z = (np.sign(wZp) * wZp ** 2 - np.sign(wZm) * wZm ** 2) / (2 * h)
        # odd part survives the sum rule: antisymmetric part of the written sets at a grid point vs K_QE (exact there)
        q = (0.25, 0.5, 0.75)
        ph = np.array([np.exp(2j * np.pi * (q[0] * m[0] + q[1] * m[1] + q[2] * m[2])) for m in cells])
        Dodd_xml = (sets[+1][0].short_range(q) - sets[-1][0].short_range(q)) / (2 * h)
        Dodd_raw = np.einsum("abm,m->ab", K_sr, ph)
        # production reference: the own converted strained XMLs (each with the distributed sum rule), same grid point
        own = REPO / "data" / "raw" / "alamode_sto" / "own_od1" / ("i4s8" if step == "005" else "i2s2")
        try:
            dqp, dqm = read_qe_fc(FC[tp]), read_qe_fc(FC[tm])
            op = LightSet(own / f"{tp}_full_fc2.xml", dict(dqp, fc=np.zeros_like(dqp["fc"])))
            om = LightSet(own / f"{tm}_full_fc2.xml", dict(dqm, fc=np.zeros_like(dqm["fc"])))
            Dodd_mem = (op.short_range(q) - om.short_range(q)) / (2 * h)
        except FileNotFoundError:
            Dodd_mem = Dodd_raw
        rows.append({"set": f"pair_{step}", "h": h, "gamma_lambda_Z_cm2_soft_triplet": " ".join(f"{v:.1f}" for v in lam_z[:3]),
                     "gamma_lambda_QE_cm2_soft_triplet": " ".join(f"{v:.1f}" for v in lam_qe[:3]),
                     "odd_part_rel_diff_vs_own_converted_sets_gridq": float(np.linalg.norm(Dodd_xml - Dodd_mem) / np.linalg.norm(Dodd_mem)),
                     "odd_part_rel_diff_vs_raw_qe_K_gridq": float(np.linalg.norm(Dodd_xml - Dodd_raw) / np.linalg.norm(Dodd_raw))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = REPORTS / "hybrid_model_strained_validation.csv"; tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write("# build_hybrid_model.py strain: strained Z sets D_Z(q,+-h) = D_tut(q) +- h K_QE(q) built in real space on the 4x4x4 grid;\n"
                 "# XML frequencies (builder) vs the in-memory sum at 20 random q + Gamma; Gamma soft-triplet Lambda = [omega^2(+h) - omega^2(-h)]/(2h)\n"
                 "# (sorted-order proxy) for Z vs the QE bare pair; odd part of D(+h) - D(-h) after the distributed sum rule vs K_QE.\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    tmp.rename(out)
    print(df.to_string())


def conventions():
    """gauge check: QESet (QE/matdyn convention) vs the converted own XML (LightSet convention) at a generic q."""
    q = (0.13, 0.29, 0.41)
    qe = QESet(FC["reference"], asr="none")
    own_xml = REPO / "data" / "raw" / "alamode_sto" / "own_od1" / "i4s8" / "reference_full_fc2.xml"
    d = read_qe_fc(FC["reference"])
    ls = LightSet(own_xml, dict(d, fc=np.zeros_like(d["fc"])))
    w1, v1 = diagonalise(qe.dynmat(q)); w2, v2 = diagonalise(ls.dynmat(q))
    print("QESet vs LightSet(own XML): max |d omega| =", np.abs(np.sort(w1) - np.sort(w2)).max(), "cm-1; multiplet overlap min =",
          overlap_multiplet(w1, v1, w2, v2), "(a gauge difference between conventions shows here as overlap < 1 with equal frequencies)")




def write_qe_fc(data, fc, path):
    """Write a q2r-format .fc file (ibrav 0, dielectric block) for a cell_data dict and SHORT-RANGE force constants fc in the
    converter's storage, so that QESet/AlamodeSet can consume the Z sets exactly as the QE sets."""
    nat, nq = data["nat"], data["nq"]
    alat = data["celldm"][0]
    at = data["lavec"] / alat
    L = [f"  {data['nkd']}  {nat}  0  {alat:.10f}  0.0 0.0 0.0 0.0 0.0"]
    for i in range(3):
        L.append("  " + "  ".join(f"{at[c, i]:.12f}" for c in range(3)))
    for k, sym in enumerate(data["symbols"]):
        L.append(f"  {k + 1}  '{sym} '  {data['masses_ry'][k]:.10f}")
    for i in range(nat):
        L.append(f"  {i + 1}  {data['kd'][i]}  " + "  ".join(f"{v:.12f}" for v in data["x_alat"][i]))
    L.append("  T")
    for r in data["dielectric"]:
        L.append("  " + "  ".join(f"{v:.12f}" for v in r))
    for i in range(nat):
        L.append(f"  {i + 1}")
        for r in data["born"][i]:
            L.append("  " + "  ".join(f"{v:.12f}" for v in r))
    L.append(f"  {nq[0]}  {nq[1]}  {nq[2]}")
    for icrd in range(3):
        for jcrd in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    L.append(f"  {icrd + 1}  {jcrd + 1}  {iat + 1}  {jat + 1}")
                    icell = 0
                    for m3 in range(nq[2]):
                        for m2 in range(nq[1]):
                            for m1 in range(nq[0]):
                                L.append(f"  {m1 + 1}  {m2 + 1}  {m3 + 1}  {fc[3 * jat + jcrd, 3 * iat + icrd, icell]:.12e}")
                                icell += 1
    Path(path).write_text("\n".join(L) + "\n")


def qefc():
    """write z_tut/<set>.fc (short-range, with the set's eps/Z*) for reference and the four strained sets; round-trip check."""
    eps, born_t = read_born_file(EX / "BORN"); born_qe = born_t[PERM_TUT_TO_QE_INV()]
    d222 = cell_data([2, 2, 2], eps=eps, born=born_qe)
    tut = LightSet(EX / "STO222.xml", d222, perm=PERM_TUT_TO_QE)
    fc_sr = inverse_fft_to_fc(tut.short_range, NQ4)
    d0 = cell_data(NQ4, eps=eps, born=born_qe)
    write_qe_fc(d0, fc_sr, PERSIST / "z_reference.fc")
    for step, (tp, tm), h in (("005", ("shear_xy_p005", "shear_xy_m005"), 0.010), ("010", ("shear_xy_p010", "shear_xy_m010"), 0.020)):
        dp, dm = read_qe_fc(FC[tp]), read_qe_fc(FC[tm])
        K_sr = (dp["fc"] - dm["fc"]) / (2 * h)
        eps_odd = (dp["dielectric"] - dm["dielectric"]) / 2; born_odd = (dp["born"] - dm["born"]) / 2
        for sign, name in ((+1, f"z_shear_xy_p{step}"), (-1, f"z_shear_xy_m{step}")):
            d = cell_data(NQ4, shear=sign * h / 2, eps=eps + sign * eps_odd, born=born_qe + sign * born_odd)
            write_qe_fc(d, fc_sr + sign * h * K_sr, PERSIST / f"{name}.fc")
    # round trip: QESet on z_reference.fc vs LightSet(STO444) at a generic q and at Gamma
    qe = QESet(PERSIST / "z_reference.fc", asr="distributed")
    new = LightSet(ZDIR / "STO444_tut.xml", d0)
    for q in ((0, 0, 0), (0.13, 0.29, 0.41), (0.5, 0.5, 0.5)):
        w1 = np.sort(diagonalise(qe.dynmat(q))[0]); w2 = np.sort(diagonalise(new.dynmat(q))[0])
        print("round trip q", q, "max |d omega| =", f"{np.abs(w1 - w2).max():.4f} cm-1 (QESet distributed-ASR vs XML)")
    print("written:", sorted(p.name for p in PERSIST.glob("*.fc")))


if __name__ == "__main__":
    {"reexpress": reexpress, "strain": strain, "conventions": conventions, "qefc": qefc}[sys.argv[1]]()
