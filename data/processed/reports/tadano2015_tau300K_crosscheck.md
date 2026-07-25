# Tadano & Tsuneyuki (2015) Fig. 7 reference values — SrTiO3 tau(300 K) / kappa(T) cross-check

Source: T. Tadano and S. Tsuneyuki, Phys. Rev. B 92, 054301 (2015)
(arXiv:1506.01781), Sec. IV C and Fig. 7. Values below digitized by eye
from the arXiv v1 figure at high zoom; read-off uncertainty ~±0.15 W/mK
on the kappa panel and ~±30% (log axis) on the tau inset bands.

## Their computational setup (for like-for-like comparison)

- VASP, PBEsol, PAW, ENCUT = 550 eV, 12x12x12 k-grid; optimized
  a = 3.896 A (our production PBEsol: 3.8930 A, -0.08% apart).
- Harmonic IFCs: finite displacement, 2x2x2 cubic supercell (40 atoms).
  Anharmonic IFCs: compressive sensing (LASSO).
- SCPH on an 8x8x8 q1 grid; kappa via BTE-RTA on a **12x12x12** q grid
  [their Eq. (29)].
- Linewidth: bubble diagram with **SCPH frequencies and eigenvectors**
  substituted [Eq. (30)]; Gamma_q is a HWHM-type imaginary self-energy;
  **tau_q = 1/(2 Gamma_q(Omega_q))** [below Eq. (29)].
- Note: the ALAMODE-distributed reference FCs
  (example/SrTiO3/reference/STO_anharm.xml) are a later regeneration
  documented as "VASP, PBEsol, ENCUT 550 eV, 2x2x2 supercell; harmonic
  by OLS; anharmonic by LASSO from 40 training structures" — same
  functional/cell as the paper but not byte-identical FCs. Modest
  (tens of %) differences from the paper's own numbers are plausible;
  order-of-magnitude differences are not.

## kappa_L(T), "This work (12x12x12)" curve (digitized)

| T (K) | kappa_L (W/mK) |
|---|---|
| 200 | 10.3 |
| 300 | 8.7 |
| 400 | 7.6 |
| 500 | 6.9 |
| 600 | 6.3 |
| 700 | 5.9 |
| 800 | 5.4 |
| 900 | 5.1 |
| 1000 | 4.9 |

Experiment at ~300 K (same figure): Muta et al. ~11.1 W/mK; Popuri et
al. ~9.8 W/mK. Paper's own text: calculated kappa agrees well with
experiment, with underestimation in the low-T region.

## tau_q(300 K) inset (digitized band summary, log tau axis)

| omega_q band (cm-1) | tau_q range (ps) | Feature |
|---|---|---|
| < 50 | ~5–20 | long-lived acoustic tail; global max ~15–20 ps |
| 100–150 | ~1–4 | dense cluster (soft TO / acoustic mixing) |
| 150–200 | ~0.7–3 | declining |
| 200–450 | ~0.4–1.2 | broad plateau centered ~0.6–0.8 ps |
| ~460–490 | ~1–2 | local rise |
| ~500–550 | ~0.12–0.5 | pronounced dip (global min ~0.12 ps) |
| ~560–730 | — | spectral gap (no modes) |
| 730–830 | ~0.2–0.45 | top optic branch |

Global envelope: ~0.12–20 ps. Their text: c-STO lifetimes are even
smaller than PbTe's, but kappa_L is higher due to larger group
velocities.

## Cross-check protocol for the local anphon RTA run (data/raw/alamode_sto/)

1. Plot per-mode tau_q(300 K) = 1/(2 Gamma_q) from the local run on the
   inset's axes (omega 0–850 cm-1, tau 0.1–100 ps, log y) and compare
   band-by-band against the table above. Pass criterion (suggested):
   band medians within ~2x, same qualitative structure (low-omega tail,
   200–450 plateau, 500–550 dip, gap, top branch).
2. Compare kappa(300 K). **Current local value 3.34 W/mK vs the paper's
   8.7 W/mK (12x12x12) — a factor ~2.6 low. This must be diagnosed
   before the SrTiO3 Gamma_anh(q,T) set is treated as validated.**
   Candidate causes, in likely order:
   (a) plain MODE=RTA uses the bare harmonic fc2 frequencies/velocities
       from the xml, whereas the paper substitutes SCPH-renormalized
       Omega_q(T) and eigenvectors into both the linewidth [Eq. (30)]
       and the group velocities in Eq. (29) — in a soft-mode material
       this is a first-order effect, and the bare surface's handling of
       the unstable/soft branches (dropped or imaginary) directly
       biases kappa;
   (b) BTE mesh: local run 8x8x8 vs paper 12x12x12;
   (c) tutorial-FC regeneration differences (should be modest, see
       above).
   If (a) is confirmed (e.g. rerunning with the SCPH-coupled kappa path
   closes most of the gap), the same caveat applies to the low-omega
   Gamma_anh values used for Route H, which is exactly the manifold
   Route H exists for — the renormalized-frequency requirement is not
   optional there.
3. The tau comparison (step 1) localizes the discrepancy in omega: if
   the 200–450 cm-1 plateau matches but the < 150 cm-1 tail does not,
   cause (a) is implicated; a uniform offset across all bands points to
   (b)/(c) or a convention slip (check the factor 2 in tau = 1/(2 Gamma)
   against the pinned gamma-convention test).

## RESULTS — protocol executed (2026-07-23), both prongs PASS at T=300 K

Executed via `scripts/crosscheck_alamode_sto_tau.py` on two runs:
bare-RTA (`STO_RTA_production.result`) and SCPH-coupled
(`STO_RTA_scph_300K.result`, harmonic fc2 replaced by the
dfc2-renormalized 300 K fc2 via anphon's FC2XML option). Band tables:
`alamode_sto_tau300K_bands_bare.csv` / `..._scph.csv` in this directory.

**Prong 1 — kappa(300 K): PASS.** Bare 3.34 W/mK -> SCPH-coupled
**9.53 W/mK** vs the paper's 8.7 (+9.5%, inside the digitization + FC
regeneration tolerance). Cause (a) confirmed: the gap was the missing
SCPH substitution, concentrated at low omega exactly as step 3 of the
protocol anticipated.

**Prong 2 — acoustic tau in the 5–20 ps band: PASS.** Caveat on binning:
the SCPH run has NO stable modes below 50 cm-1 (on the 8x8x8 mesh the
smallest nonzero-|q| acoustic frequencies are ~55 cm-1 = v*q_min; the
paper's <50 cm-1 tail comes from its finer 12x12x12 mesh reaching
smaller |q|), so the acoustic tail must be read branch-resolved rather
than from the [0,50) frequency bin. Branch-resolved tau(300 K),
SCPH-coupled run:

| branch | n | omega range (cm-1) | tau med (ps) | tau p90 | tau max |
|---|---|---|---|---|---|
| 1 (acoustic) | 34 | 54.9–130.2 | 2.39 | 4.81 | **16.19** |
| 2 (acoustic) | 34 | 54.9–136.3 | 3.10 | 3.98 | **17.54** |
| 3 (acoustic) | 34 | 67.6–166.1 | 2.75 | 4.36 | 4.64 |

The nearest-to-Gamma acoustic modes (q index 2, omega = 54.9 cm-1) sit
at 16.2 / 17.5 ps — inside the paper's 5–20 ps band and matching its
"global max ~15–20 ps". All sub-100 cm-1 acoustic modes land at
6.3–17.5 ps (in-band); the branch medians (2.4–3.1 ps) belong to the
100–170 cm-1 portions of the branches, consistent with the paper's own
1–4 ps dense-cluster band there. The remaining bands (200–450 plateau,
460–490 rise, 500–550 dip, 560–730 gap, 730–830 top branch) all match
the digitized ranges (see the scph CSV).

**Consequence**: both prongs pass -> the `<200 cm-1 PROVISIONAL` flag on
the SrTiO3 Gamma_anh dataset is LIFTED **at T=300 K** (recorded in
`data/raw/alamode_sto/PROVENANCE.md`). The other 20 temperatures of the
production set remain bare-RTA and stay provisional until the per-T
`dfc2` + RTA regeneration is run (~90 min local, pending).

## Related literature files now located (local, not committed)

- Tadano_STO.pdf = arXiv:1506.01781 (this paper).
- Tadano_BTO.pdf = Masuki, Nomoto, Arita, Tadano, arXiv:2205.08789
  (PRB 106, 224104 (2022)): SCP-based finite-T structural optimization
  applied to BaTiO3 phase transitions. NOT the pending PRB 103, 094108
  (2021) pull, but contains SCP treatment of the BaTiO3 soft mode across
  the cubic->tetra transition — assess as a Route-H BTO source/
  cross-check before requesting the 2021 paper via library.
- Vogt (1995) Fig. 6 digitization already exists, committed, in
  ferroelectric-ins-ml/qe/anchoring/vogt1995_fig6/ (F1u/A2u/Eu
  omega_s(T) + damping gamma(T), headerless CSV, T[K], value[cm-1];
  damping is FULL gamma, Gamma_HWHM = gamma/2, convention CONFIRMED via
  Vogt Eq. (13) per that folder's README). Import/reference instead of
  re-digitizing. Derived anchors there: omega0(300 K) = 89.24 cm-1 =
  11.06 meV; gamma(300 K) = 23.0 cm-1 full -> Gamma0 = 11.5 cm-1.
