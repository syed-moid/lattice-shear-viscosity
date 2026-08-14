#!/usr/bin/env python3
"""Stage C: assemble the SrTiO3 lattice shear viscosity eta_xyxy(T).

Central formula (manuscript Eq. 5, project guardrails):

    eta = (1 / (V_cell N_q k_B T)) sum_qs (hbar*omega)^2 gamma_xy^2 n(n+1) tau

with the exact two-pole lifetime tau = (Gamma^2 + omega^2)/(2 Gamma omega^2)
(latvisc.viscosity.tau_two_pole_exact; reduces to 1/(2 Gamma) underdamped,
Gamma/(2 omega^2) deep overdamped). Units come out Pa s.

Partitioned pipeline (Route S / Route H, cutoff omega0 = 175 cm-1 —
Richardson-validated for SrTiO3, see GATE_1p.md):

  Route S (omega0 >= 175 cm-1): gamma_xy = -D/(2*omega0^2) from the
    eps05 central difference on the 11^3 mesh (D = d(omega^2)/d(eps),
    signed eigenvalues; step size validated by the 5-point Richardson
    analysis, richardson_5pt_SrTiO3.csv). Bare omega0 approximates the
    renormalized frequency well in this manifold; the weight factors
    still use the mapped renormalized frequency for consistency.

  Route H (omega0 < 175 cm-1, stable + unstable): gamma =
    Lambda/(2*omega_r^2(T)) with Lambda = -D from the SAME strained
    cells and omega_r(T) the SCPH-renormalized frequency, obtained by an
    empirical bare->renormalized eigenvalue map built per temperature
    from the ALAMODE pair (bare frequencies from
    STO_RTA_production.result; renormalized from STO_RTA_scph_<T>K.result,
    same 8^3 mesh, matched by (q,branch) index).

  Linewidths Gamma(T): frequency-binned interpolation of the per-mode
    SCPH-coupled Gamma_qs(T) from STO_RTA_scph_<T>K.result (validated at
    300 K against Tadano & Tsuneyuki 2015 — both prongs, see
    tadano2015_tau300K_crosscheck.md; the per-T files use the same
    validated recipe but are not independently cross-checked, flagged
    provisional_method in the output).

  Gamma-point soft sector: the unstable zone-center TO branches use the
    Vogt (1995) experimental soft-mode frequency and HWHM damping
    (softmode_inputs_SrTiO3.csv) where the digitized series covers T;
    outside that range they fall back to the mapped values and the row
    is flagged.

Reads : data/raw/gruneisen_modes/SrTiO3/*.modes,
        data/raw/alamode_sto/STO_RTA_production.result,
        data/raw/alamode_sto/STO_RTA_scph_<T>K.result,
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
from latvisc.viscosity import bose_einstein, tau_two_pole_exact  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ALAMODE_DIR = REPO / "data" / "raw" / "alamode_sto"
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
# Revised expectation band for eta(300 K), SrTiO3 (2026-07-24 sign-off;
# see data/processed/reports/eta_SrTiO3_stageC.md, section B): grounded
# in the measured gamma distribution (zone-rms 4.2, eta weighting
# gamma^2 without sign cancellation) and the Maerten et al. GHz damping
# bracket (3-6e-3 Pa s).
# The old 1e-4..1e-3 decade came from an O(1)-gamma estimate and sits
# below every measured damping point.
ETA_300K_BAND_PAS = (1e-3, 1e-2)


def sanity_gate(eta300: float) -> tuple[bool, str]:
    """Check eta(300 K) against the revised SrTiO3 expectation band."""
    lo, hi = ETA_300K_BAND_PAS
    ok = lo <= eta300 <= hi
    return ok, "PASS" if ok else "OUTSIDE EXPECTED BAND"


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
             return_details=False, cutoff_cm1: float = CUTOFF_CM1):
    """eta_xyxy at one temperature. Returns (eta_total, sector dict, flags)
    — plus a per-mode details list when return_details is True (audit use:
    (iq, branch, sector, omega0, omega_r, gamma_hwhm, gruneisen, tau_s,
    contribution_Pas)).

    extra_gamma_hwhm_cm1: optional callable omega_r_cm1 -> additional HWHM
    (cm-1) added to the anharmonic linewidth (Matthiessen) — used for the
    Tamura isotope channel.

    cutoff_cm1: Route S / Route H partition frequency. The production value
    is CUTOFF_CM1 = 175; scripts/scan_partition_sensitivity.py varies it to
    quantify the sensitivity of the total to the partition choice.
    """
    (map_lambda, map_gamma, soft_gamma_median, soft_floor_theory,
     lam_bare_min) = build_maps(temperature)
    v_omega, v_gamma = vogt(temperature)
    vogt_used = v_omega is not None

    sectors = {"routeS": 0.0, "routeH_stable": 0.0, "routeH_unstable": 0.0,
               "gamma_sector": 0.0}
    n_extrapolated = 0
    n_skipped = 0
    details = []

    for r in rows:
        if r["acoustic"]:
            continue
        omega0 = r["omega_ref"]
        D = r["D"]
        lam0 = np.sign(omega0) * omega0 * omega0
        if lam0 < lam_bare_min:
            n_extrapolated += 1
        lam_r = map_lambda(lam0)
        if lam_r <= 0:
            n_skipped += 1
            continue
        omega_r = float(np.sqrt(lam_r))

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
            gamma_hwhm = float(map_gamma(omega_r))
            sector = "routeS" if omega0 >= cutoff_cm1 else "routeH_stable"

        if extra_gamma_hwhm_cm1 is not None:
            gamma_hwhm = gamma_hwhm + float(extra_gamma_hwhm_cm1(omega_r))

        if sector == "routeS":
            gru = -D / (2.0 * omega0 * omega0)
        else:
            gru = -D / (2.0 * omega_r * omega_r)

        w = omega_r * CM1                       # rad/s
        lw = gamma_hwhm * CM1                   # rad/s (HWHM)
        occupation = bose_einstein(w, temperature)
        tau = float(tau_two_pole_exact(w, lw))
        contrib = (HBAR * w) ** 2 * gru * gru * occupation * (occupation + 1.0) * tau
        sectors[sector] += contrib
        if return_details:
            details.append({"iq": r["iq"], "branch": r["branch"],
                            "sector": sector, "omega0": omega0,
                            "omega_r": omega_r, "gamma_hwhm": gamma_hwhm,
                            "gruneisen": gru, "tau_s": tau,
                            "contrib_raw": contrib})

    norm = 1.0 / (V_CELL * N_Q * K_B * temperature)
    eta_sectors = {k: v * norm for k, v in sectors.items()}
    eta_total = sum(eta_sectors.values())
    flags = {"vogt_used": vogt_used, "n_extrapolated": n_extrapolated,
             "n_skipped": n_skipped}
    if return_details:
        for d in details:
            d["eta_contrib"] = d.pop("contrib_raw") * norm
        return eta_total, eta_sectors, flags, details
    return eta_total, eta_sectors, flags


def main() -> None:
    rows = compute_dataset(MODES_DIR / "SrTiO3", MASSES["SrTiO3"], mesh_n=11)
    vogt, vogt_range = load_vogt()
    print(f"loaded {len(rows)} (q,branch) entries; Vogt series covers "
          f"T in [{vogt_range[0]:.0f}, {vogt_range[1]:.0f}] K")

    out_rows = ["T_K,eta_total_Pas,eta_routeS_Pas,eta_routeH_stable_Pas,"
                "eta_routeH_unstable_Pas,eta_gamma_sector_Pas,frac_lowomega,"
                "vogt_gamma_sector,n_extrapolated_modes,provisional_method_flag"]
    results = {}
    for T in TEMPS:
        eta, sec, flags = assemble(T, rows, vogt)
        low = sec["routeH_stable"] + sec["routeH_unstable"] + sec["gamma_sector"]
        frac_low = low / eta if eta > 0 else float("nan")
        # SCPH-coupled Gamma recipe independently validated at 300 K only
        provisional = "no" if T == 300 else "method_validated_at_300K_only"
        results[T] = (eta, sec, flags, frac_low)
        print(f"T={T:3d} K: eta = {eta:.3e} Pa s  "
              f"[S {sec['routeS']:.2e} | H-stable {sec['routeH_stable']:.2e} | "
              f"H-unstable {sec['routeH_unstable']:.2e} | Gamma-sector "
              f"{sec['gamma_sector']:.2e}]  low-omega frac {frac_low:.3f}  "
              f"vogt={'y' if flags['vogt_used'] else 'MAP-FALLBACK'} "
              f"extrap={flags['n_extrapolated']} {provisional}")
        out_rows.append(
            f"{T},{eta:.6e},{sec['routeS']:.6e},{sec['routeH_stable']:.6e},"
            f"{sec['routeH_unstable']:.6e},{sec['gamma_sector']:.6e},"
            f"{frac_low:.4f},{int(flags['vogt_used'])},"
            f"{flags['n_extrapolated']},{provisional}")

    eta300 = results[300][0]
    _, gate = sanity_gate(eta300)
    print(f"\nSanity gate: eta(300 K) = {eta300:.3e} Pa s -> {gate} "
          f"(expected {ETA_300K_BAND_PAS[0]:.0e}..{ETA_300K_BAND_PAS[1]:.0e}, "
          f"revised band, see eta_SrTiO3_stageC.md)")
    print(f"Low-omega (Route H + Gamma) fraction at 300 K: "
          f"{results[300][3]:.1%} (old bare-surface pathology: [0,50) cm-1 "
          f"alone carried 93% of eta0 with divergent curvature — the "
          f"partition + renormalized denominators tame this)")

    out = REPO / "data" / "processed" / "eta_SrTiO3.csv"
    header = [
        "# eta_SrTiO3.csv - produced by scripts/compute_eta_SrTiO3.py (Stage C)",
        "# eta_xyxy(T), Pa s; Route S/H partition at 175 cm-1 (Richardson-validated);",
        "# gamma from eps05 strained cells (signed eigenvalues); omega_r/Gamma from",
        "# SCPH-coupled ALAMODE per-T runs (validated at 300 K, see",
        "# tadano2015_tau300K_crosscheck.md); Gamma-point soft sector from Vogt 1995",
        "# where covered. See data/processed/reports/eta_SrTiO3_stageC.md.",
    ]
    out.write_text("\n".join(header) + "\n" + "\n".join(out_rows) + "\n")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
