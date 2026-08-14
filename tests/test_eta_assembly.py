"""Regression tests for the Stage-C eta assembly's mode-mapping layer.

Two failure modes actually hit during the first assembly (see
data/processed/reports/eta_SrTiO3_stageC.md) are pinned here:

1. CHARACTER-BLIND GAMMA: frequency-binning all modes together hands
   long-lived acoustic linewidths to soft-TO modes that happen to share
   the same renormalized frequency window (tau wrong by ~10x, eta by ~3x).
   The character-aware split (bare omega < SOFT_CHAR_CM1 -> soft-sector
   statistics; the rest -> frequency-binned stable map) must keep the two
   populations separate.

2. GAMMA-POINT RANK-PAIRING: the bare file sorts the imaginary TO1 below
   the acoustic zeros while the renormalized file sorts the zeros below
   the renormalized TO1, so raw (q,branch)-index pairing maps bare TO1 ->
   0 and poisons the unstable region of the eigenvalue map. Per-q rank
   pairing with the exact-zero acoustic entries excluded must pair
   TO1(bare, imaginary) with TO1(renormalized).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402


def _synthetic_parse_result(path, target_temp=300):
    """Stand-in for crosscheck_alamode_sto_tau.parse_result with a Gamma
    point exhibiting the real files' ordering hazard, plus a q-point where
    an acoustic and a soft mode share the renormalized frequency window."""
    name = str(path)
    if "production" in name:  # bare
        freq = {
            # Gamma: imaginary TO1 triplet sorted BELOW the acoustic zeros
            (1, 1): -60.0, (1, 2): -60.0, (1, 3): -60.0,
            (1, 4): 0.0, (1, 5): 0.0, (1, 6): 0.0,
            (1, 7): 200.0, (1, 8): 200.0, (1, 9): 200.0,
            # off-Gamma: acoustic 85 (stable char), soft 40 (soft char),
            # optic 300 (stable char)
            (2, 1): 40.0, (2, 2): 85.0, (2, 3): 300.0,
            # second stable q so the acoustic frequency bin holds >= 2 modes
            # (the map drops single-entry bins)
            (3, 1): 80.0, (3, 2): 295.0,
        }
        return freq, {}
    freq = {
        # Gamma renormalized: acoustic zeros sort BELOW the stabilized TO1
        (1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0,
        (1, 4): 170.0, (1, 5): 170.0, (1, 6): 170.0,
        (1, 7): 210.0, (1, 8): 210.0, (1, 9): 210.0,
        # off-Gamma: soft 40 renormalizes INTO the acoustic's window
        (2, 1): 88.0, (2, 2): 90.0, (2, 3): 310.0,
        (3, 1): 92.0, (3, 2): 305.0,
    }
    gamma = {
        (1, 4): 8.0, (1, 5): 8.0, (1, 6): 8.0,
        (1, 7): 1.0, (1, 8): 1.0, (1, 9): 1.0,
        (2, 1): 6.0,   # soft-character mode: heavily damped
        (2, 2): 0.2,   # acoustic at nearly the same renormalized frequency
        (2, 3): 1.0,
        (3, 1): 0.25,  # second acoustic in the same frequency bin
        (3, 2): 1.1,
    }
    return freq, gamma


@pytest.fixture()
def maps(monkeypatch):
    monkeypatch.setattr(eta_mod, "parse_result", _synthetic_parse_result)
    return eta_mod.build_maps(300)


def test_gamma_point_rank_pairing_survives_reordering(maps):
    map_lambda, _, _, soft_floor, lam_min = maps
    # bare TO1 eigenvalue -60^2 must map to the RENORMALIZED TO1 (170^2),
    # not to the acoustic zero it shares a raw branch index with
    assert map_lambda(-3600.0) == pytest.approx(170.0**2)
    # and the theory-side branch-minimum floor is the renormalized TO1
    assert soft_floor == pytest.approx(170.0)
    assert lam_min == pytest.approx(-3600.0)


def test_sanity_gate_uses_revised_band():
    # revised 2026-07-24 expectation band (eta_SrTiO3_stageC.md, section
    # B): 1e-3..1e-2 Pa s. The old 1e-4..1e-3 decade came from an
    # O(1)-gamma estimate and sits below every measured damping point —
    # the production value 3.89e-3 must PASS and the old decade's
    # midpoint must not.
    assert eta_mod.ETA_300K_BAND_PAS == (1e-3, 1e-2)
    ok, label = eta_mod.sanity_gate(3.89e-3)
    assert ok and label == "PASS"
    ok, label = eta_mod.sanity_gate(3e-4)
    assert not ok and label == "OUTSIDE EXPECTED BAND"


def test_character_aware_gamma_keeps_populations_separate(maps):
    _, map_gamma, soft_gamma_median, _, _ = maps
    # soft-sector statistics: only the bare-soft modes (Gamma TO1 at 8.0,
    # off-Gamma soft at 6.0) — median 8.0; the acoustic 0.2 must NOT enter
    assert soft_gamma_median == pytest.approx(np.median([8.0, 8.0, 8.0, 6.0]))
    # stable map near 90 cm-1: the acoustic 0.2, NOT contaminated by the
    # soft mode's 6.0 that sits at 88 cm-1 in the same frequency bin —
    # a frequency-binned-only map would return median(6.0, 0.2) = 3.1,
    # wrong for both populations
    assert map_gamma(90.0) < 1.0
    frequency_blind_median = np.median([6.0, 0.2])
    assert abs(map_gamma(90.0) - frequency_blind_median) > 1.0
