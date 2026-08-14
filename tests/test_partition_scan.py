"""Regression tests for the Route S / Route H partition-sensitivity scan.

The scan (scripts/scan_partition_sensitivity.py) varies the partition
frequency of the production assembly through the cutoff_cm1 parameter of
compute_eta_SrTiO3.assemble. Pinned here:

1. the default partition is unchanged at 175 cm-1 — the parameterization
   must not move the production number;
2. moving the cutoff reclassifies a shell mode between Route S and
   Route H, and the two routes' gamma values for the same mode differ by
   exactly the denominator ratio omega_r^2/omega0^2 (the strained-cell
   coupling D is shared);
3. shell_table reports rel_diff = (gamma_H - gamma_S)/|gamma_S|
   consistently with that identity.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compute_eta_SrTiO3 as eta_mod  # noqa: E402
import scan_partition_sensitivity as scan_mod  # noqa: E402


def _synthetic_parse_result(path, target_temp=300):
    """Two q-points, stable modes only; renormalized frequencies stiffened
    ~15% relative to bare, mimicking the SCPH map in the 150-200 shell."""
    if "production" in str(path):  # bare
        freq = {(1, 1): 100.0, (1, 2): 160.0, (1, 3): 300.0,
                (2, 1): 110.0, (2, 2): 170.0, (2, 3): 320.0}
        return freq, {}
    # pairs land in shared frequency bins so the binned Gamma(omega_r)
    # map (>= 2 modes per bin) is populated
    freq = {(1, 1): 115.0, (1, 2): 184.0, (1, 3): 330.0,
            (2, 1): 116.0, (2, 2): 185.0, (2, 3): 331.0}
    gamma = {k: 1.0 for k in freq}
    return freq, gamma


def _rows():
    return [
        {"iq": 5, "branch": 1, "omega_ref": 160.0, "D": -8000.0,
         "acoustic": False},
        {"iq": 5, "branch": 2, "omega_ref": 300.0, "D": 5000.0,
         "acoustic": False},
        {"iq": 0, "branch": 1, "omega_ref": 0.0, "D": 0.0, "acoustic": True},
    ]


def _vogt(temperature):
    return None, None


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(eta_mod, "parse_result", _synthetic_parse_result)


def test_default_cutoff_is_production_175():
    rows = _rows()
    eta_default, sec_default, _ = eta_mod.assemble(300, rows, _vogt)
    eta_175, sec_175, _ = eta_mod.assemble(300, rows, _vogt, cutoff_cm1=175.0)
    assert eta_default == pytest.approx(eta_175)
    assert sec_default == sec_175


def test_cutoff_reclassifies_shell_mode_with_denominator_ratio():
    rows = _rows()
    _, _, _, det_150 = eta_mod.assemble(300, rows, _vogt, cutoff_cm1=150.0,
                                        return_details=True)
    _, _, _, det_200 = eta_mod.assemble(300, rows, _vogt, cutoff_cm1=200.0,
                                        return_details=True)
    mode_150 = next(d for d in det_150 if d["omega0"] == 160.0)
    mode_200 = next(d for d in det_200 if d["omega0"] == 160.0)
    assert mode_150["sector"] == "routeS"
    assert mode_200["sector"] == "routeH_stable"
    # same coupling D, denominators omega0^2 (S) vs omega_r^2 (H):
    ratio = mode_150["gruneisen"] / mode_200["gruneisen"]
    expected = mode_200["omega_r"] ** 2 / mode_150["omega0"] ** 2
    assert ratio == pytest.approx(expected, rel=1e-12)
    # everything except gamma and sector is held fixed
    assert mode_150["omega_r"] == pytest.approx(mode_200["omega_r"])
    assert mode_150["tau_s"] == pytest.approx(mode_200["tau_s"])


def test_shell_table_rel_diff_identity():
    rows = _rows()
    map_lambda = eta_mod.build_maps(300)[0]
    table = scan_mod.shell_table(rows, map_lambda)
    entry = next(t for t in table if t["omega0"] == 160.0)
    assert entry["gamma_H"] == pytest.approx(
        entry["gamma_S"] * entry["omega0"] ** 2 / entry["omega_r"] ** 2)
    expected_rel = (entry["gamma_H"] - entry["gamma_S"]) / abs(entry["gamma_S"])
    assert entry["rel_diff"] == pytest.approx(expected_rel)
    # 300 cm-1 mode is outside the 150-200 shell
    assert all(t["omega0"] <= 200.0 for t in table)
