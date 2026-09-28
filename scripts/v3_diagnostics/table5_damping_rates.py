#!/usr/bin/env python3
"""Table 5: acoustic damping implied by eta_xyxy(300 K) of the production model at the wave vectors of Maerten et al.

Amplitude decay rate r = eta q^2 / (2 rho) (10^9 s^-1), comparable with the time-domain damping rate Gamma of Maerten et al.;
frequency-domain half width r / (2 pi) (GHz), comparable with their Brillouin line width beta = Gamma / (2 pi). The
finite-frequency factor eta'(Omega)/eta'(0) = sum_k eta_k / (1 + Omega^2 tau_k^2) / eta with Omega = v q, v = 8000 m/s.

Output: data/processed/v3_diagnostics/table5_damping_rates.csv
Usage: uv run python scripts/v3_diagnostics/table5_damping_rates.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAG  # noqa: E402

RHO, V = 5110.0, 8000.0
mt = pd.read_csv(DIAG / "mode_table_SrTiO3_300K_v6.csv", comment="#")
eta_k, tau_k = mt["eta_contrib_Pas"].to_numpy(), mt["tau_stress_ps"].to_numpy() * 1e-12
eta = eta_k.sum()
rows = []
for q_um in (52.0, 58.0):
    q = q_um * 1e6
    om = V * q
    fac = np.sum(eta_k / (1 + (om * tau_k) ** 2)) / eta
    r = eta * q ** 2 / (2 * RHO)
    rows.append({"q_um_inv": q_um, "wave_frequency_GHz": om / (2 * np.pi) / 1e9, "eta_Pas": eta,
                 "r_static_1e9_per_s": r / 1e9, "hwhm_static_GHz": r / (2 * np.pi) / 1e9, "finite_frequency_factor": fac,
                 "r_dynamic_1e9_per_s": r * fac / 1e9, "hwhm_dynamic_GHz": r * fac / (2 * np.pi) / 1e9})
df = pd.DataFrame(rows)
out = DIAG / "table5_damping_rates.csv"
tmp = out.with_suffix(".tmp")
with open(tmp, "w") as fh:
    fh.write("# table5_damping_rates.py: r = eta q^2/(2 rho) amplitude decay rate (10^9 s^-1) and HWHM r/(2 pi) (GHz); rho = 5110 kg/m^3,\n"
             "# v = 8000 m/s; eta = production eta_xyxy(300 K) (shear component used as a scale proxy for a longitudinal wave).\n")
    df.to_csv(fh, index=False, float_format="%.10g")
tmp.rename(out)
print(df.to_string(index=False))
