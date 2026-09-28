#!/usr/bin/env python3
"""eta_z.py - eta_A/B/C on a Z-surface mesh with the machinery: redirects the bare sets to the Z q2r-format files
(z_reference.fc, z_shear_xy_*.fc), the bare RTA to the Z bare result, and the renormalised XML names to the Z driver output.
Usage: uv run python eta_z.py <mesh tag> [--fixlw]  (fixlw: linewidths and class map fixed at the Z 2/2 RTA)"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _paths import DIAG, RAW, REPO, SCPH  # noqa: E402,F401  (sets sys.path)
import coupling_kernel as ck  # noqa: E402

Z = REPO / "data" / "raw" / "alamode_sto" / "z_tut"
for tag in list(ck.FC):
    ck.FC[tag] = Z / f"z_{tag}.fc"
    ck.BARE_XML[tag] = (Z / "i2s12" / "z_reference_full_fc2.xml") if tag == "reference" else (Z / f"z_{tag}_full_fc2.xml")
import eta_constructions as ec  # noqa: E402

ec.RTA_BARE = Z / "bare" / "STO_RTA_z_bare.result"
mesh = sys.argv[1]
fixlw = "--fixlw" in sys.argv
outer = "--outer" in sys.argv          # 9^3/11^3/13^3/15^3 outer-mesh sequence , separate output prefix
rta = Z / ("i2s2" if fixlw else mesh) / f"STO_RTA_{'i2s2' if fixlw else mesh}_od1_300K.result"
sys.argv = ["eta_constructions.py", "od1", "--meshes"] + (["9", "11", "13", "15"] if outer else ["11"]) + ["--surface-dir", str(Z / mesh),
            "--xml-pattern", f"renorm_z_{{tag}}_{mesh}_{{variant}}_300K.xml", "--rta", str(rta),
            "--out-prefix", f"eta_z_{mesh}" + ("_fixlw" if fixlw else "") + ("_outer" if outer else "")]
ec.main()
