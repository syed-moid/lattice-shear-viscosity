"""Parser for QE matdyn.x/dynmat.x mode files (flvec output).

The .modes file contains, per q-point, the block written by QE's
writemodes: a `q = qx qy qz` line, then for each of the 3*nat modes a
`freq (i) = x [THz] = y [cm-1]` line followed by nat lines of
`( Re_x Im_x Re_y Im_y Re_z Im_z )` displacement components.

Vectors are returned as complex arrays of shape (nmodes, nat, 3), exactly
as printed (QE prints eigendisplacements; mass-weighting is reapplied by
the consumer before overlap computations, which makes the overlap metric
independent of the printing convention).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["read_modes"]

_Q_RE = re.compile(r"q =\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
_FREQ_RE = re.compile(r"freq \(\s*(\d+)\)\s*=\s*([-\d.]+)\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]")
_VEC_RE = re.compile(r"\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+"
                     r"([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)")


def read_modes(path: str | Path):
    """Parse a matdyn/dynmat modes file.

    Returns
    -------
    list of (q, freqs_cm1, vectors) with q a (3,) float array in the units
    used by the run, freqs_cm1 of shape (nmodes,), vectors complex of shape
    (nmodes, nat, 3).
    """
    blocks = []
    q = None
    freqs: list[float] = []
    vectors: list[list[complex]] = []
    current: list[complex] | None = None

    def flush():
        if q is not None and freqs:
            arr = np.array(vectors, dtype=complex)  # (nmodes, nat, 3)
            blocks.append((np.array(q), np.array(freqs), arr))

    for line in Path(path).read_text().splitlines():
        m = _Q_RE.search(line)
        if m:
            flush()
            q = [float(g) for g in m.groups()]
            freqs, vectors, current = [], [], None
            continue
        m = _FREQ_RE.search(line)
        if m:
            freqs.append(float(m.group(3)))
            current = []
            vectors.append(current)
            continue
        m = _VEC_RE.search(line)
        if m and current is not None:
            g = [float(x) for x in m.groups()]
            current.append([complex(g[0], g[1]), complex(g[2], g[3]), complex(g[4], g[5])])
    flush()
    return blocks
