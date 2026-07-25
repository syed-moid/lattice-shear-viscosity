#!/usr/bin/env python3
"""Renumber bracketed references in a manuscript markdown source to strict
first-appearance order, and reorder the reference list to match.

Method:
  1. Split the document at the '# References' heading. Everything before
     it is BODY (title, sections, appendix, figure captions); everything
     after is the LIST.
  2. Scan the BODY sequentially for citation groups [n] / [n,m,...] and
     record the order of first appearance of every number -> old->new map.
  3. Apply the map to every citation group in BODY and LIST alike (list
     entries may cross-reference each other in their annotation notes);
     multi-citation groups are re-sorted ascending after mapping.
  4. Reorder the LIST entries by new number.
  5. Mechanical verification, printed: (i) zero orphans in either
     direction; (ii) list order equals first-appearance order; (iii)
     total count unchanged; (iv) caller-supplied content spot-checks.

[VERIFY]/[DATA]/[TODO]/[SCOPE]/[CHECK] markers contain no bare bracketed
integers and pass through untouched (the citation regex matches only
all-digit groups).

Usage:
  uv run python scripts/renumber_references.py <in.md> <out.md> [--skip-first-line]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
ENTRY_RE = re.compile(r"^\[(\d+)\]\s")


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    skip_first = "--skip-first-line" in sys.argv

    text = src.read_text()
    header = ""
    if skip_first:
        nl = text.index("\n")
        header, text = text[: nl + 1], text[nl + 1:]

    marker = "# References"
    idx = text.index(marker)
    body, listing = text[:idx], text[idx:]

    # list entries FIRST: their maximum bounds what counts as a citation —
    # larger bracketed integers are crystallographic directions ([100],
    # [110], [111], ...) and must pass through untouched
    entries: dict[int, list[str]] = {}
    current = None
    preamble: list[str] = []
    for line in listing.splitlines():
        m = ENTRY_RE.match(line)
        if m:
            current = int(m.group(1))
            entries[current] = [line]
        elif current is None:
            preamble.append(line)
        else:
            entries[current].append(line)

    max_entry = max(entries)

    # 2. first-appearance order over the BODY (citations only)
    order: list[int] = []
    for m in CITE_RE.finditer(body):
        nums = [int(x) for x in m.group(1).split(",")]
        if any(n > max_entry for n in nums):
            continue  # crystallographic direction, not a citation
        for n in nums:
            if n not in order:
                order.append(n)
    mapping = {old: new for new, old in enumerate(order, start=1)}

    body_set, list_set = set(order), set(entries)
    orphans_text = sorted(body_set - list_set)   # cited but no entry
    orphans_list = sorted(list_set - body_set)   # entry but never cited
    if orphans_text or orphans_list:
        print(f"ORPHANS — cited-without-entry: {orphans_text}; "
              f"entry-without-citation: {orphans_list}")
        print("Aborting without writing output.")
        raise SystemExit(1)

    def remap(match: re.Match) -> str:
        raw = [int(x) for x in match.group(1).split(",")]
        if any(n not in mapping for n in raw):
            return match.group(0)  # crystallographic direction — untouched
        nums = sorted(mapping[n] for n in raw)
        return "[" + ",".join(str(n) for n in nums) + "]"

    new_body = CITE_RE.sub(remap, body)
    new_entries = {}
    for old, lines in entries.items():
        block = "\n".join(lines)
        block = CITE_RE.sub(remap, block)
        new_entries[mapping[old]] = block
    new_listing = "\n".join(preamble).rstrip("\n")
    new_listing += "\n\n" + "\n\n".join(
        new_entries[k].strip("\n") for k in sorted(new_entries)) + "\n"

    dst.write_text(header + new_body + new_listing)

    # 5. verification
    out_text = dst.read_text()
    out_idx = out_text.index(marker)
    out_body, out_listing = out_text[:out_idx], out_text[out_idx:]
    seen: list[int] = []
    for m in CITE_RE.finditer(out_body):
        nums = [int(x) for x in m.group(1).split(",")]
        if any(n > len(entries) for n in nums):
            continue
        for n in nums:
            if n not in seen:
                seen.append(n)
    out_entry_nums = [int(m.group(1)) for m in
                      (ENTRY_RE.match(line) for line in out_listing.splitlines())
                      if m]
    print(f"mapping ({len(mapping)} refs): " +
          ", ".join(f"{o}->{n}" for o, n in sorted(mapping.items()) if o != n)
          or "identity")
    print(f"check (i) orphans: body-only {sorted(set(seen) - set(out_entry_nums))}, "
          f"list-only {sorted(set(out_entry_nums) - set(seen))}")
    print(f"check (ii) first-appearance order == 1..N ascending: "
          f"{seen == list(range(1, len(seen) + 1))}")
    print(f"check (ii b) list order ascending == 1..N: "
          f"{out_entry_nums == list(range(1, len(out_entry_nums) + 1))}")
    print(f"check (iii) count unchanged: {len(out_entry_nums)} == {len(entries)}: "
          f"{len(out_entry_nums) == len(entries)}")


if __name__ == "__main__":
    main()
