"""Verify every result macro defined in gb_results.tex is used in main.tex and
carries a real (non-placeholder) value.

Reads from the same directory emit_latex_macros.py writes to: latex/ in the
repository root, or GUARDBOX_LATEX_DIR if set (main.tex must be there too)."""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLEAF = os.environ.get("GUARDBOX_LATEX_DIR", os.path.join(REPO_ROOT, "latex"))

defined = {}
with open(f"{OVERLEAF}/gb_results.tex") as fh:
    for line in fh:
        for name, val in re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}", line):
            defined[name] = val

with open(f"{OVERLEAF}/main.tex") as fh:
    tex = fh.read()

placeholders = [n for n, v in defined.items() if v.strip() in ("X", "--", "")]
unused = [n for n in defined if ("\\" + n) not in tex]

print(f"defined result macros: {len(defined)}")
if placeholders:
    print(f"PLACEHOLDER values remain: {placeholders}")
else:
    print("no placeholder values: OK")
if unused:
    print(f"defined-but-unused (ok if intentional): {unused}")
