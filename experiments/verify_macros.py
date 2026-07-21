"""Verify every result macro defined in gb_results.tex is used in main.tex and
carries a real (non-placeholder) value."""

import re

OVERLEAF = "/Users/sinan/Downloads/GuardBox_Overleaf"

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
