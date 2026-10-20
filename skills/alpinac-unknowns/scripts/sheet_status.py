"""Is a compound in the Excel sheet, does it have a sheet RT/RI, and does it have a spectrum?

usage: python sheet_status.py "name1;name2;..."   (names, aliases or chemical names; case-insensitive)
       python sheet_status.py --near RI [RI ...]   (all sheet entries within +-10 RI, incl. Mr_/Ms_/CxHy-a placeholders)

For each name prints: sheet name, matched by (name / alias / chemical name), sheet RT, reference RI,
formula, comment, and which peaksdb sources hold a spectrum under the sheet name or any alias.
Used to decide the category of a finding (see SKILL.md step 4):
  NOT IN SHEET  -> NEW compound for the sheet
  in sheet, no RT or no spectrum -> KNOWN, pipeline cannot name it (library/sheet gap)
  in sheet with RT and spectrum  -> KNOWN, pipeline mislabelled or hid it
"""
import re
import sys

import pandas as pd

import common as C


def norm(s):
    return re.sub(r"[\s,\-()]", "", str(s)).lower()


sub = pd.read_excel(C.SUBSTANCES, sheet_name="all")
sub = sub.dropna(subset=["Compound name (Gcwerks)"])
ri, aliases = C.load_ref_ri()
S = C.load_spectra()
Snorm = {norm(k): k for k in S}
alias_of = {}
for al, comp in aliases.items():
    alias_of.setdefault(comp, []).append(al)

if len(sys.argv) < 2:
    sys.exit(__doc__)

if sys.argv[1] == "--near":
    for x in map(float, sys.argv[2:]):
        d = ri[(ri.ri - x).abs() <= 10].copy()
        d["dRI"] = x - d.ri
        print(f"--- sheet entries within +-10 RI of {x:.0f} (dRI = sample - sheet)")
        for name, r in d.sort_values("dRI", key=abs).iterrows():
            # a spectrum stored under an alias counts too
            spec = "spec" if any(norm(n) in Snorm for n in [name] + alias_of.get(name, [])) else "nospec"
            print(f"   {name:28s} RI {r.ri:6.1f}  dRI {r.dRI:+5.1f}  {str(r.formula):14s} {spec}")
    sys.exit()

for q in [k.strip() for k in sys.argv[1].split(";") if k.strip()]:
    nq = norm(q)
    hits = []
    for _, r in sub.iterrows():
        for how, col in [("name", "Compound name (Gcwerks)"), ("alias", "Alias 1"),
                         ("alias", "Alias 2"), ("chemical name", "Chemical name")]:
            v = r.get(col)
            if isinstance(v, str) and norm(v) == nq:
                hits.append((how, r))
                break
    if not hits:
        spec = Snorm.get(nq)
        print(f"{q:28s} NOT IN SHEET   spectrum: {list(S[spec]) if spec else 'none'}")
        continue
    for how, r in hits:
        name = str(r["Compound name (Gcwerks)"]).strip()
        rt = pd.to_numeric(r[C.RT_COL], errors="coerce")
        r_ri = ri["ri"].get(name, float("nan")) if name in ri.index else float("nan")
        names = [name] + [str(r[a]) for a in ("Alias 1", "Alias 2", "Chemical name") if isinstance(r.get(a), str)]
        srcs = sorted({s for n in names if norm(n) in Snorm for s in S[Snorm[norm(n)]]})
        has_rt = rt > 0
        print(f"{q:28s} in sheet as '{name}' (by {how})  RT {rt if has_rt else 'none'}  "
              f"RI {r_ri:.0f}  formula {r['Formula ']}  spectrum: {srcs or 'none'}"
              f"  comment: {str(r['Comment'])[:60] if isinstance(r['Comment'], str) else ''}")
