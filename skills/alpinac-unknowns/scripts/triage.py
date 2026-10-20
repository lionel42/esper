"""Step 1: triage the N largest compounds of a run.

usage: python triage.py RUN_DIR [--top 200] [--ri-win 8] [--out triage.csv]

Matching is done in retention index (RI), like alpinac: sample RI comes from
pipeline.nc (RT interpolated between the RI calibrants found in the run),
reference RI from the sheet RT interpolated between the sheet RTs of the same
calibrants (common.load_ref_ri). dRI = sample - reference; alpinac accepts
|dRI| < 10.

For each compound prints: label, scores, validity, spectrum, flags, and the
reference compounds whose RI is within --ri-win, with spectral cosine to the
compound alone and to the +-1 s cluster.
"""
import argparse

import numpy as np
import pandas as pd

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--top", type=int, default=200)
ap.add_argument("--ri-win", type=float, default=8.0)
ap.add_argument("--out", default=None)
ap.add_argument("--only-flagged", action="store_true")
args = ap.parse_args()

c, p = C.load_pipeline(args.run)
ref, aliases = C.load_ref_ri()
ref = ref.dropna(subset=["ri"])
alias_of = {}
for al, comp in aliases.items():
    alias_of.setdefault(comp, []).append(al)
cal = c[c.is_ref_ri.astype(bool)].sort_values("peak_rt_mean")
print(f"RI calibrants found in run ({len(cal)}): " + " ".join(f"{r.compound_name}@{r.peak_rt_mean:.0f}={r.retention_index:.0f}" for r in cal.itertuples()))
print("  -> check that each calibrant is the right peak; a wrong calibrant distorts the RI locally.")
# how well does RI work in this run: dRI of confident ids
conf = c[(c.valid_candidate) & (c.bc_score > 0.98) & ~c.is_ref_ri.astype(bool)]
d = [r.retention_index - ref.loc[r.compound_name, "ri"] for r in conf.itertuples() if r.compound_name in ref.index]
d = np.array(d)
print(f"dRI of {len(d)} confident ids: median {np.median(d):+.1f}, 90% within +-{np.percentile(np.abs(d), 90):.1f}")

NOISE = {"N2", "O2", "Ar+O2", "Ar+N2", "N2+O2", "CO2-A", "CO2-B", "H2O", "NO-A", "NO-B", "Kr", "Xe-A"}
PFC_MARKERS = {119, 131, 169, 219}

top = c.sort_values("prominences_sum", ascending=False).head(args.top)
rows = []
for i, r in top.iterrows():
    s1 = C.cmp_spec(p, [i])
    clus = c[(c.peak_rt_mean - r.peak_rt_mean).abs() < 1.0].index.tolist()
    s2 = C.cmp_spec(p, clus)
    flags = []
    if not r.valid_candidate:
        flags.append("invalid")
    if r.bc_score < 0.9:
        flags.append("lowscore")
    if r.peak_rt_count <= 2:
        flags.append("few_ions")
    if r.compound_name in NOISE:
        flags.append("bg")
    # RT consistency of the label
    if r.compound_name in ref.index and not r.is_ref_ri:
        dri = r.retention_index - ref.loc[r.compound_name, "ri"]
        if abs(dri) > 10:
            flags.append(f"label_dRI={dri:+.0f}")
    # perfluoro labels without perfluoro marker ions
    if any(k in str(r.compound_name) for k in ("C6F14", "C5F12", "C4F10", "CF4", "NOVEC", "H-1301")) and r.compound_name != "CF4":
        if not (PFC_MARKERS & set(k for k, v in s1.items() if v > 0.05 * max(s1.values()))):
            flags.append("pfc_label_without_markers")
    # Br / Cl isotope hints not explained by label
    for a_, b_, tag in [(94, 96, "CH3Br?"), (79, 81, "Br?")]:
        if s1.get(a_, 0) > 0.2 * max(s1.values()) and 0.7 < s1.get(b_, 0) / max(s1.get(a_, 1), 1) < 1.3 and "Br" not in str(r.compound_name):
            flags.append(tag)
    near = ref[(ref.ri - r.retention_index).abs() < args.ri_win].copy()
    near["dri"] = r.retention_index - near.ri
    cands = []
    for name, nr in near.sort_values("dri", key=abs).iterrows():
        rs = C.ref_spectra([name] + alias_of.get(name, []))
        assigned = c[(c.compound_name == name) & ((c.retention_index - nr.ri).abs() < 10)]
        tag = f"=cmp{assigned.index[0]}" if len(assigned) else ""
        if rs:
            k, v = max(rs.items(), key=lambda kv: np.nan_to_num(C.cos(s1, kv[1])))
            cands.append(f"{name}{tag} dRI{nr['dri']:+.1f} cos{C.cos(s1, v):.2f}/{C.cos(s2, v):.2f}")
        else:
            cands.append(f"{name}{tag} dRI{nr['dri']:+.1f} nospec")
    rows.append(dict(cmp=i, rt=round(r.peak_rt_mean, 2), ri=round(r.retention_index, 0), prom=r.prominences_sum,
                     label=r.compound_name, score=round(r.bc_score, 3), valid=bool(r.valid_candidate),
                     first=f"{r.first_candidate}({r.first_score:.2f})", second=f"{r.second_candidate}({r.second_score:.2f})",
                     ms=C.fmt_spec(s1), cluster=",".join(map(str, clus)), flag=" ".join(flags), near_refs=" | ".join(cands)))

df = pd.DataFrame(rows)
if args.out:
    df.to_csv(args.out, index=False)
if args.only_flagged:
    # background gases (N2, O2, Ar, CO2, H2O, NO, Kr, Xe) are hidden unless they carry another flag
    show = df[df["flag"].str.replace(r"\bbg\b|label_dRI=\S+", "", regex=True).str.strip() != ""]
    show = show[~(show["flag"].str.contains(r"\bbg\b") & ~show["flag"].str.contains("invalid|lowscore|Br"))]
else:
    show = df
print(f"\n{len(show)} of {len(df)} compounds shown")
for r in show.to_dict("records"):
    print(f"\n[{r['cmp']}] rt {r['rt']} RI {r['ri']:.0f} prom {r['prom']:.3g}  label={r['label']} ({r['score']}, valid={r['valid']})  {r['flag']}")
    print(f"   cand: {r['first']} {r['second']}   cluster+-1s: {r['cluster']}")
    print(f"   ms:   {r['ms']}")
    if r["near_refs"]:
        print(f"   refs: {r['near_refs']}")
