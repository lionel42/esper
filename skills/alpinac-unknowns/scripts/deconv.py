"""Step 2a: group raw unit-mass ions by apex time around one or more RTs and library-search each group.

usage: python deconv.py RUN_DIR RT [RT ...]
Writes deconv_<rt>.png in the current directory.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

run = sys.argv[1]
chrom = C.load_chrom(run)

for rt in map(float, sys.argv[2:]):
    sub = chrom[(chrom.index > rt - 8) & (chrom.index < rt + 8)]
    tw = sub.index.values
    base = (sub.iloc[:12].median() + sub.iloc[-12:].median()) / 2
    noise = (sub.iloc[:12].std() + sub.iloc[-12:].std()) / 2 + 1
    y = sub - base
    core = (tw > rt - 1) & (tw < rt + 1)
    rows = []
    for m in y.columns:
        if m < 12:
            continue
        yy = y[m].values
        i = int(np.argmax(np.where(core, yy, -np.inf)))
        h = yy[i]
        if h < 6 * noise[m] or h < 30:
            continue
        ta = tw[i]
        # parabolic apex only at a real local maximum; on a neighbour's flank keep the sample time
        if 0 < i < len(yy) - 1 and yy[i - 1] <= yy[i] >= yy[i + 1]:
            a, b, c_ = yy[i - 1], yy[i], yy[i + 1]
            den = a - 2 * b + c_
            if den:
                ta += 0.5 * (a - c_) / den * (tw[1] - tw[0])
        half = yy >= h / 2
        j0 = j1 = i
        while j0 > 0 and half[j0 - 1]:
            j0 -= 1
        while j1 < len(yy) - 1 and half[j1 + 1]:
            j1 += 1
        rows.append((m, h, ta, tw[j1] - tw[j0]))
    print(f"\n##### rt {rt}")
    if not rows:
        print("  nothing above noise")
        continue
    df = pd.DataFrame(rows, columns=["mz", "h", "apex", "fwhm"]).sort_values("apex")
    gid, last, grp = -1, None, []
    for a in df.apex:
        if last is None or a - last > 0.25:
            gid += 1
        grp.append(gid)
        last = a
    df["grp"] = grp
    for _, d in df.groupby("grp"):
        if d.h.max() < 0.03 * df.h.max() and len(d) < 3:
            continue
        spec = dict(zip(d.mz, d.h))
        print(f"  group apex {d.apex.median():.2f}  fwhm {d.fwhm.median():.2f}s  n={len(d)}  max h={d.h.max():.0f}")
        print(f"     {C.fmt_spec(spec, 20)}")
        print("     lib:", " | ".join(f"{n} {c:.2f}" for c, n in C.libsearch(spec)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in df.sort_values("h", ascending=False).mz[:10]:
        ax.plot(tw, y[m] / df.h.max(), label=str(m))
    ax.axvline(rt, color="k", ls=":")
    ax.set_xlim(rt - 4, rt + 4)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(f"deconv_{rt:.1f}.png", dpi=80)
    plt.close(fig)
