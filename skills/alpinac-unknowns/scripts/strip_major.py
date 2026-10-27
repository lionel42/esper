"""Strip the tail of a big peak: subtract its scaled profile (one clean ion of it) from every ion,
then group the residual ions by apex and library-search them.

usage: python strip_major.py RUN_DIR RT_APEX PROFILE_MZ [BEFORE=14] [AFTER=40]
  e.g. toluene: RT 2199 PROFILE_MZ 92 (use an ion that is not saturated: check that its ratio to
  another ion of the big compound is flat across the peak). Writes strip_<rt>.png.
"""

import sys

import common as C
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1]
rt0 = float(sys.argv[2])
pm = int(sys.argv[3])
lo = rt0 - (float(sys.argv[4]) if len(sys.argv) > 4 else 14)
hi = rt0 + (float(sys.argv[5]) if len(sys.argv) > 5 else 40)
ch = C.load_chrom(R)
w = ch[(ch.index > lo) & (ch.index < hi)]
b = (w.iloc[:15].median() + w.iloc[-15:].median()) / 2
y = w - b
t = y.index.values
noise = w.iloc[:15].std() + 1
P = y[pm].values / y[pm].max()
core = P > 0.2
res = {}
for m in y.columns:
    if m < 20:
        continue
    v = y[m].values
    a = max(np.quantile(v[core] / P[core], 0.1), 0) if v[core].max() > 0 else 0
    res[m] = v - a * P
res = pd.DataFrame(res, index=t)
# residual peaks per ion: apex, height, SNR
rows = []
for m in res.columns:
    v = res[m].rolling(3, center=True, min_periods=1).mean().values
    i = v.argmax()
    if v[i] > 6 * noise[m] and v[i] > 40:
        rows.append((m, t[i], v[i], v[i] / noise[m]))
d = pd.DataFrame(rows, columns=["mz", "apex", "h", "snr"]).sort_values("apex")
# group by apex within 0.5 s
d["g"] = (d.apex.diff() > 0.5).cumsum()
for g, gg in d.groupby("g"):
    if len(gg) < 2 and gg.h.max() < 300:
        continue
    ap = np.average(gg.apex, weights=gg.h)
    spec = dict(zip(gg.mz, gg.h))
    mx = max(spec.values())
    top = " ".join(
        f"{m}({100 * h / mx:.0f})"
        for m, h in sorted(spec.items(), key=lambda x: -x[1])[:12]
    )
    lib = C.libsearch(spec, n=4)
    print(
        f"apex {ap:7.2f}  n={len(gg):2d} max h={mx:6.0f}  {top}\n      lib: "
        + " | ".join(f"{n} {c:.2f}" for c, n in lib)
    )
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, y[pm], label=f"profile m/z {pm}")
ax[0].set_yscale("symlog", linthresh=100)
ax[0].legend()
for m in d.sort_values("h").mz.values[-10:]:
    ax[1].plot(t, res[m], label=str(m))
ax[1].legend(ncol=5, fontsize=7)
ax[1].set_xlabel("rt (s)")
ax[1].set_title(f"residual after m/z {pm} profile subtraction")
plt.tight_layout()
plt.savefig(f"strip_{rt0:.0f}.png", dpi=110)
