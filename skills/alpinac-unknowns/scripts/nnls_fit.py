"""Step 2b: per-scan NNLS fit of library spectra on the raw unit-mass traces.

usage: python nnls_fit.py RUN_DIR RT "name[src];name[src];..." [HALF_WINDOW=5]
  src is one of nist, mygu, nist_CH, nist_CHON, nist_CF, nist_CHF
Prints each component's apex and height, the residual spectrum (+-1 s) and a
library search of the residual. A good model leaves < ~10 % under-fit
and < ~20 % absolute residual.
Writes nnls_<rt>.png in the current directory.
"""

import difflib
import sys

import matplotlib

matplotlib.use("Agg")
import common as C
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

run, rt, comps = (
    sys.argv[1],
    float(sys.argv[2]),
    [k for k in sys.argv[3].split(";") if k],
)
half = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
S = C.load_spectra()

chrom = C.load_chrom(run)
sub = chrom[(chrom.index > rt - half) & (chrom.index < rt + half)]
base = pd.concat([sub.iloc[:8], sub.iloc[-8:]]).median()
Y = (sub - base).clip(lower=0)
masses = [m for m in Y.columns if m >= 25]
Y = Y[masses]
tw = Y.index.values

A = []
for k in comps:
    name, src = k.rsplit("[", 1)
    src = src.rstrip("]")
    if src not in S.get(name, {}):
        low = {n.lower(): n for n in S}
        close = [
            low[n]
            for n in difflib.get_close_matches(name.lower(), low, n=8, cutoff=0.6)
        ]
        sys.exit(
            f"no spectrum {k}; sources for {name}: {list(S.get(name, {}))}; similar names: {close}"
        )
    spec = S[name][src]
    v = np.array([spec.get(m, 0.0) for m in masses], float)
    A.append(v / v.max())
A = np.array(A).T
coef = np.array([nnls(A, row)[0] for row in Y.values])
res = Y.values - coef @ A.T

print(f"##### rt {rt}")
for j, k in enumerate(comps):
    i = coef[:, j].argmax()
    print(f"  {k:<45} apex {tw[i]:.2f}  height {coef[i, j]:.0f}")
core = (tw > rt - 1) & (tw < rt + 1)
rsum = res[core].clip(min=0).sum(axis=0)
total = Y.values[core].sum()
# under-fit (signal the model misses) and |residual| (also counts over-fit)
print(
    f"  residual fraction in +-1 s: under {rsum.sum() / total:.2f}  abs {np.abs(res[core]).sum() / total:.2f}"
)
spec = {masses[i]: rsum[i] for i in range(len(masses)) if rsum[i] > 0.02 * rsum.max()}
print("  residual:", C.fmt_spec(spec, 15))
print("  residual lib:", " | ".join(f"{n} {c:.2f}" for c, n in C.libsearch(spec, 8)))
for i in np.argsort(-rsum)[:8]:
    k = int(np.argmax(np.where(core, res[:, i], -np.inf)))
    print(f"     m/z {masses[i]:>3}: residual apex {tw[k]:.2f}  h {res[k, i]:.0f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for j, k in enumerate(comps):
    ax[0].plot(tw, coef[:, j], label=k)
ax[0].legend(fontsize=7)
for i in np.argsort(-rsum)[:8]:
    ax[1].plot(tw, res[:, i], label=str(masses[i]))
ax[1].legend(ncol=4, fontsize=7)
fig.savefig(f"nnls_{rt:.1f}.png", dpi=80)
