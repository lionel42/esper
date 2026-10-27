"""Step 3: is an "extra" ion a co-eluting compound or instrument-dependent fragmentation?

Tracks ion ratios at a compound's peak across many runs.
usage: python ratio_trend.py RT_APPROX REF_MZ MZ[,MZ...] [--n-air 160] [--n-std 60] [--years 2024,2025,2026] [--out trend.csv]

Interpretation: if air, std and tank measured in the same period share the
ratio and it steps with date -> instrument (fragmentation drift). If it
scatters between neighbouring air runs and not in std/tank -> co-eluting compound.
Slow on the external drive (~10 min for ~300 runs): run in background.
"""

import argparse
import glob

import common as C
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("rt", type=float)
ap.add_argument("ref_mz", type=int)
ap.add_argument("mzs")
ap.add_argument("--n-air", type=int, default=160)
ap.add_argument("--n-std", type=int, default=60)
ap.add_argument("--years", default="2024,2025,2026")
ap.add_argument(
    "--search", type=float, default=20.0, help="+- s around rt to find the ref_mz apex"
)
ap.add_argument("--out", default="trend.csv")
a = ap.parse_args()
mzs = [int(x) for x in a.mzs.split(",")]

runs = []
for y in a.years.split(","):
    runs += glob.glob(str(C.TOF_ROOT / y / "*.alpinac"))
runs = sorted(runs)
rng = np.random.default_rng(0)


def pick(k, n):
    return list(
        rng.choice(
            [r for r in runs if k in r],
            min(n, sum(k in r for r in runs)),
            replace=False,
        )
    )


sel = (
    [r for r in runs if ".tank." in r] + pick(".air.", a.n_air) + pick(".std.", a.n_std)
)

rows = []
for r in sel:
    f = glob.glob(r + "/*.unit_mass_chromatos.parquet")
    if not f:
        continue
    try:
        d = C.load_chrom(path=f[0], masses=[a.ref_mz] + mzs)
    except Exception:
        continue
    w = d[(d.index > a.rt - a.search - 10) & (d.index < a.rt + a.search + 10)]
    if w.empty:
        continue
    y = w - w.rolling(60, center=True, min_periods=1).quantile(0.1)
    i = y[a.ref_mz].values.argmax()
    t = w.index[i]
    if abs(t - a.rt) > a.search or y[a.ref_mz].iloc[i] < 300:
        continue
    s = y[(y.index > t - 1.2) & (y.index < t + 1.2)].clip(lower=0).sum()
    name = r.split("/")[-1].replace(".alpinac", "")
    row = dict(run=name, kind=name.split(".")[2], rt=t, h_ref=y[a.ref_mz].iloc[i])
    row.update({f"r{m}": s[m] / s[a.ref_mz] for m in mzs})
    rows.append(row)

if not rows:
    raise SystemExit(
        f"no run has a m/z {a.ref_mz} peak >= 300 within +-{a.search} s of {a.rt}: no trend evidence"
    )
df = pd.DataFrame(rows).sort_values("run")
df.to_csv(a.out, index=False)
df["q"] = pd.to_datetime(df.run.str[:6], format="%y%m%d").dt.to_period("Q")
pd.set_option("display.width", 200)
for m in mzs:
    print(f"\n=== r{m} = {m}/{a.ref_mz}, median by quarter and sample kind")
    print(df.groupby(["q", "kind"])[f"r{m}"].median().round(2).unstack().to_string())
air = df[df.kind == "air"].sort_values("run")
for m in mzs:
    k = f"r{m}"
    roll = air[k] - air[k].rolling(9, center=True, min_periods=3).median()
    print(
        f"{k}: air overall sd {air[k].std():.2f}, sd around rolling median {roll.std():.2f}"
    )
print(f"\nwritten {a.out}")
