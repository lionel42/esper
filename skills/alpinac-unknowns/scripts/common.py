"""Shared loaders for the alpinac unknown-compound search.

A run directory is `<...>/<file>.alpinac/` containing `pipeline.nc` and
`<file>.unit_mass_chromatos.parquet`.
"""
import os
import re
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

PEAKSDB = Path("/shared/data/peaksdb")
SUBSTANCES = Path("/shared/data/substances.xlsx")
# Reference RT column (weird name). Hydrocarbon values are 7-15 s too early: see fit_rt().
RT_COL = "PeakID 240312.2257_w_conversion + [estimates from old RTs]"
# Other measured RT columns, used as extra evidence
RT_EXTRA = ["E-200 241218.0810.std.9.h5", "GSN-004, 240925.0035.tank.5", "KBZ-006"]
TOF_ROOT = Path("/media/Shuttle/TOF-Data")


# ---------------------------------------------------------------- run data
def run_files(run_dir):
    run_dir = Path(run_dir)
    name = run_dir.name.replace(".alpinac", "")
    return run_dir / "pipeline.nc", run_dir / f"{name}.unit_mass_chromatos.parquet"


def load_pipeline(run_dir):
    nc, _ = run_files(run_dir)
    ds = xr.open_dataset(nc)
    c = ds[[v for v in ds.data_vars if "compound" in ds[v].dims]].to_dataframe()
    p = ds[[v for v in ds.data_vars if "peak" in ds[v].dims]].to_dataframe()
    return c, p


def load_chrom(run_dir=None, path=None, masses=None):
    if path is None:
        _, path = run_files(run_dir)
    cols = None if masses is None else [str(m) for m in masses]
    d = pd.read_parquet(path, columns=cols)
    d = d[[k for k in d.columns if str(k).isdigit()]]
    d.columns = d.columns.astype(int)
    return d


def cmp_spec(p, ids):
    q = p[p.compound_id.isin(list(ids))]
    return q.groupby(q.mass_of_peak.round().astype(int))["prominences"].sum().to_dict()


# ---------------------------------------------------------------- references
def is_hydrocarbon(formula):
    f = str(formula).replace(" ", "")
    return bool(re.fullmatch(r"(C\d*|H\d*)+", f)) and "C" in f and "H" in f


@lru_cache(None)
def load_refs():
    sub = pd.read_excel(SUBSTANCES, sheet_name="all")
    keep = ["Compound name (Gcwerks)", "Alias 1", "Formula ", "Comment", RT_COL] + RT_EXTRA
    ref = sub[keep].copy()
    ref.columns = ["name", "alias", "formula", "comment", "rt"] + [f"rt_{i}" for i in range(len(RT_EXTRA))]
    for k in ["rt"] + [f"rt_{i}" for i in range(len(RT_EXTRA))]:
        ref[k] = pd.to_numeric(ref[k], errors="coerce")
    ref = ref.dropna(subset=["name"])
    ref["name"] = ref["name"].astype(str).str.strip()
    ref = ref[ref.rt > 0]
    ref["hc"] = ref.formula.map(is_hydrocarbon)
    return ref


def ri_refs_path():
    """RI calibrant table shipped with the alpinac package (override with ALPINAC_RI_REFS)."""
    if os.environ.get("ALPINAC_RI_REFS"):
        return Path(os.environ["ALPINAC_RI_REFS"])
    import alpinac
    return Path(alpinac.__file__).parent / "mode_extraction" / "rt_refs_v2.csv"


@lru_cache(None)
def load_ref_ri():
    """Reference RI per compound, computed exactly like alpinac does
    (alpinac.mode_extraction.read_from_excel.load_excel_compounds): the sheet RT
    is interpolated piecewise-linearly between the sheet RTs of the RI calibrants
    in rt_refs_v2.csv (Kr 25, Xe 75, ethane 200, propane 300, CFC-12 350, n-butane 400,
    CFC-11 480, n-pentane 500, CH2Cl2 543, n-hexane 600, n-heptane 700, benzene 727,
    octane 800, toluene 850, ethylbenzene 1050, mp-xylene 1150, o-xylene 1180).
    Compounds with no sheet RT (0 / NaN) get ri = NaN.
    Returns (DataFrame indexed by compound with rt, ri, formula, hc; aliases dict)."""
    from alpinac.mode_extraction.read_from_excel import load_excel_compounds
    df, aliases = load_excel_compounds(SUBSTANCES, ri_refs_path())
    df.loc[~(df.rt > 0), "ri"] = np.nan
    df["hc"] = df.formula.map(is_hydrocarbon)
    return df, aliases


def fit_rt(c, ref, span=40.0, min_anchors=3):
    """Local RT correction, separately for hydrocarbons and others.

    Anchors = confidently identified compounds (valid, score > 0.98) found in
    the reference sheet. The offset is not linear (hydrocarbons: ~+4 s at C3-C5,
    +11..15 s around C6, +6 s at C8), so for every reference compound the
    predicted RT is ref_rt + median offset of same-class anchors within
    +-span s of ref_rt (or the min_anchors nearest ones).
    Returns (ref with 'rt_pred' and 'rt_unc', anchors DataFrame).
    """
    good = c[(c.valid_candidate) & (c.bc_score > 0.98)]
    rows = []
    for i, r in good.iterrows():
        m = ref[ref.name == r.compound_name]
        if len(m):
            rows.append(dict(name=r.compound_name, cmp=i, hc=bool(m.hc.iloc[0]),
                             ref_rt=m.rt.iloc[0], rt=r.peak_rt_mean, d=r.peak_rt_mean - m.rt.iloc[0]))
    anch = pd.DataFrame(rows)
    # drop gross outliers (wrong label or wrong sheet value): > 10 s from class neighbours
    keep = []
    for _, a in anch.iterrows():
        nb = anch[(anch.hc == a.hc) & (anch["name"] != a["name"]) & ((anch.ref_rt - a.ref_rt).abs() < 80)]
        keep.append(len(nb) == 0 or abs(a.d - nb.d.median()) < 10)
    anch["used"] = keep
    ref = ref.copy()
    pred, unc = [], []
    for _, r in ref.iterrows():
        a = anch[(anch.hc == r.hc) & anch.used]
        if a.empty:
            a = anch[anch.used]
        near = a[(a.ref_rt - r.rt).abs() < span]
        if len(near) < min_anchors:
            near = a.iloc[(a.ref_rt - r.rt).abs().argsort()[:min_anchors]]
        pred.append(r.rt + near.d.median())
        unc.append(near.d.std() if len(near) > 1 else np.nan)
    ref["rt_pred"], ref["rt_unc"] = pred, unc
    return ref, anch


def _read_jdx(f):
    txt = f.read_text(errors="ignore")
    parts = txt.split("##PEAK TABLE=(XY..XY)")
    if len(parts) < 2:
        return None
    s = defaultdict(float)
    for x, y in re.findall(r"(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)", parts[1].split("##END")[0]):
        s[int(round(float(x)))] += float(y)
    return dict(s)


def _read_txt(f):
    try:
        d = pd.read_csv(f, sep="\t")
        return d.groupby(d["mass"].round().astype(int))["area"].sum().to_dict()
    except Exception:
        return None


@lru_cache(None)
def load_spectra():
    """{name: {source: {mz: intensity}}}"""
    spectra = {}
    for f in (PEAKSDB / "nist_db").glob("*.jdx"):
        if (s := _read_jdx(f)):
            spectra.setdefault(f.stem, {})["nist"] = s
    for f in (PEAKSDB / "myriam_ei").glob("*.txt"):
        if (s := _read_txt(f)):
            spectra.setdefault(f.stem, {})["mygu"] = s
    for sub in ["nist_CH", "nist_CHON", "nist_CF", "nist_CHF"]:
        for f in (PEAKSDB / sub).glob("*.jdx"):
            if (s := _read_jdx(f)):
                spectra.setdefault(f.stem.split("-", 1)[-1], {})[sub] = s
    return spectra


def cos(a, b, mz_min=25, mz_range=None):
    """Cosine on sqrt intensities. mz_range=(lo, hi) restricts to marker ions."""
    ks = [k for k in set(a) | set(b) if k >= mz_min]
    if mz_range:
        ks = [k for k in ks if mz_range[0] <= k <= mz_range[1]]
    va = np.sqrt([max(a.get(k, 0), 0) for k in ks])
    vb = np.sqrt([max(b.get(k, 0), 0) for k in ks])
    if not ks or va.sum() == 0 or vb.sum() == 0:
        return np.nan
    return float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))


def libsearch(spec, n=6, **kw):
    out = []
    for name, d in load_spectra().items():
        for src, v in d.items():
            c = cos(spec, v, **kw)
            if c == c:
                out.append((c, f"{name}[{src}]"))
    return sorted(out, reverse=True)[:n]


def ref_spectra(names):
    S = load_spectra()
    return {f"{n}[{k}]": s for n in names if isinstance(n, str) and n in S for k, s in S[n].items()}


def fmt_spec(spec, n=8):
    items = sorted(spec.items(), key=lambda kv: -kv[1])[:n]
    if not items:
        return ""
    top = items[0][1]
    return " ".join(f"{m}({v / top * 100:.0f})" for m, v in items)
