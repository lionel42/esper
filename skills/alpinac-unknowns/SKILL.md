---
name: alpinac-unknowns
description: Find compounds that the alpinac pipeline missed or misidentified in one TOF run (by default the newest not-yet-analysed air run) (pipeline.nc + unit_mass_chromatos.parquet), using the reference RTs in /shared/data/substances.xlsx, the spectra in /shared/data/peaksdb, and NNLS deconvolution of the raw traces. Use when asked to check a run / pipeline.nc for unidentified, misidentified or interesting compounds.
---

# Unknown-compound search in an alpinac run

Goal: from the N largest compounds (default 200, by `prominences_sum`) of one run,
report the ones that are **real and not (correctly) identified**, not noise and not
fragments of an identified compound. Give each a conclusion, a confidence and the evidence.

Inputs: a run directory `/media/Shuttle/TOF-Data/<year>/<file>.alpinac/` with
`pipeline.nc` and `<file>.unit_mass_chromatos.parquet`. New runs arrive every day
(~16 per day, `<yymmdd>.<hhmm>.<type>.<n>`, types air / std / tank / blank / …).

### 0. Choose the run and load the findings log (always)
- **Findings log** (Claude's own memory for this skill, outside the skill):
  `~/.local/share/alpinac/unknowns/findings.md` (curated open issues, library/sheet issues,
  settled questions, unexplained peaks) and `~/.local/share/alpinac/unknowns/runs.csv`
  (one line per analysed run). Read both before starting. Do not re-investigate what is listed
  under "Settled"; check the "Open issues" RTs first.
- **No run given** → do not ask. Take the newest **air** run that has both `pipeline.nc` and the
  parquet and is not yet in `runs.csv`:
  ```
  for d in $(ls -d /media/Shuttle/TOF-Data/$(date +%Y)/*.air.*.alpinac | sort -r); do
    [ -f $d/pipeline.nc ] && ls $d/*.unit_mass_chromatos.parquet >/dev/null 2>&1 && \
    ! grep -q "^$(basename $d .alpinac)," ~/.local/share/alpinac/unknowns/runs.csv && { echo $d; break; }
  done
  ```
  (In January also look in the previous year's folder.) If a newer run is still being processed
  (no pipeline.nc yet), skip it. Say which run you picked at the top of the reply.
  If the user names a date or "today", take the newest air run of that day.
- A std/tank run is only analysed when explicitly asked (or to check a ratio in step 3).

Scripts are in `~/.claude/skills/alpinac-unknowns/scripts/` (python at `/shared/pyenv/bin/python`).
Run them **from the scratchpad directory** (they write png/csv to the cwd), passing the
run directory as first argument. `common.py` holds the loaders.

## Procedure

### 1. Triage (always)
```
python $S/triage.py RUN_DIR --top 200 --only-flagged --out triage.csv > triage.txt
```
- **Work in retention index (RI), not seconds.** RI is alpinac's modified Kováts index:
  RT interpolated piecewise-linearly between calibrants with fixed RI
  (Kr 25, Xe 75, ethane 200, propane 300, CFC-12 350, n-butane 400, CFC-11 480, n-pentane 500,
  CH2Cl2 543, n-hexane 600, n-heptane 700, benzene 727, octane 800, toluene 850,
  ethylbenzene 1050, mp-xylene 1150, o-xylene 1180; file
  `~/gitlabs/halosearch/alpinac/mode_extraction/rt_refs_v2.csv`). The sample RI is
  `retention_index` in pipeline.nc; the reference RI is the sheet RT
  (`PeakID 240312.2257_w_conversion …` column) interpolated between the sheet RTs of the same
  calibrants (`C.load_ref_ri()`, identical to alpinac's `load_excel_compounds`). This absorbs RT
  drift and the non-linear offsets between the sheet and the current method.
  alpinac marks a candidate valid if |ΔRI| < 10 and score > 0.97.
- The header prints the calibrants found in the run and the ΔRI spread of confident ids
  (261019.0109.air.1: median 0.0, 90 % within ±4.5). If a calibrant is missing or is the wrong
  peak, RI is distorted between its neighbours: check this first, fix before trusting ΔRI.
- For each flagged compound: label, scores, spectrum, flags, and sheet references within ±8 RI
  with `cos alone / cos ±1 s cluster` (`=cmpN` means that reference is already assigned).
- Flags: `invalid`, `lowscore`, `few_ions`, `label_dRI=` (label is > 10 RI from its reference,
  i.e. likely wrong label), `pfc_label_without_markers`, `CH3Br?`/`Br?` (1:1 isotope pairs).
- Sheet compounds without an RT (e.g. methylcyclopentane, 2,3-dimethylbutane) have no RI and
  never appear as refs: rely on the spectrum and say the RI could not be checked.
Read the whole triage.txt. Sort compounds into: (a) background/ghost (N2, O2, Ar, CO2, H2O, NO,
single ion, Ar ghost at m/z 40), (b) fragment of a neighbouring identified compound,
(c) right name but flagged invalid, (d) candidate missed/misnamed → step 2.

### 2. Resolve candidates on the raw traces
- `python $S/deconv.py RUN_DIR RT [RT...]` groups ions by apex time (0.25 s) and library-searches
  each group. Use it to see how many components are under a peak.
- `python $S/nnls_fit.py RUN_DIR RT "name[src];name[src];..." [half]` fits library spectra
  scan by scan. Include every identified neighbour within ±5 s plus your hypotheses.
  A good model leaves **< ~10 %** residual and each component has a sensible apex.
  Library-search the residual: that is how hidden compounds show up
  (e.g. 2,2-dimethylbutane under CFC-113, methylcyclopentane next to 3-methylpentane).
  Sources: `nist`, `mygu` (myriam_ei, own instrument), `nist_CH`, `nist_CHON`, `nist_CF`, `nist_CHF`.
- For co-elution, score on **marker ions** only: `C.cos(spec, ref, mz_range=(78, 97))` for
  CH3Br; 119/131/169/219 for perfluoroalkanes; M⁺ and isotope ratios (Cl 3:1, Cl2 9:6:1, Br 1:1).
- Sum adjacent split compounds (apex within ~0.5 s) before matching: `C.cmp_spec(p, [i, j])`.
- For candidates without a peaksdb spectrum (sheet names like `C8H18-a`, `Mr_*`, `Ms_*`),
  accept on RT + formula consistency (M⁺, class) and say so.

### 3. Before claiming a new co-eluting compound from "extra" ions
Check whether the ions are fragments of the identified compound whose ratio drifts with the
instrument state:
```
python $S/ratio_trend.py RT REF_MZ MZ1,MZ2 --out trend.csv     # ~10 min, run in background
```
If air, std and tank of the same period share the ratio and it steps by date → instrument
fragmentation, not a compound. Only a ratio that scatters between neighbouring air runs (and
not in std/tank) supports a co-eluter. Never decide from a single standard measured at another date.

### 4. Classify every finding against the Excel sheet (always, before the report)
The user's question is: *is this a known compound the pipeline got wrong, or something new
that is not in the sheet?* Answer it explicitly for every finding:
```
python $S/sheet_status.py "name1;name2;..."    # in sheet? by name/alias/chemical name; RT, RI, spectrum, comment
python $S/sheet_status.py --near RI [RI ...]   # every sheet entry (incl. Mr_/Ms_/CxHy-a placeholders) within ±10 RI
```
Use exactly one of these categories (label them by name in the report):
- **MISLABELLED**: the compound is in the sheet with an RT and a spectrum. The pipeline gave
  this peak (or its largest part) another name. Fix: matching/RI.
- **HIDDEN**: the compound is in the sheet with an RT and a spectrum. It co-elutes under a peak that the
  pipeline named (correctly) after another compound, so the pipeline reports only one. Fix: deconvolution.
- **IN SHEET, CANNOT BE NAMED**: in the sheet but without an RT or without a spectrum
  (incl. placeholders like `C7H16-a`, `CxFy-a`, `Mr_*`/`Ms_*` with a compatible formula ±10 RI).
  The pipeline can never validate it, so a mislabel or "invalid" is unavoidable. Fix: add RT /
  spectrum. Give the proposed identity of a placeholder if the spectrum supports one.
- **NEW (not in sheet)**: no sheet entry by name, alias or chemical name, **and** `--near` shows no
  entry or placeholder with a compatible formula within ±10 RI. Only this category means
  "add a new compound to the sheet". Say how sure the identity is (exact compound vs class only).
Confidence is reported twice: in the category (is it really absent from / present in the sheet) and in
the identity (is it really that compound). Never call something new from a name lookup alone:
isomer names (2,4-dimethylpentane) are often in the sheet as placeholders (C7H16-a).

### 5. Report
Main table, grouped by the four categories above (one sub-table or a `category` column):
**cmp first** (the `compound` index in pipeline.nc, plus merged pieces in brackets; for HIDDEN
compounds, the cmp of the peak they sit under), then rt, pipeline label, what it really is,
identity confidence (high / medium / low), evidence (ΔRI to the sheet, cos, NNLS residual, marker ions),
sheet entry used. **Sort each table by cmp** (ascending). Use the same cmp-first, cmp-sorted
format in the reply to the user and in the saved report. Then separate short lists (also sorted by cmp):
"right name but flagged invalid", "not real (noise / fragments)", "not resolvable".
Finish with library/sheet issues found (missing spectra, wrong sheet RTs, labels the pipeline
uses wrongly). Keep scratch outputs in the scratchpad and give full paths.
Start the report with what changed versus the findings log: new issues, open issues seen again,
open issues no longer present (fixed in the pipeline?).

### 6. Update the findings log (always, last step)
- `findings.md`: for each open issue seen again, update `last seen` and `n runs` (and RT/ΔRI if
  it moved); add new issues; move an issue to "Settled" when it has been explained, or note
  "not seen in <run>" when it disappears (remove after ~5 consecutive absences). Add library/sheet
  issues and unexplained real peaks. Edit in place, keep it short; no per-run narrative.
- `runs.csv`: append `run,analysed_on,top,n_flagged,n_new_findings,report` (report = path of a
  saved copy, if any). Copy the final report to `~/.local/share/alpinac/unknowns/reports/<run>.md`.
- General lessons about the *method* (a new pitfall, a script bug) go into this SKILL.md
  "Known pitfalls", not into the log. Compound findings go into the log, not the skill.

## Known pitfalls (method)
- Do not compare sheet RTs to sample RTs in seconds: the offset is not constant
  (hydrocarbons +4 s at C3–C5, +11..15 s at C6, +6 s at C8 in Oct 2026). Use ΔRI.
  `C.fit_rt()` (local offset in seconds) is only a fallback when RI calibrants are missing.
- Other sheet columns (`E-200 …`, `GSN-004 …`, `KBZ-006`) are recent measured RTs in seconds:
  useful as a second opinion, but convert them to RI with the calibrants of that same run.
- Compound-specific findings (missed compounds, ΔRI values, settled questions) live in the
  findings log, not here.
- PFC labels on hydrocarbon spectra that only share m/z 69 are common. Need 119/131/219 for a PFC.
- Isomers / homologues with near-identical spectra (HFO-1234yf vs ze(E), C4F10/C5F12/C6F14):
  decide by RI, not by cos.
- Fragment ratios drift with instrument state (e.g. HCFC-141b 80/81): never claim a co-eluter
  from extra ions without step 3.
- `pipeline.nc` lists candidates from the library actually used by the pipeline; a compound
  never appearing as a candidate in any of the ~500 compounds is probably missing from it, even
  if peaksdb has the spectrum (config: `~/.config/alpinac/mode_extraction_config.json`).
- In `ratio_trend.py` output, also check each ratio against `h_ref`. A real fragment keeps the
  ratio constant whatever the amount. A ratio that falls roughly as 1/h_ref is an independent compound
  (or background), even when air, std and tank share the quarterly medians.
- Tails of very large peaks (toluene, CFC-12, n-butane, acetone, …) hide small co-eluters and
  their marker ions: the baseline-subtracted spectrum then shows "missing" ions that are really there.
  Before rejecting a candidate for a missing marker ion within ~10 s after a big peak, run
  `python $S/strip_major.py RUN_DIR RT_APEX PROFILE_MZ` (e.g. toluene 2199 92). It subtracts the big peak's own
  profile, so ions of the big compound itself are only reliable down to ~1 % of its height; other ions
  keep the normal detection limit. (Example: chlorobenzene's m/z 77 only appeared after stripping toluene.)
- In pandas `iterrows`, `row.name`, `row.first`, `row.flags`, `row.dt` are attributes/methods,
  not columns: use `row["..."]`.
