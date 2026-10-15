# esper

Collects the peaks found by the alpinac extraction pipeline (`<measurement>.alpinac/pipeline.nc`)
into a SQLite database, together with the mass calibration and the unit mass chromatograms.

## Install

```
pip install -e .
```

## Usage

```
esper upload /path/to/alpinac/results /path/to/esper/data   # import / update
esper gui /path/to/esper/data                               # open the GUI
```

`upload` accepts a single `.alpinac` folder or a directory searched recursively.
Measurements whose `pipeline.nc` did not change are skipped (`--force` to reimport).

`esper gui` without argument uses `$ESPER_DATA_DIR`, or File > Open.

## GUI

- **Chromatograms** (Ctrl+1): one mass of one measurement around the selected compound,
  with the peaks found at that mass (baseline, rt, compound, height H, area A, width W).
  Click on a peak apex to select its compound. Changing file keeps the compound and the mass.
  Right click > Masses overlays other chromatograms: 2nd to 5th most prominent mass of the
  compound, all compound masses, all masses, TIC; "scale to main mass" rescales each overlay
  to the main mass in the visible window.
- **Results** (Ctrl+2): one compound (Cmp) across all measurements. Click a point to select
  that measurement/compound, double click to open its chromatogram.
- **Search** (Ctrl+3): peaks of some masses (e.g. `85, 87 101-103`) in an rt or retention
  index window, across all measurements. Color = mass, symbol = run type. "only runs with all
  masses" keeps the runs where every mass has a peak in the window. "From selection" fills the
  window and the 3 main masses from the selected compound. Click / double click as in Results.
  The retention index of a peak is the one of its compound.

A is computed by esper from the chromatogram, above a straight baseline over
±3 half widths at half maximum, limited to the prominence bases of the peak.

## Data directory

```
<data_dir>/
    esper.sqlite
    chromatograms/<measurement>.parquet
```

Tables: `measurement`, `mass_cal`, `mass_cal_rt`, `mass_cal_mass`, `compound`,
`compound_candidate`, `peak`, and the view `peak_full` (peaks joined with their
measurement, compound and best library candidate). See `src/esper/db.py`.
