# esper

Collects the peaks found by the alpinac extraction pipeline (`<measurement>.alpinac/pipeline.nc`)
into a SQLite database, together with the mass calibration and the unit mass chromatograms.

## Install

```
pip install -e .
```

## Usage

```
esper upload /path/to/alpinac/results /path/to/esper/data --instrument tofwerk  # first upload
esper upload /path/to/alpinac/results /path/to/esper/data   # import / update, and fit the peaks
esper fit /path/to/esper/data                               # fit the measurements not fitted yet
esper targets /shared/data/substances.xlsx /path/to/esper/data  # import the target compounds
esper gui /path/to/esper/data                               # open the GUI
```

A data directory holds the runs of one instrument, declared in its `esper.yaml`
(`instrument: tofwerk` or `gcwerks`, see [Data sources](#data-sources)): give it with
`--instrument` to the first upload, which writes it there. `upload --no-fit` imports without fitting; `-j N` (upload and fit) uses N processes for the
fit (default: all CPUs, ~1 min per measurement with 10).

### Data sources

Each instrument is a module of `src/esper/sources/` that finds the runs of a source path and
reads them (`find_runs`, `read`, `refresh`; see `esper/sources/__init__.py`); `esper.upload`
stores what they read. The measurement table keeps the `instrument` of each run and where its
peaks come from (`peak_source`); only the peaks of alpinac are fitted.

| instrument | runs | peaks |
|---|---|---|
| `tofwerk` | `<run>.alpinac` folders: the alpinac extraction, the chromatograms of `esper chromatograms`, the mass calibration and the h5 file next to the folder | `alpinac`, fitted |
| `gcwerks` | `<yy>/chromatograms/channel0/<run>` of a GCwerks directory | none (chromatograms only) |

Data directories made before the instrument was declared are `tofwerk`: their first upload
declares it. A data directory refuses runs of another instrument.

#### GCwerks chromatograms

The chromatograms of a GCwerks directory (e.g. a Medusa GC-MS), read with
[gcwerks](https://github.com/lionel42/gcwerks) (`src/esper/sources/gcwerks.py`):

```
esper upload /path/to/gcwerks/jungfraujoch-medusa /path/to/esper/data --instrument gcwerks  # all the years
esper upload /path/to/gcwerks/jungfraujoch-medusa/26 /path/to/esper/data   # one year
```

The source can be the GCwerks directory, a year, its `chromatograms` folder or one file. The
files `<yy>/chromatograms/channel0/<name>` (not the `tune` runs) are written as
`chromatograms/<name>.parquet` (index rt, one column per unit mass and `tic`), the start of the
run from the file (`acq_start_source = 'gcwerks'`, to the s). Only the chromatograms: no peaks,
no fit (`peak_source` NULL). The quadrupole measures each mass only in its time windows: the
column of a mass is NaN outside them (the GUI draws the measured scans only). The signal is as
GCwerks stores it, baseline included. Unchanged files (same size and mtime) are skipped. The
files written before ~April 2012 (an older layout) cannot be read yet.

### Peak fit

After the import, the peaks of each mass are fitted in the unit mass chromatograms with
[peaxart](https://github.com/lionel42/peaxart) (`src/esper/fitting.py`): Pap-Papai peaks
on a constant baseline, starting at the rts of the extraction, by groups of overlapping
peaks. peaxart adds a peak where the data show a coelution. The fit is stored in the peak
table (`fit_rt_s`, `fit_height`, `fit_sigma_s`, `fit_tailing`, `fit_baseline`, `fit_area`,
their uncertainties `*_u*`, `fit_sn`; NULL when the fit rejected the peak).

Each fitted peak goes to the nearest extraction peak. When several are within its half
maximum crossings, the extraction peak is split: its row takes the nearest piece, new rows
the others, all with `source = 'fit'` (else `'extraction'`), their rt and height from the
fit, and no ips. A fitted peak outside them (a peak the extraction missed) is a new row too.
These peaks are given to the compound that matches them best: rt within 0.3 peak sigma
(widened by 3 rt uncertainties) and width within a factor 2 of the compound's (weighted
medians of its peaks fitted as extracted); a peak matching no compound gets none
(`compound_idx` NULL). The thresholds are in `fitting.py`, from how well the peaks of a
compound agree in 251104.1724.air.1 (95 % within 0.28 sigma). Example: the single peak of
2- and 3-methylhexane on m/z 43 is split in two, one for each.

A measurement is fitted once (`measurement.fit_version`): to fit it again, import it again
(`esper upload --force`).

### Mass calibration

`esper calibrate` fits the mass calibration of h5 files (`src/esper/calibrate.py`): the spectra
are read with [twutils](https://github.com/lionel42/twutils) (only the scans of each rt window and
the tof indices around the calibrants) and the calibration is fitted with [masscal](../masscal)
(peaks with peaxart), following its drift with the retention time. It writes
`<run>.alpinac/<run>.seg0_mc.json` (where the extraction and esper find it) and the calibrant
peaks (`.peaks.csv`).

```
esper -v calibrate run.h5                                   # default calibrants: air, F, Re, PFTBA
esper calibrate "/media/Shuttle/TOF-Data/2024/2403*.air.*.h5" -c /shared/scripts/tof-processing/config_calib.yml
esper calibrate run.h5 --calibrant CF3 C2F5 C3F5 N2 O2 Ar --spike 1200 C4F9 --averaging 240 --spacing 60
esper calibrate run.h5 --plots                              # + <run>.seg0_mc.<plot>.png (matplotlib)
esper plot-calibration run.alpinac/run.seg0_mc.json         # or the h5 file; --plot drift residuals --show
```

In Python: `esper.calibrate.calibrate_run(h5, calibrants, ...)` returns a
`masscal.CalibrationResult`; `calibrate_to_file` also writes it. Calibrant formulas (not the
default ones) need the alpinac of gitlabs/alpinac (`PYTHONPATH=../alpinac/src` in `/shared/pyenv`).

The mass calibration of each run is read with masscal, found as twutils does
(`src/esper/mass_cal.py`): `<run>.seg0_mc.json` of `esper calibrate` if there is one, else
the legacy `<run>.EI_mc.txt`. Each calibration point (rt window) is stored in `mass_cal_rt` with
its absolute `time` = start of the acquisition + rt, UTC: the start is read from the h5 file next
to the `.alpinac` folder (`TimingData/AcquisitionTimeZero`, to the ms; `measurement.acq_start`),
else taken from the run name (to the minute, `acq_start_source = 'name'`). The files of masscal
also give, per window, the calibrants found and used and the residuals (`rms_ppm`, `max_ppm`),
and the windows whose fit was rejected (`ok = 0`). The calibrant peaks (`mass_cal_peak`: tof,
residual, used, per calibrant and window) come from `<run>.seg0_mc.peaks.csv` (written by
`esper calibrate`) or the legacy `<run>.calibration_peaks.csv`.

Without a calibration file, nothing of the calibration is stored. `upload` also reads again the
calibration of the measurements it skips as unchanged (e.g. after `esper calibrate`, or for a
database imported by an older esper), without importing or fitting them again.

### Unit mass chromatograms

`esper chromatograms` makes the unit mass chromatograms of h5 files (`src/esper/chromatograms.py`):
the spectra are read with twutils, summed around each unit mass and their baseline removed with
peaxart (`peaxart.unit_mass_sums`, `peaxart.remove_baseline`), with the mass calibration of the run
(`esper calibrate`, else the legacy `*_mc.txt`).

```
esper chromatograms 241007.0423.air.1.h5        # -> 241007.0423.air.1.alpinac/241007.0423.air.1.unit_mass_chromatos.parquet
esper chromatograms run.h5 --rt-stop 196.6      # until the filament failed
esper chromatograms run.h5 --legacy --output-dir v2/   # as the legacy pipeline (unit_mass_chromatos)
esper chromatograms run.h5 --no-steps           # one baseline over the whole run
```

The run is read in blocks
of whole writes (~600 scans), the spectra summed in `m - 0.5 < m/z < m + 0.5`
for each unit mass (default masses: those of the spectra, 2 to 300), then each
chromatogram smoothed and its arPLS baseline removed (peaxart, see its README).
**Steps of the acquisition**: the acquisition switches the filament emission
target (0.5 mA -> 0.05 at ~15 s, 0.5 at ~205 s, 0.05 at ~385 s, 0.5 at ~1105 s:
the background of all masses x10) and the resonant ejection of the quadrupole
(`QUADA_REXAMPL1-4`, at 25.9 / 30.5 / 43 / 83.91 Th: N2 / O2 (m/z 28-34) from
~191 s, CO2 (42-46) during ~1181-1320 s, while the air peaks elute). A smooth
baseline over the whole run lags behind such steps for 30-100 s (false signal);
and arPLS weights use the noise of the whole series, so with the x10 emission
steps the whole-run baseline sinks to the bottom of the noise late in the run
(an offset is left, e.g. m/z 70: 60 under troughs of 100-200). By default the
changes are read from `TPS2` (full resolution, logged ~0.5 s after they take
effect), the scan of each step located in the data (tic for the emission, each
mass for the ejection) and the baseline fitted separately between the steps,
with lam 1e7 (`peaxart.StepConfig`, cuts by `peaxart.step_cuts`): all masses at the emission
steps, at the ejection switches the masses whose level drops by > x2 (from >= 20)
when the ejection is switched on. The cuts are in `result.steps` and the
provenance. `--no-steps`: one baseline over the whole run.
The m/z windows follow the calibration at the rt of each scan (`--mass-axis
drift`); the legacy pipeline used the calibration at the end of the run for all
scans (`end`). Default smoothing with edge padding (the legacy zero padding pulls
the baseline down at the ends of the run). `--legacy` (`--mass-axis end
--no-steps`, lam 1e5, zero padding) reproduces the legacy
`<run>.unit_mass_chromatos.parquet` (within 1e-6 of each column maximum). Output
`<run>.alpinac/<run>.unit_mass_chromatos.parquet`, the file of the legacy pipeline
(replaced): index `rt` (s), one column per unit mass (`"2"`, ...) and `tic` (their
sum), how it was made in its metadata (`esper.chromatograms.provenance`). Files with
several segments: the segment range in the column names (`"seg1:69"`, `"seg1:tic"`).
Next to it (to check the raw data; `--no-raw`: not
written), same layout in float32: `.raw.parquet`, the raw sums before smoothing,
and `.baseline.parquet`, the baselines removed (corrected = smoothed raw -
baseline, clipped at 0); `read_chromatograms(path, "raw")`. ~30 s per 50 min run
(half reading, half baselines).

### Application menu / desktop icon

```
sudo /shared/pyenv/bin/esper install --system --data-dir /shared/toffee/apre1   # all users
esper install --data-dir /shared/toffee/apre1 --desktop-shortcut                # only you
```

Adds the esper entry to the application menu (`share/applications/esper.desktop`, icon in
the hicolor theme) and the `esper` command to the PATH (`bin/esper`), in `/usr/local` with
`--system`, otherwise in `~/.local`. The entry starts `esper gui <data dir>`.
`--desktop-shortcut` also puts the icon on your desktop (with sudo: on the desktop of the
user who ran sudo). `--uninstall` removes it all.

`upload` (tofwerk) accepts a single `.alpinac` folder or a directory searched recursively.
Measurements whose `pipeline.nc` did not change are skipped (`--force` to reimport).

`targets` replaces all the target compounds of the database. The source is read by a loader
chosen from its file suffix (`esper.targets.LOADERS`); the Excel loader uses alpinac's
`load_excel_compounds` and the Q0–Q8 ion columns. The retention index of a target comes from
its rt in the reference run and alpinac's RI references, extrapolated linearly beyond the
first/last reference. To use another source, write a function returning `esper.targets.Targets`
and register it in `LOADERS`.

`esper gui` without argument uses `$ESPER_DATA_DIR`; if it is not set, a window lists the
data directories opened before (last 10, also in File > Open recent) to choose from, or to
browse for another one.

## GUI

- **Chromatograms** (Ctrl+1): one mass of one measurement around the selected compound,
  with the peaks found at that mass (rt, compound, height H, area A, width W).
  Right click > Integration: the fit integration (default: bases where the fitted peak falls
  to 1 % of its height, constant baseline, fitted shape dashed, and the fit values: rt,
  height, area ± uncertainty, σ, tailing, S/N), the bounds and baseline from the ips of
  the extraction, the fitted peaks (filled in the color of the mass, also for the overlaid masses; darker for
  the selected compound), or none (saved in the settings).
  Click on a peak apex to select its compound. Changing file keeps the compound and the mass.
  Right click > Masses overlays other chromatograms: 2nd to 5th most prominent mass of the
  compound, all compound masses, masses peaking here (every mass with a peak apex in the
  visible rt range, most prominent first; follows pans and zooms), all masses, TIC. The main
  mass is black, up to 5 overlays get the qualifier colors (`style.QUALIFIER_COLORS`), more
  are colored along magma. The relative chromatograms below (x axis
  linked) show the same curves, each rescaled to the range of the main mass in the visible
  window.
- **Results** (Ctrl+2): one compound (Cmp) across all measurements. Click a point to select
  that measurement/compound, double click to open its chromatogram. Each run type has its
  color (`style.RUNTYPE_COLORS`, by prefix: air blue; std red; tank orange; cal green;
  blank purple; tert pink; labair black; others from `style.COLORS`).
- **Search** (Ctrl+3): peaks of some masses (e.g. `85, 87 101-103`) in an rt or retention
  index window, across all measurements. Color = mass, symbol = run type. "only runs with all
  masses" keeps the runs where every mass has a peak in the window. "From selection" fills the
  window and the 3 main masses from the selected compound. Click / double click as in Results.
  The retention index of a peak is the one of its compound.
- **Mass calibration** (Ctrl+4): above, the calibration of all the runs over time (or by
  measurement, the points of a run spread by rt): its parameters p1, p2, … and the drift of
  the m/z of the settings (File > Settings > Mass calibration, default 20, 50, 100, 230),
  one line per run. "Drift in" ppm (error on the m/z if the median calibration of the plotted
  points were used) or tof index (shift of the tof of the m/z). Click a point to select the
  run. Below, the selected run against the rt: the drift of the calibration since its first
  window at the same m/z (lines), and of each calibrant peak (points colored by m/z, hollow:
  left out of the fit of its window; hover for the calibrant and its residual). The
  calibrant residuals (View menu) show each calibrant peak minus the calibration of its window.
  The y ranges leave out the few outliers (right click > View All shows them).

The View menu lists the optional panels of the current view: compound table, relative
chromatograms and spectrum (Chromatograms), chromatogram and spectrum of the selected point (Results),
drift in the run and calibrant residuals (Mass calibration). The
spectrum shows one stick per mass of the selected compound (prominence, % of the largest);
the masses plotted in the chromatogram panel of the same view have the same colors there
(main mass black, others grey). Clicking a stick selects that mass. Panel heights are adjustable
with the splitters.

The target table (left of Chromatograms, below the compounds) lists the targets with their RI,
the rt expected in the selected measurement (from the RI and the measurement's reference
compounds), whether they are found (a compound is named after the target or an alias) and
their expected ions. Choosing a target selects its compound if found; otherwise the
chromatogram goes to the expected rt with the target ions overlaid (whole run if the target
has no RI). The expected rt is a dashed line in the chromatogram, the expected ions dashed
lines in the spectrum. Changing measurement follows the target.

The spectrum is compared with a library spectrum mirrored below it (default: alpinac's best
candidate; right click > Reference library to show it from another library), with the cosine
score computed live. The **Best matches** table in the left column (the compound against all
spectra of the enabled libraries, computed only while shown) chooses another reference. Its
search box searches the spectra of all the configured libraries, also the disabled ones, by
`library/name` (e.g. `nist_ch/` or `CFC-11`), scored against the compound.

File > Settings (saved in `~/.config/esper/settings.json`):
- intensity of the measured peaks: prominence (default), height, area (integrated in the
  chromatograms between the ips bounds, slower) or fit_area (of the fitted peaks);
- spectral libraries (peaksdb folders: NIST `.jdx` and alpinac fragment `.txt` files, read like
  peaksdb's `db_reader` but without it, whose import takes seconds), checked = used, ordered by
  priority: a spectrum in several libraries is taken from the first one. alpinac's references
  (`mygu_X`, `nist_X`) are looked up in the libraries of that family (`mygu…`, `nist…`).
  Libraries are read in the background when first needed (progress in the status bar) and
  kept in memory;
- matching (alpinac's processing): cutoff mass, ignored masses, normalization, number of best
  matches. The score is the cosine of the unit mass spectra (same as matchms `CosineGreedy`).

The compound table (left of Chromatograms) lists the compounds extracted in the selected
measurement (rt, RI, name, scores, valid, number of peaks, main ion); it is sortable, has a
text filter and a "valid only" filter, and choosing a row selects the compound.

With the ips integration, A is computed by esper from the chromatogram, above a straight
baseline over ±3 half widths at half maximum, limited to the prominence bases of the peak
(none for the peaks split by the fit). With the fit integration and the fitted peaks, A is
`fit_area`.

## GUI structure (`src/esper/gui/`)

- **Views** (`views/`) are the tabs: a toolbar above a vertical stack of panels
  (`BaseView.add_panel`). The first panel is the main one; optional panels are listed in
  the View menu.
- **Panels** (`panels/`) show something for the current selection (`BasePanel`). They are
  redrawn when the selection fields they `watch` change, only while visible; a hidden panel
  is redrawn when shown. A panel implements `redraw()` and calls `invalidate()` when its own
  options change. Panels go in the main stack or in the left column (`left=True`).
- **Table panels** (`panels/table.py`, `TablePanel`) declare their `columns` and implement
  `load()` (rows indexed by item key), `current_key()` and `select(key)`; sorting, text
  filter, row count and following the selection are shared.
- **Selection** (`selection.py`) is the state shared by all panels: measurement, compound,
  compound name, mass, rt.
- **EsperData** (`data.py`) queries the database and the chromatogram files on demand, with
  small caches.

## Data directory

```
<data_dir>/
    esper.yaml      instrument (written by the first upload), optional database path
    esper.sqlite
    chromatograms/<measurement>.parquet
    chromatograms/<measurement>.raw.parquet        (esper chromatograms only)
    chromatograms/<measurement>.baseline.parquet   (esper chromatograms only)
```

`upload` copies the unit mass chromatograms of each run there (`<run>.unit_mass_chromatos.parquet`,
of `esper chromatograms` or of the legacy pipeline), with the raw sums and baselines of
`esper chromatograms` when they are next to it. The database
only stores the path of the chromatograms (`measurement.chromatogram_file`); the raw sums and
baselines are next to it (`esper.chromatograms.read_chromatograms(path, "raw")`).

The database can live elsewhere, e.g. on a faster disk than the chromatograms: put an
`esper.yaml` in the data directory with the path of the database (absolute, `~/...`, or
relative to the data directory), and the instrument of its runs (written by the first
`upload --instrument`):

```yaml
instrument: tofwerk
database: /home/me/Data/esper/esper.sqlite
```

Without `esper.yaml`, or without `database` in it, the database is `esper.sqlite` in the data
directory. All commands (`upload`, `fit`, `targets`, `gui`) follow it; `upload` creates the database
there if it does not exist yet.

Tables: `measurement`, `mass_cal`, `mass_cal_rt`, `mass_cal_mass`, `mass_cal_peak`, `compound`,
`compound_candidate`, `peak`, and the view `peak_full` (peaks joined with their
measurement, compound and best library candidate). See `src/esper/db.py`.

## Development

Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/); CI fails if either check does not pass.

```bash
pip install ruff
ruff check --fix .
ruff format .
```
