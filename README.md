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
esper targets /shared/data/substances.xlsx /path/to/esper/data  # import the target compounds
esper gui /path/to/esper/data                               # open the GUI
```

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

`upload` accepts a single `.alpinac` folder or a directory searched recursively.
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
  with the peaks found at that mass (baseline, rt, compound, height H, area A, width W).
  Click on a peak apex to select its compound. Changing file keeps the compound and the mass.
  Right click > Masses overlays other chromatograms: 2nd to 5th most prominent mass of the
  compound, all compound masses, masses peaking here (every mass with a peak apex in the
  visible rt range, most prominent first; follows pans and zooms), all masses, TIC. The main
  mass is black, up to 5 overlays get the qualifier colors (`style.QUALIFIER_COLORS`), more
  are colored along magma. The relative chromatograms below (x axis
  linked) show the same curves, each rescaled to the range of the main mass in the visible
  window.
- **Results** (Ctrl+2): one compound (Cmp) across all measurements. Click a point to select
  that measurement/compound, double click to open its chromatogram.
- **Search** (Ctrl+3): peaks of some masses (e.g. `85, 87 101-103`) in an rt or retention
  index window, across all measurements. Color = mass, symbol = run type. "only runs with all
  masses" keeps the runs where every mass has a peak in the window. "From selection" fills the
  window and the 3 main masses from the selected compound. Click / double click as in Results.
  The retention index of a peak is the one of its compound.

The View menu lists the optional panels of the current view: compound table, relative
chromatograms and spectrum (Chromatograms), chromatogram and spectrum of the selected point (Results). The
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
- intensity of the measured peaks: prominence (default), height or area (integrated in the
  chromatograms, slower);
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

A is computed by esper from the chromatogram, above a straight baseline over
±3 half widths at half maximum, limited to the prominence bases of the peak.

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
    esper.yaml      (optional)
    esper.sqlite
    chromatograms/<measurement>.parquet
```

The database can live elsewhere, e.g. on a faster disk than the chromatograms: put an
`esper.yaml` in the data directory with the path of the database (absolute, `~/...`, or
relative to the data directory):

```yaml
database: /home/me/Data/esper/esper.sqlite
```

Without `esper.yaml`, or without `database` in it, the database is `esper.sqlite` in the data
directory. All commands (`upload`, `targets`, `gui`) follow it; `upload` creates the database
there if it does not exist yet.

Tables: `measurement`, `mass_cal`, `mass_cal_rt`, `mass_cal_mass`, `compound`,
`compound_candidate`, `peak`, and the view `peak_full` (peaks joined with their
measurement, compound and best library candidate). See `src/esper/db.py`.
