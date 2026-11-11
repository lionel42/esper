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