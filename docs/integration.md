# Integration of the peaks

How esper computes the area of each peak found by alpinac, and its uncertainty. Run at
`esper upload` (unless `--skip-integration`), with `esper integrate`, or from the GUI with
Calculate > Integrate (see the [README](../README.md)).

## Method

Code: [`src/esper/integration.py`](../src/esper/integration.py) (noise model: [`src/esper/noise.py`](../src/esper/noise.py)). Areas are in counts·s.

- **Window:** from 4 half widths at half maximum before the apex to 6 after (each side with
  its own half width, longer on the tail side because GC peaks tail), limited to the
  prominence bases of the peak.
- **Overlapping peaks** of the same mass (windows overlapping directly, or through a broader
  peak whose window contains them) form a group with one baseline, from the start of the
  first window to the end of the last one, split at the lowest points between consecutive
  apexes (drop line). Where the signal comes back to the baseline between two peaks (within
  10 % of the smaller one, or the noise), the group is split there with separate baselines
  (baseline reset). A narrow peak on a broad one keeps only its own window; the tail of the
  broad peak after it goes back to the broad peak (integrated in two pieces, its stored bounds
  are the outer ones).
- **Small peaks on a tail** (height above the tail under 20 % of the large peak, window after
  its apex, tail not bending down like the side of a gaussian): the tail is extrapolated under
  the small peak, exponential from the valley to the end of the small peak (exponential skim);
  the large peak keeps the tail. On synthetic tailing peaks: −18 % to +1 % for the small peak
  (drop line: +13 % to +91 %). Shoulders on a gaussian side and peaks on the front of a larger
  one keep the drop line. Known limit: when the group ends with the small peak, the baseline
  ends on the slow tail of the large peak and takes part of it.
- **Valleys:** an outer bound of a group falling on another peak that alpinac did not pick
  is moved back to the valley before it.
- **Baseline:** straight from the start to the end of the group, on the signal smoothed over
  11 scans, but never clearly above the signal: where the signal goes below it by more than
  4× the noise, it follows the lower convex hull of the signal. When the window is cut by the
  end of the chromatogram, the baseline is flat.

## Uncertainty

`area_unc`, 1σ, statistical only: not the choice of window or baseline.

- noise model per measurement, from the quiet parts of 60 of its chromatograms: variance per
  scan = floor (per mass) + gain × signal (counting noise), and the correlation between
  neighbouring scans (on the TOF data ~0.7–0.85 at 1 scan, ~0.3 at 2), which makes the noise
  of a sum of scans ~3× larger than for independent scans;
- propagated through the trapezoid sum: noise of the signal under the peak, of the baseline
  levels (means of 11 scans) and, for split peaks, of the split position (noise of the slope
  over the curvature of the valley; opposite for the two peaks, so their sum is not affected);
- summed areas (`area_sum`, a q with several peaks) add the uncertainties in quadrature;
- when there is too little data to estimate the noise, the uncertainty is unknown (NULL, and
  so is a sum including it), never 0; the baseline is then the straight line (the hull and
  the valleys compare the signal with the noise).

Checks: on synthetic chromatograms with this kind of noise, the scatter of the areas over
repeated noise is 0.85–0.95× the uncertainty (single, large, small, split and tailing peaks).
On empty quiet windows of real chromatograms: about right above ~10 counts of background,
conservative (up to ~1.7×) on nearly empty masses; areas of empty windows are positive by
~0.3σ on average (the baseline rules can only lower the baseline). Bumps not picked by alpinac
are signal, not noise, and are not covered.

## Stored results

Stored per peak (`int_start_s`, `int_stop_s`, `base_start`, `base_stop`, `area`,
`area_unc`), per compound (`area_sum`, `area_sum_unc`, and `q1_mass`, `q1_area`,
`q1_area_unc` … `q5_…`: its masses by decreasing area, peaks on the same mass summed) and
per measurement (`integration_version`, `integrated_at`, and the noise parameters
`noise_gain`, `noise_factor`, `noise_d2_ratio`, so that the GUI integrates exactly as stored).
`INTEGRATION_VERSION` is increased when the method changes, so that
`esper integrate` redoes the measurements integrated before. The chromatogram view
integrates the shown mass itself, to draw the exact baselines; the areas are the same.

## Tests

Tests on synthetic gaussian and tailing (exponentially modified gaussian) peaks, with
correlated counting noise for the uncertainties, and an end-to-end test of a small synthetic
data directory (`tests/test_database.py`): `pip install -e .[dev]` then `pytest tests`.
