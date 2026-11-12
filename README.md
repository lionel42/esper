# esper

Collects the peaks found by the alpinac extraction pipeline (`<measurement>.alpinac/pipeline.nc`)
into a SQLite database, together with the mass calibration and the unit mass chromatograms.

## Install

```
pip install -e .
```

Once esper is installed, the `esper` command is available in the PATH.

You can install the gui app on linux by running

```
esper install
```

## Usage


`esper --help` lists the commands, and `esper <command> --help` gives the options of each.

Esper needs for each instrument to have its own data directory.
for example, a typical workflow is:

```bash
# import the runs of you instrument data and processes them
esper upload /path/to/runs /path/to/esper/data

# export the results 
esper export /path/to/esper/data /path/to/exported/results.csv

# Then you are able to visualize this in the esper GUI.
esper gui /path/to/esper/data
```

A data directory is declared in its `esper.yaml` which should 
contain all the metadata and settings. 
The other files are described in the [Data directory](#data-directory) section.

Esper is also available as a python package, and can be used in scripts:

```python
from esper import EsperData

data = EsperData("/path/to/esper/data")
# get the peaks of a measurement
peaks = data.peaks(measurement="2023-01-01_12-00-00")
# get the chromatograms of a measurement
chromatograms = data.chromatograms(measurement="2023-01-01_12-00-00")
``` 

### Data sources

Each instrument is a module of `src/esper/sources/` that finds the runs of a source path and
reads them (`find_runs`, `read`, `refresh`; see `esper/sources/__init__.py`); `esper.upload`
stores what they read. The measurement table keeps the `instrument` of each run and where its
peaks come from (`peak_source`: `alpinac`, or `esper` when esper found them).

| instrument | runs | peaks |
|---|---|---|
| `tofwerk` | `<run>.alpinac` folders: the alpinac extraction, the chromatograms of `esper chromatograms`, the mass calibration and the h5 file next to the folder | `alpinac`, fitted |
| `gcwerks` | `<yy>/chromatograms/channel0/<run>` of a GCwerks directory | none: found by esper (see [Peaks](#peaks)) |

[gcwerks](https://github.com/lionel42/gcwerks) (`src/esper/sources/gcwerks.py`):


You can specify the instrument at upload time:
```
esper upload /path/to/gcwerks/jungfraujoch-medusa /path/to/esper/data --instrument gcwerks
```


### Peaks

The runs imported without peaks (gcwerks) get them from `esper peaks`, or from `esper upload`
once the data directory has a method (`src/esper/processing/`):

1. the peaks of each mass are found in each window where it was measured, after removing the
   baseline there, and grouped into compounds of co-eluting peaks
   ([peaxart](https://github.com/lionel42/peaxart));
2. the compounds are identified by the method of the data directory: `target` (target search:
   the expected rt and ions of the targets, e.g. from GCwerks peakid) or `suspect` (suspect
   screening: the spectrum of each compound in the libraries, as alpinac). The compounds that
   match nothing are kept, without a name;
3. the peaks are fitted (below).

```
esper peaks /path/to/esper/data --method target
```

The method and the parameters of each step belong to the data directory: see `processing:`
in [Data directory](#data-directory).

### Peak fit

After the import (or `esper peaks`), the peaks of each mass are fitted in the chromatograms
with [peaxart](https://github.com/lionel42/peaxart) (`src/esper/fitting.py`), with the fit
parameters of the data directory.


### Mass calibration

`esper calibrate` fits the mass calibration of h5 files (`src/esper/calibrate.py`): the spectra
are read with [twutils](https://github.com/lionel42/twutils) (only the scans of each rt window and
the tof indices around the calibrants) and the calibration is fitted with [masscal](../masscal)
(peaks with peaxart), following its drift with the retention time. 


### Chromatograms

`esper chromatograms` makes the unit mass chromatograms (`src/esper/chromatograms.py`):
form GC-TOF-MS data, the spectra are read with twutils, summed around each unit mass and their baseline removed with
peaxart (`peaxart.unit_mass_sums`, `peaxart.remove_baseline`), with the mass calibration of the run
(`esper calibrate`, else the legacy `*_mc.txt`).



## Data directory

```
<data_dir>/
    esper.yaml      instrument (written by the first upload), processing, optional database path
    esper.sqlite
    chromatograms/<measurement>.parquet
    chromatograms/<measurement>.raw.parquet        (esper chromatograms only)
    chromatograms/<measurement>.baseline.parquet   (esper chromatograms only)
```

The SQLite database can live elsewhere, e.g. on a faster disk than the chromatograms: put an
`esper.yaml` in the data directory with the path of the database (absolute, `~/...`, or
relative to the data directory), and the instrument of its runs (written by the first
upload):

```yaml
instrument: tofwerk
database: /home/me/Data/esper/esper.sqlite
```

How the peaks are found, identified and fitted is set by `processing:` (all optional but the
method; the others are [peaxart](https://github.com/lionel42/peaxart)'s parameters, see
`src/esper/processing/config.py`):

```yaml
processing:
  method: target            # target | suspect (written by --method)
  retention_index: false    # retention indices used (default: not with target search)
  baseline:                 # peaxart.BaselineConfig, before the detection (false: none)
    lam: 1.0e+7
  detection:                # peaxart.LegacyDeconvolutionConfig: thresholds in counts,
    width: 5                # minimum width (half prominence), scans (default 5)
    height: 20
    prominence: 60
  fit:                      # peaxart.ChromatogramFitConfig
    peak_shape: emg
  target:
    window_s: 20            # window of the targets without one in their source
  suspect:
    min_score: 0.8          # valid from this spectral score on
```

The GUI shows only what the data directory uses: with target search, no library match
(score, spectrum) and, unless `retention_index: true`, no retention index.

## Development

Each command of `esper` is a module of `src/esper/cli/` (`NAME`, `HELP`, `add_arguments`,
`run`); to add one, write its module and add it to `COMMANDS` in `src/esper/cli/main.py`.

Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/); CI fails if either check does not pass.

```bash
pip install ruff
ruff check --fix .
ruff format .
```
