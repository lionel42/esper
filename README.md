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
peaks come from (`peak_source`); only the peaks of alpinac are fitted.

| instrument | runs | peaks |
|---|---|---|
| `tofwerk` | `<run>.alpinac` folders: the alpinac extraction, the chromatograms of `esper chromatograms`, the mass calibration and the h5 file next to the folder | `alpinac`, fitted |
| `gcwerks` | `<yy>/chromatograms/channel0/<run>` of a GCwerks directory | none (chromatograms only) |

[gcwerks](https://github.com/lionel42/gcwerks) (`src/esper/sources/gcwerks.py`):


You can specify the instrument at upload time:
```
esper upload /path/to/gcwerks/jungfraujoch-medusa /path/to/esper/data --instrument gcwerks
```


### Peak fit

After the import, the peaks of each mass are fitted in the unit mass chromatograms with
[peaxart](https://github.com/lionel42/peaxart) (`src/esper/fitting.py`)


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
    esper.yaml      instrument (written by the first upload), optional database path
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

## Development

Each command of `esper` is a module of `src/esper/cli/` (`NAME`, `HELP`, `add_arguments`,
`run`); to add one, write its module and add it to `COMMANDS` in `src/esper/cli/main.py`.

Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/); CI fails if either check does not pass.

```bash
pip install ruff
ruff check --fix .
ruff format .
```
