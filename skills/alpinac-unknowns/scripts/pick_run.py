"""Step 0: choose the run to analyse.

usage: python pick_run.py [--date yymmdd|today] [--type air]

Prints the newest run of that type that has both pipeline.nc and the
unit-mass parquet and is not yet listed in runs.csv (findings log). With
--date, only runs of that day are considered. Runs still being processed
(no pipeline.nc yet) are skipped. Exits with status 1 if there is none.
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None)
ap.add_argument("--type", default="air")
args = ap.parse_args()

state = Path(os.environ.get("ALPINAC_UNKNOWNS_DIR", Path.home() / ".local/share/alpinac/unknowns"))
runs_csv = state / "runs.csv"
done = set()
if runs_csv.exists():
    done = {line.split(",", 1)[0] for line in runs_csv.read_text().splitlines()[1:] if line}

day = dt.date.today().strftime("%y%m%d") if args.date == "today" else args.date
if day:
    years = [2000 + int(day[:2])]
else:
    # also look in last year's folder (January)
    y = dt.date.today().year
    years = [y, y - 1]

for year in years:
    for d in sorted((C.TOF_ROOT / str(year)).glob(f"*.{args.type}.*.alpinac"), reverse=True):
        name = d.name.removesuffix(".alpinac")
        if day and not name.startswith(day):
            continue
        nc, parquet = C.run_files(d)
        if name not in done and nc.exists() and parquet.exists():
            print(d)
            sys.exit(0)
sys.exit(f"no un-analysed {args.type} run found" + (f" for {day}" if day else ""))
