#!/usr/bin/env python3
"""
inspect_radar_metadata.py — dump Rainfields3 product metadata to check for
gauge-correction provenance.

Run ON GADI in the radar_sat venv:
    module load python3/3.11.7
    source /scratch/no50/nk4326/venvs/radar_sat/bin/activate
    python3 inspect_radar_metadata.py

Paste the full output back.
"""
import zipfile
from pathlib import Path
import netCDF4 as nc4

RADAR_ID = 71   # Sydney Terrey Hills
YEAR = 2022
BASE = Path("/g/data/rq0/rainfields3")

print(f"=== sibling product directories under {BASE / str(RADAR_ID)} ===")
radar_dir = BASE / str(RADAR_ID)
if radar_dir.exists():
    for yr_dir in sorted(radar_dir.iterdir()):
        if yr_dir.is_dir():
            products = sorted(p.name for p in yr_dir.iterdir() if p.is_dir())
            print(f"  {yr_dir.name}: {products}")
else:
    print(f"  !! {radar_dir} does not exist")

print(f"\n=== looking for docs at {BASE} and {BASE.parent} ===")
for d in (BASE, BASE.parent):
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".pdf", ".txt", ".md", ".csv") or p.name.lower().startswith("readme"):
                print(f"  {p}")

yr_dir = BASE / str(RADAR_ID) / str(YEAR) / "prcp-m15"
zips = sorted(yr_dir.glob("*.zip"))
print(f"\n=== sample file: radar {RADAR_ID} / {YEAR}, {len(zips)} daily zips found ===")
if not zips:
    raise SystemExit("No data — check the path above.")

with zipfile.ZipFile(zips[0]) as zf:
    nc_name = sorted(n for n in zf.namelist() if n.endswith(".nc"))[0]
    print(f"  reading {nc_name} from {zips[0].name}")
    raw = zf.read(nc_name)

with nc4.Dataset("mem", memory=raw) as ds:
    print("\n=== GLOBAL attributes ===")
    for a in ds.ncattrs():
        print(f"  {a} = {getattr(ds, a)!r}")

    print("\n=== variables ===")
    print(" ", list(ds.variables))

    print("\n=== 'precipitation' variable attributes ===")
    precip = ds.variables["precipitation"]
    for a in precip.ncattrs():
        print(f"  {a} = {getattr(precip, a)!r}")

    # dump attrs for any other non-coordinate variable too (proj/quality/etc.)
    for vname, v in ds.variables.items():
        if vname in ("precipitation", "x", "y"):
            continue
        print(f"\n=== '{vname}' variable attributes ===")
        for a in v.ncattrs():
            print(f"  {a} = {getattr(v, a)!r}")
