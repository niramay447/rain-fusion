#!/usr/bin/env python3
"""
explore_radar55.py — derive the Wagga (radar 55) constants needed to build a
clean copy of src/radar/au_preprocessor.py for the rural study.

Run ON GADI in the radar_sat venv:
    module load python3/3.11.7
    source /scratch/no50/nk4326/venvs/radar_sat/bin/activate
    python3 explore_radar55.py

Paste the full output back and I'll fill in the new preprocessor's constants:
  _PROJ params, _R0/_R1/_C0/_C1 crop, suggested stride, n_nodes.
"""
import zipfile
from pathlib import Path
import numpy as np
import netCDF4 as nc4

RADAR_ID = 55                      # Wagga Wagga
YEAR = 2022
BASE = Path("/g/data/rq0/rainfields3")

# Wagga rural 75-gauge bbox (lat_min, lat_max, lon_min, lon_max)
BBOX = (-36.254, -34.335, 146.256, 148.943)

yr_dir = BASE / str(RADAR_ID) / str(YEAR) / "prcp-m15"
zips = sorted(yr_dir.glob("*.zip"))
print(f"radar {RADAR_ID} / {YEAR}: {len(zips)} daily zips found in {yr_dir}")
if not zips:
    raise SystemExit("No data — check RADAR_ID / YEAR / path (ls /g/data/rq0/rainfields3/).")

with zipfile.ZipFile(zips[0]) as zf:
    nc_name = sorted(n for n in zf.namelist() if n.endswith(".nc"))[0]
    raw = zf.read(nc_name)

with nc4.Dataset("mem", memory=raw) as ds:
    print("\n=== variables ===")
    print(list(ds.variables))

    print("\n=== proj variable attributes (for _PROJ) ===")
    proj_var = None
    for cand in ("proj", "albers_conical_equal_area", "crs", "grid_mapping"):
        if cand in ds.variables:
            proj_var = ds.variables[cand]; print(f"(found proj var: '{cand}')"); break
    if proj_var is None:
        # fall back: any var with a grid_mapping_name attr
        for v in ds.variables.values():
            if "grid_mapping_name" in v.ncattrs():
                proj_var = v; break
    if proj_var is not None:
        for a in proj_var.ncattrs():
            print(f"   {a} = {getattr(proj_var, a)}")
    else:
        print("   !! no proj variable found — paste the variable list above and I'll adapt")

    x = np.array(ds.variables["x"][:], dtype=np.float64)
    y = np.array(ds.variables["y"][:], dtype=np.float64)
    precip = ds.variables["precipitation"]
    print("\n=== grid ===")
    print(f"   precipitation shape : {precip.shape}   (rows=y, cols=x)")
    print(f"   x: n={len(x)}  min={x.min():.3f} max={x.max():.3f}  units={getattr(ds.variables['x'],'units','?')}")
    print(f"   y: n={len(y)}  min={y.min():.3f} max={y.max():.3f}  units={getattr(ds.variables['y'],'units','?')}")

# ---- try to build the projection from CF attrs and compute the crop ----
print("\n=== crop indices for the Wagga bbox ===")
try:
    from pyproj import Proj
    a = {at: getattr(proj_var, at) for at in proj_var.ncattrs()}
    sp = a.get("standard_parallel", [None, None])
    sp = list(sp) if np.ndim(sp) else [sp, sp]
    P = Proj(
        proj="aea",
        lat_1=float(sp[0]), lat_2=float(sp[1]),
        lat_0=float(a["latitude_of_projection_origin"]),
        lon_0=float(a["longitude_of_central_meridian"]),
        x_0=float(a.get("false_easting", 0.0)),
        y_0=float(a.get("false_northing", 0.0)),
        a=float(a.get("semi_major_axis", 6378137.0)),
        b=float(a.get("semi_minor_axis", 6356752.31414)),
    )
    lat_min, lat_max, lon_min, lon_max = BBOX
    # project the four bbox corners; x/y arrays look to be in km → ×1000 to metres
    xs, ys = [], []
    for la in (lat_min, lat_max):
        for lo in (lon_min, lon_max):
            px, py = P(lo, la)
            xs.append(px); ys.append(py)
    xkm, ykm = x * 1000.0, y * 1000.0   # adjust if units printed above are already metres
    c0 = int(np.searchsorted(xkm, min(xs)) - 1)
    c1 = int(np.searchsorted(xkm, max(xs)) + 1)
    # y is typically descending (north→south); handle both
    if ykm[0] > ykm[-1]:
        r0 = int(len(ykm) - np.searchsorted(ykm[::-1], max(ys)) - 1)
        r1 = int(len(ykm) - np.searchsorted(ykm[::-1], min(ys)) - 1)
    else:
        r0 = int(np.searchsorted(ykm, min(ys)) - 1)
        r1 = int(np.searchsorted(ykm, max(ys)) + 1)
    r0, r1 = sorted((max(0, r0), min(len(y) - 1, r1)))
    c0, c1 = sorted((max(0, c0), min(len(x) - 1, c1)))
    print(f"   _R0, _R1 = {r0}, {r1}   ({r1 - r0 + 1} rows)")
    print(f"   _C0, _C1 = {c0}, {c1}   ({c1 - c0 + 1} cols)")
    for stride in (10, 15, 20, 25):
        nr = len(range(r0, r1 + 1, stride)); nc = len(range(c0, c1 + 1, stride))
        print(f"   stride={stride:2d} -> grid {nr}x{nc} = {nr*nc} nodes "
              f"(~{stride*0.5:.0f} km res)   (Sydney was ~952 nodes @ stride 10)")
except Exception as e:
    print(f"   auto-crop failed ({e}) — paste the proj attrs + x/y ranges above and I'll compute it.")
