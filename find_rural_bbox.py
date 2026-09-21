#!/usr/bin/env python3
"""
find_rural_bbox.py — locate a NSW bounding box with ~N numeric rain gauges at
LOWER density (larger gauge spacing) than the Sydney study box.

Density is summarised by mean/median nearest-neighbour (NN) distance between
gauges: Sydney is dense (small spacing); a rural box with the same N gauges has
larger spacing — that's the "sparser" regime where radar/satellite fusion should
help more.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

FOI = "database/australia/raw_sydney/_foi_all_rainfall.xml"
SYD_META = "database/australia/station_metadata_sydney75.csv"
N = 75
EARTH_KM = 6371.0
NS = {"wml2": "http://www.opengis.net/waterml/2.0", "gml": "http://www.opengis.net/gml/3.2"}

# Approximate NSW BoM radar locations (lat, lon) — VERIFY against `ls /g/data/rq0/rainfields3/`
# and BoM radar coverage before committing. The rural box must sit inside a radar's range.
RADARS = {
    "71 Sydney/Terrey Hills": (-33.701, 151.210),
    "03 Wollongong/Appin":    (-34.264, 150.874),
    "40 Canberra/Captains F.":(-35.662, 149.512),
    "55 Wagga Wagga":         (-35.166, 147.467),
    "04 Newcastle":           (-32.730, 152.030),
    "69 Namoi/Gunnedah":      (-31.024, 150.192),
    "94 Hillston":            (-33.552, 145.523),
}


def load_numeric_stations(foi_path):
    root = ET.parse(foi_path).getroot()
    rows = []
    for mp in root.iter("{%s}MonitoringPoint" % NS["wml2"]):
        ident = mp.find("gml:identifier", NS)
        pos = mp.find(".//gml:Point/gml:pos", NS)
        if ident is None or pos is None:
            continue
        sid = ident.text.rstrip("/").split("/")[-1]
        if not sid.isdigit():
            continue
        try:
            lat, lon = map(float, pos.text.split())
        except (ValueError, AttributeError):
            continue
        rows.append({"station_id": sid, "latitude": lat, "longitude": lon})
    return pd.DataFrame(rows)


def nn_dists_km(coords):
    """Mean/median nearest-neighbour distance (km) within a set of [lat,lon] points."""
    if len(coords) < 2:
        return np.nan, np.nan
    rad = np.radians(coords)
    nn = NearestNeighbors(n_neighbors=2, metric="haversine").fit(rad)
    d, _ = nn.kneighbors(rad)
    nn_km = d[:, 1] * EARTH_KM
    return float(nn_km.mean()), float(np.median(nn_km))


def stats(coords):
    lat, lon = coords[:, 0], coords[:, 1]
    lat_km = (lat.max() - lat.min()) * 111.0
    lon_km = (lon.max() - lon.min()) * 111.0 * np.cos(np.radians(lat.mean()))
    area = max(lat_km * lon_km, 1e-6)
    mean_nn, med_nn = nn_dists_km(coords)
    return {
        "n": len(coords),
        "bbox": (round(lat.min(), 3), round(lat.max(), 3), round(lon.min(), 3), round(lon.max(), 3)),
        "area_km2": round(area, 0),
        "mean_nn_km": round(mean_nn, 2),
        "median_nn_km": round(med_nn, 2),
        "density_per_1000km2": round(len(coords) / area * 1000, 2),
    }


def main():
    df = load_numeric_stations(FOI)
    print(f"National numeric rainfall stations: {len(df)}")

    # ---- Sydney baseline (the actual 75 study gauges) ----
    syd = pd.read_csv(SYD_META, header=None,
                      names=["name", "id", "net", "latitude", "longitude"])
    syd_coords = syd[["latitude", "longitude"]].to_numpy()
    s = stats(syd_coords)
    print("\n=== SYDNEY baseline (your 75 study gauges) ===")
    print(f"  bbox(lat_min,lat_max,lon_min,lon_max) = {s['bbox']}")
    print(f"  area={s['area_km2']:.0f} km²  mean_NN={s['mean_nn_km']} km  "
          f"median_NN={s['median_nn_km']} km  density={s['density_per_1000km2']}/1000km²")
    syd_mean_nn = s["mean_nn_km"]

    # ---- restrict national list to NSW ----
    nsw = df[(df.latitude.between(-37.5, -28.0)) & (df.longitude.between(141.0, 153.7))].reset_index(drop=True)
    coords = nsw[["latitude", "longitude"]].to_numpy()
    rad = np.radians(coords)
    nn = NearestNeighbors(n_neighbors=N, metric="haversine").fit(rad)

    # ---- scan a grid of candidate centres; for each take the N nearest gauges ----
    cand = []
    for clat in np.arange(-37.0, -28.0, 0.4):
        for clon in np.arange(141.0, 153.6, 0.4):
            d, idx = nn.kneighbors(np.radians([[clat, clon]]))
            sub = coords[idx[0]]
            st = stats(sub)
            # spacing from centre to its Nth gauge (how "reachable" the cluster is)
            st["center"] = (round(clat, 2), round(clon, 2))
            st["far_km"] = round(d[0, -1] * EARTH_KM, 1)   # dist to 75th gauge
            cand.append(st)

    cdf = pd.DataFrame(cand)
    # only clusters denser-reachable than ~half a radar range, and SPARSER than Sydney
    cdf = cdf[(cdf.far_km <= 180) & (cdf.mean_nn_km > syd_mean_nn)]
    cdf = cdf.sort_values("mean_nn_km", ascending=False)

    # dedupe centres that are close together (keep the sparsest in each ~0.8° cell)
    seen, picks = set(), []
    for _, r in cdf.iterrows():
        key = (round(r["center"][0] / 0.8), round(r["center"][1] / 0.8))
        if key in seen:
            continue
        seen.add(key)
        picks.append(r)
        if len(picks) >= 12:
            break

    print(f"\n=== Candidate rural boxes (~{N} gauges, sparser than Sydney mean_NN={syd_mean_nn} km) ===")
    print("ranked by mean_NN (sparsest first); --bbox order = lat_min lat_max lon_min lon_max\n")
    for r in picks:
        bm = min(RADARS.items(),
                 key=lambda kv: (kv[1][0] - r["center"][0]) ** 2 + (kv[1][1] - r["center"][1]) ** 2)
        print(f"  center {r['center']}  n={r['n']}  mean_NN={r['mean_nn_km']} km "
              f"({r['mean_nn_km']/syd_mean_nn:.1f}× Sydney)  area={r['area_km2']:.0f} km²")
        print(f"     --bbox {r['bbox'][0]} {r['bbox'][1]} {r['bbox'][2]} {r['bbox'][3]}   "
              f"nearest radar: {bm[0]}\n")


if __name__ == "__main__":
    main()
