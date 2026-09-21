#!/usr/bin/env python3
"""
build_gauge_dataset.py
======================

End-to-end builder for a rain-gauge dataset from the Bureau of Meteorology
WaterData Online SOS API, producing the two files the training pipeline
consumes (see src/raingauge/utils.py):

  1. <region>_rainfall_long.csv   columns: timestamp, station_id, rainfall_mm
                                  hourly, timezone-naive local time
  2. <region>_station_metadata.csv  NO header, columns:
                                  name, station_id, network, latitude, longitude

This consolidates four scattered capstone scripts into one parameterised tool:
  - data_fetch.py / create_station_csv.py  (SOS GetObservation per station)
  - combine_hourly_data.py                 (tipping-bucket -> hourly)
  - sydney_bbox_filter.ipynb               (station discovery by bbox)

Pipeline
--------
  1. DISCOVER  SOS GetFeatureOfInterest -> all rainfall stations nationally,
               then keep numeric-ID stations inside --bbox.
  2. DOWNLOAD  SOS GetObservation per station -> raw XML -> raw per-station CSV
               (timestamp, rainfall_mm). XML/CSV are cached in --workdir so
               re-runs are cheap; use --force to re-download.
  3. HOURLY    UTC -> local tz, dedup, keep tip values {0.0, --tip-size},
               resample('h').sum(), reindex to a full hourly grid (fill 0).
  4. COMBINE   concat all stations -> long CSV; write metadata for the
               stations that actually produced data.

Two things that were hardcoded to Sydney in the original scripts and WILL
silently break elsewhere are now explicit parameters:
  --tip-size   tipping-bucket resolution (mm). Sydney gauges are 0.5 mm, but
               many rural BoM gauges are 0.2 mm. With the wrong value the tip
               filter discards every real tip and you get an all-zero dataset.
               The script prints a diagnostic and warns if the filter drops
               almost everything.
  --timezone   local timezone for hour bucketing (e.g. Australia/Perth for WA).

NOTE on hour labelling: resample('h') is left-labelled, so hour H is the
accumulation over [H:00, H+1:00).  This matches the original gauge pipeline
(combine_hourly_data.py) and is kept deliberately for comparability with the
existing Sydney dataset.  Be aware the radar preprocessor uses end-of-period
labelling ([H-1:00, H:00)); reconciling the two is a separate decision.

Example
-------
  python build_gauge_dataset.py \
      --region wagga \
      --bbox -35.50 -35.00 147.00 147.60 \
      --start 2021-01-01 --end 2025-12-31 \
      --timezone Australia/Sydney \
      --tip-size 0.5 \
      --outdir database/australia
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import requests

BASE_URL = "https://www.bom.gov.au/waterdata/services"
RAINFALL_PROPERTY = "http://bom.gov.au/waterdata/services/parameters/Rainfall"

# Namespaces used in the SOS / WaterML2 responses
NS = {
    "wml2": "http://www.opengis.net/waterml/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
}

# The BoM backend ("WDP") intermittently returns an OWS ExceptionReport with
# "Error connecting to WDP" even for valid requests. We retry on that.
_TRANSIENT_MARKER = "ExceptionReport"


def _sos_get(params: dict, *, max_retries: int = 5, backoff: float = 3.0,
             timeout: int = 180) -> bytes:
    """
    Issue a GET to the SOS endpoint, retrying on transient OWS exceptions and
    HTTP errors. Returns the raw response bytes, or raises on persistent failure.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
        except requests.RequestException as e:
            last_err = f"request error: {e}"
        else:
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
            elif _TRANSIENT_MARKER in resp.text[:600]:
                # OWS ExceptionReport (usually transient "Error connecting to WDP")
                last_err = "OWS ExceptionReport"
            else:
                return resp.content
        if attempt < max_retries:
            wait = backoff * attempt
            print(f"    retry {attempt}/{max_retries - 1} after {last_err} "
                  f"(waiting {wait:.0f}s)")
            time.sleep(wait)
    raise RuntimeError(f"SOS request failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# 1. DISCOVER
# ---------------------------------------------------------------------------
def discover_stations(bbox: tuple, workdir: str, force: bool = False) -> pd.DataFrame:
    """
    GetFeatureOfInterest for all rainfall stations nationally, parse to a
    DataFrame, then keep numeric-ID stations inside the bbox.

    bbox = (lat_min, lat_max, lon_min, lon_max)
    Returns columns: name, station_id, latitude, longitude
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    foi_path = os.path.join(workdir, "_foi_all_rainfall.xml")

    if force or not os.path.exists(foi_path):
        print("Discovering stations via GetFeatureOfInterest ...")
        content = _sos_get({
            "service": "SOS",
            "version": "2.0",
            "request": "GetFeatureOfInterest",
            "observedProperty": RAINFALL_PROPERTY,
        })
        with open(foi_path, "wb") as f:
            f.write(content)
        print(f"  saved national station list -> {foi_path} ({len(content):,} bytes)")
    else:
        print(f"Using cached national station list {foi_path}")

    tree = ET.parse(foi_path)
    root = tree.getroot()

    rows = []
    for mp in root.iter("{%s}MonitoringPoint" % NS["wml2"]):
        ident = mp.find("gml:identifier", NS)
        name = mp.find("gml:name", NS)
        pos = mp.find(".//gml:Point/gml:pos", NS)
        if ident is None or pos is None:
            continue
        # identifier: http://bom.gov.au/waterdata/services/stations/504019
        station_id = ident.text.rstrip("/").split("/")[-1]
        # pos: "lat lon" in EPSG:4326
        try:
            lat_str, lon_str = pos.text.split()
            lat, lon = float(lat_str), float(lon_str)
        except (ValueError, AttributeError):
            continue
        rows.append({
            "name": name.text if name is not None else station_id,
            "station_id": station_id,
            "latitude": lat,
            "longitude": lon,
        })

    df = pd.DataFrame(rows)
    print(f"  parsed {len(df):,} rainfall stations nationally")

    # Keep only numeric station IDs (drops GW…, PI_… analytic series)
    numeric = df["station_id"].apply(lambda s: str(s).isdigit())
    in_bbox = (
        (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    )
    out = df[numeric & in_bbox].reset_index(drop=True)
    print(f"  {len(out)} numeric-ID stations inside bbox "
          f"lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")
    if out.empty:
        print("  ⚠️  No stations in bbox — check your --bbox (lat_min lat_max "
              "lon_min lon_max, EPSG:4326).")
    return out


# ---------------------------------------------------------------------------
# 2. DOWNLOAD (per station: GetObservation -> raw CSV)
# ---------------------------------------------------------------------------
def download_station(station_id: str, start: str, end: str, workdir: str,
                     sleep: float, force: bool = False) -> pd.DataFrame | None:
    """
    Download one station's full rainfall series and parse to a raw DataFrame
    (timestamp, rainfall_mm). Caches XML and raw CSV in workdir. Returns None
    if the station has no data.
    """
    xml_path = os.path.join(workdir, f"{station_id}_rainfall.xml")
    raw_csv = os.path.join(workdir, f"{station_id}_rainfall_raw.csv")

    if not force and os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        return df if not df.empty else None

    if force or not os.path.exists(xml_path):
        content = _sos_get({
            "service": "SOS",
            "version": "2.0",
            "request": "GetObservation",
            "observedProperty": RAINFALL_PROPERTY,
            "featureOfInterest":
                f"http://bom.gov.au/waterdata/services/stations/{station_id}",
            "temporalFilter": f"om:phenomenonTime,{start}/{end}",
        })
        with open(xml_path, "wb") as f:
            f.write(content)
        time.sleep(sleep)  # be polite to the BoM endpoint

    tree = ET.parse(xml_path)
    root = tree.getroot()
    times, values = [], []
    for point in root.findall(".//wml2:MeasurementTVP", NS):
        t = point.find("wml2:time", NS)
        v = point.find("wml2:value", NS)
        if t is not None and v is not None:
            times.append(t.text)
            values.append(v.text)

    if not times:
        return None
    df = pd.DataFrame({"timestamp": times, "rainfall_mm": values})
    df.to_csv(raw_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. HOURLY (tipping-bucket -> hourly accumulation)
# ---------------------------------------------------------------------------
def to_hourly(df: pd.DataFrame, station_id: str, timezone: str, tip_size: float,
              hourly_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert a raw per-tip series to a complete hourly accumulation series.

    Steps mirror combine_hourly_data.py:
      - parse timestamps as UTC, convert to local tz, drop tz
      - dedup timestamps (keep first)
      - coerce numeric, keep only tip values {0.0, tip_size}
      - resample('h').sum()  (left-labelled: hour H = [H:00, H+1:00))
      - reindex to the full hourly grid, filling gaps with 0
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(timezone).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0)

    # --- value handling -----------------------------------------------------
    # The feed mixes individual tips (e.g. 0.2 mm, the dominant positive value)
    # with rare contaminating aggregate/cumulative records (e.g. 202, 1222 mm).
    # Keeping only {0, tip} discards the contaminants and keeps the real tips
    # (losing only the rare multi-tip readings — negligible).
    #   tip_size == 'auto'  → per-station tip = dominant (modal) positive value.
    #                         Correct for heterogeneous networks (0.1/0.2/0.5 mm).
    #   tip_size == 'none'  → no filter; dedup + sum all (only safe if the feed
    #                         has no aggregate/daily-total records).
    #   tip_size == float   → fixed {0, tip} (uniform networks, e.g. Sydney 0.5).
    detected_tip = None
    mode = str(tip_size).lower()
    if mode in ("none", "sum", "all"):
        pass  # no value filter
    else:
        if mode == "auto":
            pos = df.loc[df["rainfall_mm"] > 0, "rainfall_mm"].round(2)
            detected_tip = float(pos.mode().iloc[0]) if len(pos) else None
            ts = detected_tip
        else:
            ts = float(tip_size)
        if ts is not None:
            df = df[df["rainfall_mm"].isin([0.0, ts])]
        else:
            df = df[df["rainfall_mm"] == 0.0]  # station had no rain

    df = df.set_index("timestamp")
    hourly = df.resample("h").sum()
    hourly = hourly.reindex(hourly_range, fill_value=0)
    hourly = hourly.reset_index()
    hourly.columns = ["timestamp", "rainfall_mm"]
    hourly["station_id"] = station_id
    return hourly[["timestamp", "station_id", "rainfall_mm"]], detected_tip


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Build a gauge rainfall dataset from the BoM WaterData SOS API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--region", required=True,
                   help="Short label used in output filenames, e.g. 'wagga'")
    p.add_argument("--bbox", required=True, nargs=4, type=float,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                   help="Bounding box in EPSG:4326")
    p.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--timezone", default="Australia/Sydney",
                   help="Local timezone for hourly bucketing")
    p.add_argument("--tip-size", type=str, default="0.5",
                   help="Tipping-bucket resolution in mm for the {0,tip} filter "
                        "(Sydney=0.5). Use 'none' to disable the filter and sum all "
                        "deduped readings — correct for interval-accumulation feeds "
                        "(rural NSW water gauges report this way).")
    p.add_argument("--max-annual-mm", type=float, default=None,
                   help="Drop stations whose total exceeds this many mm per year "
                        "(× number of years) as implausible — e.g. 2500 removes "
                        "alpine snow gauges. Default: keep all.")
    p.add_argument("--min-annual-mm", type=float, default=None,
                   help="Drop stations whose total is below this many mm per year "
                        "as faulty/stuck (full-year coverage but ~no rain) — "
                        "e.g. 100. Default: keep all.")
    p.add_argument("--outdir", default="database/australia",
                   help="Where the final long + metadata CSVs are written")
    p.add_argument("--workdir", default=None,
                   help="Cache dir for XML/raw CSV (default: <outdir>/raw_<region>)")
    p.add_argument("--sleep", type=float, default=2.0,
                   help="Seconds to sleep between downloads (BoM throttling)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N discovered stations (for testing)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if cached files exist")
    args = p.parse_args()

    workdir = args.workdir or os.path.join(args.outdir, f"raw_{args.region}")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    print("=" * 80)
    print(f"Region        : {args.region}")
    print(f"BBox          : lat[{args.bbox[0]}, {args.bbox[1]}]  "
          f"lon[{args.bbox[2]}, {args.bbox[3]}]")
    print(f"Period        : {args.start} .. {args.end}")
    print(f"Timezone      : {args.timezone}")
    print(f"Tip size      : {args.tip_size} mm")
    print(f"Out dir       : {args.outdir}")
    print(f"Work/cache dir: {workdir}")
    print("=" * 80)

    # 1. discover
    stations = discover_stations(tuple(args.bbox), workdir, force=args.force)
    if stations.empty:
        sys.exit("No stations discovered — aborting.")
    if args.limit:
        stations = stations.head(args.limit)
        print(f"--limit set: processing first {len(stations)} stations only")

    # full hourly grid (left-labelled, timezone-naive)
    hourly_range = pd.date_range(
        start=f"{args.start} 00:00:00", end=f"{args.end} 23:00:00", freq="h"
    )

    # 2 + 3. per-station download and hourly conversion
    all_hourly = []
    kept_meta = []
    tip_hist = {}   # detected tip -> count of stations (auto mode)
    n_ok = n_empty = n_fail = 0
    total = len(stations)
    for i, row in enumerate(stations.itertuples(index=False), 1):
        sid = row.station_id
        print(f"[{i}/{total}] {sid} ({row.name}) ...", end=" ")
        try:
            raw = download_station(sid, args.start, args.end, workdir,
                                   sleep=args.sleep, force=args.force)
            if raw is None or raw.empty:
                print("no data")
                n_empty += 1
                continue
            hourly, det_tip = to_hourly(raw, sid, args.timezone, args.tip_size, hourly_range)
            if det_tip is not None:
                tip_hist[det_tip] = tip_hist.get(det_tip, 0) + 1
            total_mm = hourly["rainfall_mm"].sum()
            wet_hours = int((hourly["rainfall_mm"] > 0).sum())
            tip_str = f" tip={det_tip}" if det_tip is not None else ""
            print(f"ok | {wet_hours} wet hours | {total_mm:.1f} mm{tip_str}")
            all_hourly.append(hourly)
            kept_meta.append(row)
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — keep going on per-station errors
            print(f"ERROR: {e}")
            n_fail += 1
            continue

    print("=" * 80)
    print(f"Stations: {n_ok} ok, {n_empty} empty, {n_fail} failed, {total} total")
    if tip_hist:
        hist = ", ".join(f"{k} mm: {v}" for k, v in sorted(tip_hist.items()))
        print(f"Detected tip sizes (auto): {hist}")
    if not all_hourly:
        sys.exit("No station produced data — aborting.")

    # 4. combine + write outputs
    combined = pd.concat(all_hourly, ignore_index=True)

    # Drop stations whose total exceeds the implausibility cap (e.g. alpine
    # snow gauges that over-count): compares each station's period total to
    # max_annual_mm × n_years.
    keep_ids = None
    if args.max_annual_mm is not None or args.min_annual_mm is not None:
        n_years = int(args.end[:4]) - int(args.start[:4]) + 1
        tot = combined.groupby("station_id")["rainfall_mm"].sum()
        hi = (args.max_annual_mm * n_years) if args.max_annual_mm is not None else np.inf
        lo = (args.min_annual_mm * n_years) if args.min_annual_mm is not None else -np.inf
        dropped = tot[(tot > hi) | (tot < lo)]
        if len(dropped):
            print(f"Dropping {len(dropped)} station(s) outside "
                  f"[{args.min_annual_mm}, {args.max_annual_mm}] mm/yr: "
                  + ", ".join(f"{sid} ({v:.0f} mm)" for sid, v in dropped.items()))
        keep_ids = set(tot[(tot <= hi) & (tot >= lo)].index)
        combined = combined[combined["station_id"].isin(keep_ids)]

    combined.sort_values(["timestamp", "station_id"], inplace=True)
    long_path = os.path.join(args.outdir, f"{args.region}_rainfall_long.csv")
    combined.to_csv(long_path, index=False)

    # metadata: NO header, columns name, station_id, network, latitude, longitude
    meta = pd.DataFrame(kept_meta)
    if keep_ids is not None:
        meta = meta[meta["station_id"].isin(keep_ids)]
    meta_out = pd.DataFrame({
        "name": meta["name"],
        "station_id": meta["station_id"],
        "network": "unknown",  # not provided by GetFeatureOfInterest; unused in training
        "latitude": meta["latitude"],
        "longitude": meta["longitude"],
    })
    meta_path = os.path.join(args.outdir, f"{args.region}_station_metadata.csv")
    meta_out.to_csv(meta_path, header=False, index=False)

    print(f"✓ rainfall  -> {long_path}")
    print(f"  rows {len(combined):,} | stations {combined['station_id'].nunique()} | "
          f"{combined['timestamp'].min()} .. {combined['timestamp'].max()}")
    print(f"  total rainfall {combined['rainfall_mm'].sum():,.1f} mm | "
          f"non-zero hours {(combined['rainfall_mm'] > 0).sum():,}")
    print(f"✓ metadata  -> {meta_path} ({len(meta_out)} stations)")
    print("\nTo train on this region, point config dataset_parameters at:")
    print(f"    rainfall_file: '{long_path}'")
    print(f"    metadata_file: '{meta_path}'")
    print("Done.")


if __name__ == "__main__":
    main()
