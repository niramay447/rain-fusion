import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import netCDF4 as nc4
import numpy as np
import pandas as pd
from pyproj import Proj

# ─────────────────────────────────────────────────────────────────────────────
# WAGGA WAGGA (radar 55) variant of au_preprocessor.py.
# Identical logic to the Sydney (radar 71) preprocessor — only the four
# radar-specific constants differ: radar_id, _PROJ, crop indices, and (via
# config) stride. The Sydney file is untouched; train.py imports one or the
# other by config. Timezone handling is unchanged (Wagga is in NSW → same
# Australia/Sydney zone; radar timestamps stay UTC; start-of-period hour
# convention preserved).
# ─────────────────────────────────────────────────────────────────────────────

# Albers Equal Area projection centred on radar 55 (Wagga Wagga)
# Parameters read directly from the prcp-m15 NetCDF proj variable:
#   standard_parallel = [-33.6, -36.7], lon_0 = 147.467, lat_0 = -35.167
_PROJ = Proj(
    proj="aea",
    lat_1=-33.6, lat_2=-36.7,
    lon_0=147.467, lat_0=-35.167,
    x_0=0, y_0=0,
    a=6378137.0, b=6356752.31414,
)

# Full 512×512 grid. Unlike Sydney (a metro sub-crop of radar 71), the Wagga
# rural study region spans essentially the whole radar grid (~±127.75 km), so we
# use the full extent. Gauges sit within ~120 km of the radar; the few grid
# corner cells beyond radar range carry no rain and connect to no gauge.
_R0, _R1 = 0, 511
_C0, _C1 = 0, 511


def _zip_path(base: Path, radar_id: int, date: datetime) -> Path:
    return (base / str(radar_id) / date.strftime("%Y") / "prcp-m15" /
            f"{radar_id}_{date.strftime('%Y%m%d')}.prcp-m15.zip")


def _read_nc_precip(raw: bytes, stride: int) -> np.ndarray:
    """Read precipitation from raw nc bytes, crop and subsample. Returns (H, W) float32."""
    with nc4.Dataset("mem", memory=raw) as ds:
        data = np.array(ds.variables["precipitation"][:], dtype=np.float32)
    return data[_R0:_R1 + 1:stride, _C0:_C1 + 1:stride]


def _load_day_arrays(zip_path: Path, stride: int) -> dict[str, np.ndarray]:
    """
    Open one daily zip and return a dict mapping 'HHMMSS' → (H, W) array.
    Missing files are absent from the dict; callers should treat them as zeros.
    """
    if not zip_path.exists():
        return {}
    result = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".nc"):
                continue
            # filename: {id}_{YYYYMMDD}_{HHMMSS}.prcp-m15.nc
            hhmmss = name.split("_")[2].split(".")[0]
            result[hhmmss] = _read_nc_precip(zf.read(name), stride)
    return result


class AURadarPreprocessor:
    """
    Loads BOM Rainfields3 prcp-m15 data for radar 55 (Wagga Wagga),
    aggregates 15-min accumulations to hourly, over the full radar grid.

    Parameters
    ----------
    base_path : path to /g/data/rq0/rainfields3
    radar_id  : BOM radar station ID (default 55 = Wagga Wagga)
    stride    : spatial subsampling factor on the 0.5-km grid.
                stride=10 → 5 km resolution, full 512 grid → ~52×52 = 2704 nodes
                (matches Sydney's 5 km resolution; raise to 15 → ~1225 nodes if
                runtime is a concern).
    """

    def __init__(
        self,
        base_path: str = "/g/data/rq0/rainfields3",
        radar_id: int = 55,
        stride: int = 10,
    ):
        self.base_path = Path(base_path)
        self.radar_id  = radar_id
        self.stride    = stride
        self.lat, self.lon = self._compute_coords()
        self.n_nodes = len(self.lat)
        self.grid_shape = (
            len(range(_R0, _R1 + 1, stride)),
            len(range(_C0, _C1 + 1, stride)),
        )

    def _compute_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute flat (n_nodes,) lat/lon arrays for the subsampled full grid."""
        yr_dir = self.base_path / str(self.radar_id) / "2022" / "prcp-m15"
        sample_zip = sorted(yr_dir.glob("*.zip"))[0]
        with zipfile.ZipFile(sample_zip) as zf:
            nc_name = sorted(n for n in zf.namelist() if n.endswith(".nc"))[0]
            with nc4.Dataset("mem", memory=zf.read(nc_name)) as ds:
                x_km = np.array(ds.variables["x"][:])
                y_km = np.array(ds.variables["y"][:])

        x_crop = x_km[_C0:_C1 + 1:self.stride] * 1000.0
        y_crop = y_km[_R0:_R1 + 1:self.stride] * 1000.0
        xx, yy = np.meshgrid(x_crop, y_crop)
        lon2d, lat2d = _PROJ(xx, yy, inverse=True)
        return lat2d.flatten(), lon2d.flatten()

    def get_node_coords(self) -> pd.DataFrame:
        """Return DataFrame with latitude and longitude for each radar node."""
        return pd.DataFrame({"latitude": self.lat, "longitude": self.lon})

    def load_day_hourly(
        self,
        date: datetime,
        next_day_arrays: Optional[dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Load one day and aggregate 15-min → 24 hourly accumulations.

        Start-of-period labelling: the hour labelled H covers [H:00, H+1:00) UTC,
        matching the gauge convention (gauge hour H = rain during [H:00, H+1:00)).
        This keeps gauge and radar accumulation windows aligned at the same
        timestamp (see also the satellite preprocessor).
          - Hours 00:00–22:00: all four 15-min files are within this day's zip
          - Hour 23:00: window [23:00, 24:00) uses files 23:15–23:45 from this
            day plus the 00:00 file from next_day_arrays (zeros if unavailable)

        Returns DataFrame with columns: timestamp (UTC), data (n_nodes,)
        """
        arrays = _load_day_arrays(_zip_path(self.base_path, self.radar_id, date), self.stride)
        nd     = next_day_arrays or {}

        # Shape of one grid cell (used to build zero arrays for missing files)
        h_nodes = len(range(_R0, _R1 + 1, self.stride))
        w_nodes = len(range(_C0, _C1 + 1, self.stride))
        zero = np.zeros((h_nodes, w_nodes), dtype=np.float32)

        records = []

        # Window [H:00, H+1:00) is summed from the 15-min files at H:15, H:30,
        # H:45, H+1:00 and labelled by its START hour H (start-of-period).
        # Hours 00:00–22:00: all four 15-min files are within this day's zip.
        for h in range(0, 23):
            slots = (
                [f"{h:02d}{m:02d}00" for m in (15, 30, 45)]
                + [f"{h + 1:02d}0000"]
            )
            hourly = sum(arrays.get(s, zero) for s in slots)
            ts = datetime(date.year, date.month, date.day, h, tzinfo=timezone.utc)
            records.append({"timestamp": ts, "data": hourly.flatten()})

        # Hour 23:00 — window [23:00, 24:00) straddles two zips: 23:15/23:30/23:45
        # from this day plus the 00:00 file from the next day.
        slots_mid = ["231500", "233000", "234500"]
        hourly_mid = (
            sum(arrays.get(s, zero) for s in slots_mid)
            + nd.get("000000", zero)
        )
        ts_midnight = datetime(date.year, date.month, date.day, 23, tzinfo=timezone.utc)
        records.append({"timestamp": ts_midnight, "data": hourly_mid.flatten()})

        return pd.DataFrame(records)

    def build_dataset(self, year: int) -> pd.DataFrame:
        """
        Build the full hourly radar dataset for one year.

        Returns DataFrame with columns:
          timestamp : pd.Timestamp, UTC, hourly, start-of-period
          data      : np.ndarray of shape (n_nodes,), mm of accumulation
        """
        yr_dir   = self.base_path / str(self.radar_id) / str(year) / "prcp-m15"
        all_zips = sorted(yr_dir.glob("*.zip"))

        print(f"Radar {self.radar_id} / {year}: {len(all_zips)} days, "
              f"stride={self.stride}, n_nodes={self.n_nodes}")

        dfs = []
        for i, zip_path in enumerate(all_zips):
            date_str = zip_path.stem.split("_")[1].split(".")[0]
            date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)

            # Load the 00:00 file from the next day for midnight stitching
            next_zip = _zip_path(self.base_path, self.radar_id, date + timedelta(days=1))
            nd_arrays: dict[str, np.ndarray] = {}
            if next_zip.exists():
                with zipfile.ZipFile(next_zip) as zf:
                    names = {n.split("_")[2].split(".")[0]: n
                             for n in zf.namelist() if n.endswith(".nc")}
                    if "000000" in names:
                        nd_arrays["000000"] = _read_nc_precip(
                            zf.read(names["000000"]), self.stride
                        )

            dfs.append(self.load_day_hourly(date, nd_arrays))

            if (i + 1) % 30 == 0:
                print(f"  {i + 1}/{len(all_zips)} days")

        df = (pd.concat(dfs, ignore_index=True)
              .sort_values("timestamp")
              .reset_index(drop=True))
        print(f"Done. {len(df)} hourly timesteps.")
        return df

    def build_dataset_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Build the hourly radar dataset for a range of years and concatenate."""
        dfs = []
        for year in range(start_year, end_year + 1):
            yr_dir = self.base_path / str(self.radar_id) / str(year) / "prcp-m15"
            if not yr_dir.exists():
                print(f"  Skipping {year}: no data at {yr_dir}")
                continue
            dfs.append(self.build_dataset(year))
        df = (pd.concat(dfs, ignore_index=True)
              .sort_values("timestamp")
              .reset_index(drop=True))
        print(f"Total across {start_year}–{end_year}: {len(df)} hourly timesteps.")
        return df
