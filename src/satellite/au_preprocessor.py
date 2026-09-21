from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import pandas as pd
from pyproj import Proj

# Geostationary projection for Himawari-8/9 centred at 140.7°E
_PROJ = Proj(
    proj='geos', lon_0=140.7, h=35785863,
    x_0=0, y_0=0, a=6378137, b=6356752.3, units='m',
)

# Sydney bbox row/col indices on the 5500×5500 geostationary grid (2 km pixels)
# Derived from dataset exploration: covers ~150.82–151.62°E, ~34.14–33.62°S
_R0, _R1 = 4445, 4466
_C0, _C1 = 3202, 3234

_VERSION   = "v2.1"
_FILL      = 65535
_SCALE     = 0.1


def _file_path(base: Path, dt: datetime) -> Path:
    return (
        base / _VERSION
        / dt.strftime("%Y/%m/%d")
        / f"S_NWC_CRRPh_HIMA08_HIMA-N-NR_{dt.strftime('%Y%m%dT%H%M%S')}Z.nc"
    )


def _read_accum(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read crrph_accum crop from one file.

    Returns (accum, valid) both of shape (H*W,) float32, or None if file missing.
      accum : rain accumulation in mm (fill → 0)
      valid : 1.0 where retrieval succeeded, 0.0 where fill value was present
    """
    if not path.exists():
        return None
    with nc4.Dataset(path) as ds:
        raw = np.array(
            ds.variables['crrph_accum'][_R0:_R1 + 1, _C0:_C1 + 1],
            dtype=np.float32,
        )
    valid = (raw != _FILL).astype(np.float32).flatten()
    accum = np.where(raw == _FILL, 0.0, raw * _SCALE).flatten()
    return accum, valid


class AUSatellitePreprocessor:
    """
    Loads Himawari-8/9 CRRPH data from NCI rv74 and produces hourly
    accumulations over the Sydney domain.

    Aggregation: crrph_accum at H:00:00 UTC is the product's own trailing
    1-hour accumulation, i.e. rain over [H-1:00, H:00). We relabel it by the
    START of that window (H-1) so the convention matches the gauge (gauge hour
    H = rain during [H:00, H+1:00)), keeping gauge/satellite windows aligned at
    the same timestamp.
    Missing :00 files are treated as zero (no detected convective rain).

    Parameters
    ----------
    base_path : path to the CRRPH root, i.e.
                /g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph
    """

    def __init__(
        self,
        base_path: str = (
            "/g/data/rv74/satellite-products/arc/der/himawari-ahi/precip/crrph"
        ),
    ):
        self.base_path = Path(base_path)
        self.grid_shape = (
            len(range(_R0, _R1 + 1)),
            len(range(_C0, _C1 + 1)),
        )
        self.n_nodes = self.grid_shape[0] * self.grid_shape[1]
        self.lat, self.lon = self._compute_coords()

    def _compute_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute flat (n_nodes,) lat/lon for the Sydney crop from the fixed grid."""
        # Himawari CRRPH grid: 5500 pixels, 2 km spacing, centred on sub-satellite point.
        # Pixel centres (in metres) derived from dataset exploration.
        nx_full = np.linspace(-5498995.5, 5498995.0, 5500, dtype=np.float64)
        ny_full = np.linspace(5498995.5, -5498995.0, 5500, dtype=np.float64)
        x_crop = nx_full[_C0:_C1 + 1]
        y_crop = ny_full[_R0:_R1 + 1]
        xx, yy = np.meshgrid(x_crop, y_crop)
        lon2d, lat2d = _PROJ(xx, yy, inverse=True)
        return lat2d.flatten().astype(np.float32), lon2d.flatten().astype(np.float32)

    def get_node_coords(self) -> pd.DataFrame:
        return pd.DataFrame({"latitude": self.lat, "longitude": self.lon})

    def load_day_hourly(self, date: datetime) -> pd.DataFrame:
        """
        Load one day and return 24 hourly records with two feature channels:
          data[:,0] : crrph_accum (mm), fill → 0
          data[:,1] : valid flag (1.0 = retrieval succeeded, 0.0 = fill value)

        Start-of-period labelling: the file at H:00:00 UTC is the trailing
        accumulation over [H-1:00, H:00), so it is labelled H-1.
          - Hours 00:00–22:00: from files 01:00:00–23:00:00 UTC of this day.
          - Hour 23:00: window [23:00, 24:00) from the 00:00:00 file of date+1.
        """
        zero_accum = np.zeros(self.n_nodes, dtype=np.float32)
        zero_valid = np.zeros(self.n_nodes, dtype=np.float32)
        records = []

        for h in range(1, 24):
            dt = datetime(date.year, date.month, date.day, h, tzinfo=timezone.utc)
            result = _read_accum(_file_path(self.base_path, dt))
            if result is not None:
                accum, valid = result
            else:
                accum, valid = zero_accum, zero_valid
            records.append({
                # file at H:00 covers [H-1, H); label by window start (H-1)
                "timestamp": dt - timedelta(hours=1),
                "data": np.stack([accum, valid], axis=1),  # (n_nodes, 2)
            })

        next_day = date + timedelta(days=1)
        dt_mid = datetime(next_day.year, next_day.month, next_day.day, 0, tzinfo=timezone.utc)
        result = _read_accum(_file_path(self.base_path, dt_mid))
        if result is not None:
            accum, valid = result
        else:
            accum, valid = zero_accum, zero_valid
        records.append({
            # next-day 00:00 file covers [23:00, 24:00) of `date`; label = date 23:00
            "timestamp": dt_mid - timedelta(hours=1),
            "data": np.stack([accum, valid], axis=1),  # (n_nodes, 2)
        })

        return pd.DataFrame(records)

    def build_dataset(self, year: int) -> pd.DataFrame:
        """Build the full hourly satellite dataset for one year."""
        days = []
        d = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        while d < end:
            days.append(d)
            d += timedelta(days=1)

        print(
            f"Satellite CRRPH / {year}: {len(days)} days, "
            f"grid={self.grid_shape}, n_nodes={self.n_nodes}"
        )
        dfs = []
        for i, dt in enumerate(days):
            dfs.append(self.load_day_hourly(dt))
            if (i + 1) % 30 == 0:
                print(f"  {i + 1}/{len(days)} days")

        df = (
            pd.concat(dfs, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        print(f"Done. {len(df)} hourly timesteps.")
        return df

    def build_dataset_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Build the hourly satellite dataset for a range of years."""
        dfs = [self.build_dataset(y) for y in range(start_year, end_year + 1)]
        df = (
            pd.concat(dfs, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        print(f"Total across {start_year}–{end_year}: {len(df)} hourly timesteps.")
        return df
