import pandas as pd
from datetime import datetime


def get_station_coordinate_mappings():
    """Stub — only used by Singapore visualisation code, not called in AU training."""
    raise NotImplementedError("get_station_coordinate_mappings is not available for the Australian dataset")


def load_raingauge_dataset(
    rainfall_file: str,
    metadata_file: str,
    start: int = None,
    end: int = None,
    uptime_threshold: float = 0.9,
) -> tuple:
    """
    Load Australian raingauge dataset from a single pre-merged CSV file.

    Parameters
    ----------
    rainfall_file : str
        Path to all_stations_rainfall_hourly_combined.csv
        Expected columns: timestamp, station_id, rainfall_mm
    metadata_file : str
        Path to database/australia/station_metadata.csv
        No header; columns: name, station_id, network, latitude, longitude
    start : int, optional
        Filter to keep only rows with year >= start
    end : int, optional
        Filter to keep only rows with year <= end
    uptime_threshold : float
        Stations with fraction of valid readings below this value are dropped

    Returns
    -------
    formatted_gauge_df : pd.DataFrame
        Pivoted dataframe — index: timestamp, columns: station_id (str)
    raingauge_mappings_df : pd.DataFrame
        Station metadata with columns: id, latitude, longitude, order
    """

    print(f"Loading Australian raingauge data from {rainfall_file}")
    gauge_df = pd.read_csv(rainfall_file)

    # Parse timestamps — format: '2021-01-01 00:00:00'
    gauge_df["timestamp"] = pd.to_datetime(gauge_df["timestamp"])

    # Optionally restrict to a year range
    if start is not None:
        gauge_df = gauge_df[gauge_df["timestamp"].dt.year >= start]
    if end is not None:
        gauge_df = gauge_df[gauge_df["timestamp"].dt.year <= end]

    # Normalise station_id to string so it matches metadata
    gauge_df["station_id"] = gauge_df["station_id"].astype(str)

    # Data is already hourly mm — no multiplication needed (Singapore * 12 was for 5-min data)
    formatted_gauge_df = gauge_df.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    )
    print(f"Pivoted dataframe shape (before uptime filter): {formatted_gauge_df.shape}")

    # Load station metadata (no header row in this file)
    print(f"Loading station metadata from {metadata_file}")
    meta_df = pd.read_csv(
        metadata_file,
        header=None,
        names=["name", "id", "network", "latitude", "longitude"],
    )
    # Keep only stations that appear in the rainfall file
    meta_df["id"] = meta_df["id"].astype(str)
    meta_df = meta_df[meta_df["id"].isin(formatted_gauge_df.columns)].copy()
    meta_df = meta_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Apply uptime filter
    if uptime_threshold is not None:
        print(f"Filtering by uptime threshold = {uptime_threshold}")
        kept = filter_uptime(formatted_gauge_df, uptime_threshold=uptime_threshold)
        formatted_gauge_df = formatted_gauge_df[kept.index]
        meta_df = meta_df[meta_df["id"].isin(kept.index)].reset_index(drop=True)

    print(f"Dataframe shape after filter: {formatted_gauge_df.shape}")
    print(f"Stations retained: {len(meta_df)}")

    meta_df["order"] = range(len(meta_df))
    meta_df.reset_index(drop=True, inplace=True)

    return formatted_gauge_df, meta_df


def filter_uptime(raingauge_df: pd.DataFrame, uptime_threshold: float = 0.9) -> pd.Series:
    """
    Returns a Series (column name → uptime fraction) for stations above the threshold.
    """
    uptime = raingauge_df.notna().sum() / len(raingauge_df)
    return uptime[uptime >= uptime_threshold]
