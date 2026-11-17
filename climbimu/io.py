"""Reading IMU files and configuration.

Each IMU exports its data in its own way. Instead of writing a reader for
every device, a small YAML file says which column is which and in which
units it is. Changing device means changing the configuration, not the code.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from .metadata import get_metadata
from .recording import Recording

G = 9.80665  # standard gravity, m/s^2

DEFAULT_CONFIG = {
    "columns": {
        "time": "time",
        "acc_x": "acc_x",
        "acc_y": "acc_y",
        "acc_z": "acc_z",
        "yaw": "yaw",
        "pitch": "pitch",
        "roll": "roll",
    },
    "units": {"time": "s", "acc": "g", "angles": "deg"},
    "sheet": 0,
    "metadata": {"filename_pattern": None, "table": None, "id_column": "file"},
    "quality": {"gap_factor": 3.0},
    "segmentation": {
        "reference_window_s": 0.5, "gravity_cutoff_hz": 1.0,
        "swing_threshold_deg": 4.0, "climb_threshold_deg": 20.0,
        "climb_min_duration_s": 1.0, "unloading_threshold_g": 0.8,
        "unloading_min_duration_s": 0.15, "landing_search_s": 1.0,
        "events_table": None,
    },
    "preprocessing": {
        "despike": {"enabled": True, "window": 9, "n_sigmas": 4.0,
                    "min_threshold_g": 0.5, "min_threshold_deg": 10.0,
                    "max_iterations": 5},
        "unwrap_angles": {"enabled": True},
        "resample": {"enabled": True, "fs": 100.0},
        "lowpass": {"enabled": False, "cutoff_hz": 10.0, "order": 4},
    },
}

REQUIRED = ("time", "acc_x", "acc_y", "acc_z")


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML configuration. Missing values take the defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _deep_update(config, user)
    return config


def load_recording(path: str | Path, config: dict | None = None) -> Recording:
    """Read one IMU file (Excel or CSV) into a :class:`Recording`."""
    config = config or load_config()
    path = Path(path)

    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, sheet_name=config.get("sheet", 0))
    else:
        raw = pd.read_csv(path, sep=None, engine="python")

    # Rename the device columns to the standard names used by the tool.
    mapping = {}
    for standard, original in config["columns"].items():
        if original is None:
            continue
        if original in raw.columns:
            mapping[original] = standard
        elif standard in REQUIRED:
            raise ValueError(
                f"Column '{original}' (for '{standard}') not found in {path.name}. "
                f"Available columns: {list(raw.columns)}"
            )
    data = raw[list(mapping)].rename(columns=mapping)
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=["time"])
    data = data.sort_values("time").reset_index(drop=True)

    units = config["units"]
    if units.get("time") == "ms":
        data["time"] = data["time"] / 1000.0
    if units.get("acc") in ("m/s2", "m/s^2", "ms2"):
        for col in ("acc_x", "acc_y", "acc_z"):
            data[col] = data[col] / G
    if units.get("angles") == "rad":
        for col in ("yaw", "pitch", "roll"):
            if col in data:
                data[col] = data[col] * 180.0 / 3.141592653589793

    rec = Recording(data=data, meta=get_metadata(path, config), source=str(path))
    rec.add_log("load", file=path.name, columns=mapping, units=units)
    return rec


def save_recording(rec: Recording, path: str | Path) -> None:
    """Save the signal table as CSV and the processing log as YAML next to it."""
    path = Path(path)
    rec.data.to_csv(path, index=False)
    with open(path.with_suffix(".log.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": _plain(rec.meta), "fs": rec.fs, "log": _plain(rec.log)},
                       f, allow_unicode=True, sort_keys=False)


def _deep_update(base: dict, new: dict) -> None:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _plain(obj):
    """Convert numpy types to plain Python so YAML can save them."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj
