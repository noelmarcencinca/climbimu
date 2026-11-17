"""The Recording object: one IMU file, its labels and its processing history."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class Recording:
    """One IMU recording.

    Attributes
    ----------
    data : pandas.DataFrame
        Signal table. Always has a ``time`` column in seconds. Acceleration
        columns (``acc_x``, ``acc_y``, ``acc_z``) are in g. Angle columns
        (``yaw``, ``pitch``, ``roll``), if present, are in degrees.
    meta : dict
        Labels for this recording (for example climber, attempt, boulder).
        The tool does not assume any fixed label: you define them.
    source : str
        Path of the original file.
    fs : float or None
        Sampling frequency in Hz once the signal has been resampled.
        ``None`` means the sampling is still the original (irregular) one.
    log : list of dict
        Every processing step applied, with its parameters. This is used
        to report exactly what was done to the data.
    """

    data: pd.DataFrame
    meta: dict = field(default_factory=dict)
    source: str = ""
    fs: float | None = None
    log: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return str(self.meta.get("id", ""))

    @property
    def duration(self) -> float:
        t = self.data["time"]
        return float(t.iloc[-1] - t.iloc[0])

    def acc_columns(self) -> list[str]:
        return [c for c in ("acc_x", "acc_y", "acc_z") if c in self.data]

    def angle_columns(self) -> list[str]:
        return [c for c in ("yaw", "pitch", "roll") if c in self.data]

    def add_log(self, step: str, **params) -> None:
        self.log.append(
            {"step": step, "time": datetime.now().isoformat(timespec="seconds"), **params}
        )

    def copy(self) -> "Recording":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        fs = f"{self.fs:g} Hz" if self.fs else "irregular sampling"
        return (
            f"Recording(id={self.id!r}, samples={len(self.data)}, "
            f"duration={self.duration:.2f} s, {fs}, steps={len(self.log)})"
        )
