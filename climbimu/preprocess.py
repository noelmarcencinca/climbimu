"""Signal cleaning: spikes, angle wrapping, resampling and filtering.

Recommended order (used by :func:`preprocess`):

1. ``despike`` (acceleration) -- before resampling, so spikes are not spread
   by interpolation.
2. ``unwrap_angles`` -- before resampling, so interpolation does not cross
   the 0/360 degree jump. Angle spikes are removed right after unwrapping.
3. ``resample`` -- to a fixed sampling frequency. Entropy measures assume
   equally spaced samples.
4. ``lowpass`` -- optional. Filtering changes entropy values, so it is off
   by default and, if used, it must be reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .recording import Recording


# ---------------------------------------------------------------- spikes

def find_spikes(x, window: int = 9, n_sigmas: float = 4.0,
                min_threshold: float = 0.5) -> np.ndarray:
    """Find isolated spikes with a Hampel filter.

    A sample is a spike when it is far from the median of its neighbours:
    more than ``n_sigmas`` robust standard deviations (MAD) and more than
    ``min_threshold`` (in the units of the signal, g for acceleration).
    The minimum threshold avoids marking normal small variations as spikes
    in very quiet parts of the signal.

    Returns a boolean array (True = spike).
    """
    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = np.maximum(n_sigmas * 1.4826 * mad, min_threshold)
    return ((s - med).abs() > threshold).to_numpy()


def despike(rec: Recording, columns=None, window: int = 9, n_sigmas: float = 4.0,
            min_threshold: float = 0.5, max_iterations: int = 5) -> Recording:
    """Replace spikes with the local median. Repeats until no spikes remain.

    Repeating is needed because some recordings have bursts of alternating
    spikes (bad, good, bad, good...). One pass leaves some of them.
    ``min_threshold`` is in the units of the columns (g for acceleration,
    degrees for angles). By default the acceleration columns are cleaned.
    """
    out = rec.copy()
    columns = columns or out.acc_columns()
    counts = {}
    for col in columns:
        x = out.data[col].to_numpy(dtype=float).copy()
        total = 0
        for _ in range(max_iterations):
            bad = find_spikes(x, window, n_sigmas, min_threshold)
            if not bad.any():
                break
            med = pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()
            x[bad] = med[bad]
            total += int(bad.sum())
        out.data[col] = x
        counts[col] = total
    out.add_log("despike", method="Hampel filter (rolling median + MAD), iterative",
                columns=list(columns), window=window, n_sigmas=n_sigmas,
                min_threshold=min_threshold,
                max_iterations=max_iterations, replaced=counts)
    return out


# ---------------------------------------------------------------- angles

def unwrap_angles(rec: Recording, columns=None) -> Recording:
    """Remove the artificial 360 degree jumps in angle signals."""
    out = rec.copy()
    columns = columns or out.angle_columns()
    for col in columns:
        rad = np.deg2rad(out.data[col].to_numpy(dtype=float))
        out.data[col] = np.rad2deg(np.unwrap(rad))
    out.add_log("unwrap_angles", columns=list(columns))
    return out


# ---------------------------------------------------------------- resampling

def resample(rec: Recording, fs: float = 100.0) -> Recording:
    """Linear interpolation to a fixed sampling frequency ``fs`` (Hz).

    This also fills the gaps where the device lost samples. Check the
    quality report to know how long those gaps are.
    """
    out = rec.copy()
    t = out.data["time"].to_numpy(dtype=float)
    new_t = np.arange(t[0], t[-1], 1.0 / fs)
    new = {"time": new_t}
    for col in out.data.columns:
        if col != "time":
            new[col] = np.interp(new_t, t, out.data[col].to_numpy(dtype=float))
    out.data = pd.DataFrame(new)
    out.fs = float(fs)
    out.add_log("resample", method="linear interpolation", fs_hz=float(fs),
                samples_before=len(t), samples_after=len(new_t))
    return out


# ---------------------------------------------------------------- filtering

def lowpass(rec: Recording, cutoff_hz: float = 10.0, order: int = 4,
            columns=None) -> Recording:
    """Zero-lag Butterworth low-pass filter. Needs a resampled recording."""
    if rec.fs is None:
        raise ValueError("Resample the recording before filtering.")
    out = rec.copy()
    columns = columns or out.acc_columns()
    b, a = butter(order, cutoff_hz / (out.fs / 2.0), btype="low")
    for col in columns:
        out.data[col] = filtfilt(b, a, out.data[col].to_numpy(dtype=float))
    out.add_log("lowpass", method="Butterworth, zero-lag (filtfilt)",
                cutoff_hz=cutoff_hz, order=order, columns=list(columns))
    return out


# ---------------------------------------------------------------- helpers

def add_acc_norm(rec: Recording) -> Recording:
    """Add the acceleration magnitude (``acc_norm``, in g).

    The magnitude does not depend on how the sensor is oriented, which
    makes it useful when sensor placement varies between climbers.
    """
    out = rec.copy()
    acc = out.data[out.acc_columns()].to_numpy(dtype=float)
    out.data["acc_norm"] = np.sqrt((acc ** 2).sum(axis=1))
    out.add_log("add_acc_norm")
    return out


def preprocess(rec: Recording, config: dict) -> Recording:
    """Run the full cleaning pipeline following the configuration."""
    steps = config.get("preprocessing", {})
    out = rec
    d = steps.get("despike", {})
    if d.get("enabled", True):
        out = despike(out, columns=out.acc_columns(), window=d.get("window", 9),
                      n_sigmas=d.get("n_sigmas", 4.0),
                      min_threshold=d.get("min_threshold_g", 0.5),
                      max_iterations=d.get("max_iterations", 5))
    if steps.get("unwrap_angles", {}).get("enabled", True) and out.angle_columns():
        out = unwrap_angles(out)
        # Angle spikes are cleaned after unwrapping, otherwise the 0/360
        # jumps would look like spikes.
        if d.get("enabled", True) and d.get("min_threshold_deg") is not None:
            out = despike(out, columns=out.angle_columns(), window=d.get("window", 9),
                          n_sigmas=d.get("n_sigmas", 4.0),
                          min_threshold=d["min_threshold_deg"],
                          max_iterations=d.get("max_iterations", 5))
    if steps.get("resample", {}).get("enabled", True):
        out = resample(out, fs=steps["resample"].get("fs", 100.0))
    if steps.get("lowpass", {}).get("enabled", False):
        p = {k: v for k, v in steps["lowpass"].items() if k != "enabled"}
        out = lowpass(out, **p)
    out = add_acc_norm(out)
    return out
