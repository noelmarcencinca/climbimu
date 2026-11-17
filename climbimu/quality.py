"""Quality report of a raw recording: sampling, gaps, spikes and angle jumps.

Run it on the raw data, before any cleaning. It tells you what the
cleaning will have to fix, and it gives numbers you can report in the
methods section of a paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .preprocess import find_spikes
from .recording import Recording


@dataclass
class QualityReport:
    id: str
    n_samples: int
    duration_s: float
    median_interval_ms: float
    nominal_fs_hz: float
    min_interval_ms: float
    max_interval_ms: float
    n_gaps: int
    missing_time_s: float
    gap_times_s: list = field(default_factory=list)
    spikes: dict = field(default_factory=dict)
    angle_wraps: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = [
            f"Quality report: {self.id}",
            f"  Samples: {self.n_samples}   Duration: {self.duration_s:.2f} s",
            f"  Sampling: median interval {self.median_interval_ms:.1f} ms "
            f"(~{self.nominal_fs_hz:.0f} Hz), range "
            f"{self.min_interval_ms:.1f}-{self.max_interval_ms:.1f} ms",
            f"  Gaps: {self.n_gaps} (missing time {self.missing_time_s * 1000:.0f} ms, "
            f"{100 * self.missing_time_s / max(self.duration_s, 1e-9):.1f}% of the recording)",
            "  Spikes: " + (", ".join(f"{k} {v}" for k, v in self.spikes.items()) or "none"),
            "  Angle jumps (0/360): "
            + (", ".join(f"{k} {v}" for k, v in self.angle_wraps.items()) or "no angles"),
        ]
        if self.warnings:
            lines.append("  Warnings:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def check_quality(rec: Recording, config: dict | None = None) -> QualityReport:
    """Describe the quality of a raw recording."""
    config = config or {}
    gap_factor = config.get("quality", {}).get("gap_factor", 3.0)
    spike_cfg = config.get("preprocessing", {}).get("despike", {})

    t = rec.data["time"].to_numpy(dtype=float)
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    is_gap = dt > gap_factor * median_dt
    missing = float(np.sum(dt[is_gap] - median_dt))

    spikes = {}
    for col in rec.acc_columns():
        bad = find_spikes(rec.data[col].to_numpy(dtype=float),
                          spike_cfg.get("window", 9), spike_cfg.get("n_sigmas", 4.0),
                          spike_cfg.get("min_threshold_g", 0.5))
        spikes[col] = int(bad.sum())

    wraps = {}
    for col in rec.angle_columns():
        jumps = np.abs(np.diff(rec.data[col].to_numpy(dtype=float)))
        wraps[col] = int(np.sum(jumps > 180))

    warnings = []
    jitter = (dt[~is_gap].max() - dt[~is_gap].min()) / median_dt if (~is_gap).any() else 0
    if jitter > 0.2:
        warnings.append("Irregular sampling interval: resampling is needed before "
                        "computing entropy.")
    if is_gap.any():
        warnings.append(f"{int(is_gap.sum())} gaps longer than {gap_factor:g}x the "
                        "normal interval: they will be filled by interpolation.")
        if is_gap.sum() >= 5:
            intervals = np.diff(t[1:][is_gap])
            if intervals.std() < 0.1 * intervals.mean():
                warnings.append(f"Gaps appear at regular intervals (~{intervals.mean():.2f} s). "
                                "This often comes from the device (data packets), "
                                "not from the movement.")
    if sum(spikes.values()) > 0:
        warnings.append("Isolated acceleration spikes found: remove them before "
                        "computing variability or entropy.")
    if any(v > 0 for v in wraps.values()):
        warnings.append("Angles jump between 0 and 360 degrees: unwrap them before "
                        "analysis.")

    return QualityReport(
        id=rec.id,
        n_samples=len(t),
        duration_s=float(t[-1] - t[0]),
        median_interval_ms=median_dt * 1000,
        nominal_fs_hz=1.0 / median_dt,
        min_interval_ms=float(dt.min() * 1000),
        max_interval_ms=float(dt.max() * 1000),
        n_gaps=int(is_gap.sum()),
        missing_time_s=missing,
        gap_times_s=[round(float(x), 3) for x in t[1:][is_gap]],
        spikes=spikes,
        angle_wraps=wraps,
        warnings=warnings,
    )
