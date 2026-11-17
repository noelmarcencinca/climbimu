"""Segmentation of a climbing attempt: start movement, climb and descent.

Protocol assumed (it can change in future studies):

1. The climber stands still for a moment (reference posture).
2. The climber makes a hip flexion-extension swing (synchronisation movement).
3. The climber climbs the boulder.
4. The climber comes down from the top (down-climbing, jumping or dropping).

Events (times in seconds, in the time base of the recording):

``swing_start``  start of the synchronisation swing
``climb_start``  start of the climb (the trunk leans towards the wall)
``climb_end``    end of the climb = start of the descent from the top
``landing``      landing on the ground

Segments:

``swing``    swing_start -> climb_start
``climb``    climb_start -> climb_end
``descent``  climb_end   -> landing

Detection is semi-automatic. :func:`detect_events` proposes the events and
:func:`plot_events` shows them. You review the figure and, if needed, correct
the times in the events table (CSV, it opens in Excel). Rows marked as checked
are always used instead of the automatic detection.

Two signals are used, both independent of how the sensor is oriented on the
body, and neither depends on the orientation angles of the device:

- **Trunk tilt**: angle (deg) between the gravity direction measured by the
  sensor at each moment and during the initial quiet stance. Standing gives
  values near 0; climbing gives large values because the trunk leans.
- **Acceleration magnitude** (g): below 1 g when the body is unloaded,
  for example when the climber lets go at the top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .recording import Recording

EVENTS = ("swing_start", "climb_start", "climb_end", "landing")
SEGMENTS = {
    "swing": ("swing_start", "climb_start"),
    "climb": ("climb_start", "climb_end"),
    "descent": ("climb_end", "landing"),
}

DEFAULTS = {
    "reference_window_s": 0.5,
    "gravity_cutoff_hz": 1.0,
    "swing_threshold_deg": 4.0,
    "climb_threshold_deg": 20.0,
    "climb_min_duration_s": 1.0,
    "unloading_threshold_g": 0.8,
    "unloading_min_duration_s": 0.15,
    "landing_search_s": 1.0,
    "events_table": None,
}


@dataclass
class Events:
    """Event times of one recording and where they come from."""

    times: dict = field(default_factory=dict)   # event name -> time (s) or None
    source: str = "auto"                        # "auto" or "table"
    checked: bool = False
    notes: list = field(default_factory=list)

    def __getitem__(self, name):
        return self.times.get(name)

    def missing(self) -> list:
        return [e for e in EVENTS if self.times.get(e) is None]


# ---------------------------------------------------------------- signals

def _settings(config: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update((config or {}).get("segmentation", {}) or {})
    return s


def _lowpass(x, fs, cutoff, order=2):
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x, axis=0)


def compute_tilt(rec: Recording, reference_window_s: float = 0.5,
                 cutoff_hz: float = 1.0) -> np.ndarray:
    """Trunk tilt (deg) relative to the initial quiet stance.

    The acceleration is low-pass filtered to keep mainly the gravity direction.
    The cutoff (1 Hz by default) is a compromise: lower values remove more
    movement noise but also smooth out the synchronisation swing.
    The reference is the mean gravity direction in the first
    ``reference_window_s`` seconds, so the recording must start with the
    climber standing still.
    """
    if rec.fs is None:
        raise ValueError("Resample the recording first (see climbimu.preprocess).")
    acc = rec.data[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    g = _lowpass(acc, rec.fs, cutoff_hz)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    n_ref = max(1, int(reference_window_s * rec.fs))
    ref = g[:n_ref].mean(axis=0)
    ref /= np.linalg.norm(ref)
    return np.degrees(np.arccos(np.clip(g @ ref, -1.0, 1.0)))


def add_tilt(rec: Recording, config: dict | None = None) -> Recording:
    """Add the ``tilt`` column (deg) to a resampled recording."""
    s = _settings(config)
    out = rec.copy()
    out.data["tilt"] = compute_tilt(out, s["reference_window_s"], s["gravity_cutoff_hz"])
    out.add_log("add_tilt", reference_window_s=s["reference_window_s"],
                gravity_cutoff_hz=s["gravity_cutoff_hz"])
    return out


def _runs(mask: np.ndarray) -> list:
    """Start and end index (inclusive) of each run of True values."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


# ---------------------------------------------------------------- detection

def detect_events(rec: Recording, config: dict | None = None) -> Events:
    """Propose the event times of one attempt. Always review them."""
    s = _settings(config)
    fs = rec.fs
    if fs is None:
        raise ValueError("Resample the recording first (see climbimu.preprocess).")
    t = rec.data["time"].to_numpy()
    tilt = compute_tilt(rec, s["reference_window_s"], s["gravity_cutoff_hz"])
    if "acc_norm" in rec.data:
        norm = rec.data["acc_norm"].to_numpy(dtype=float)
    else:
        norm = np.linalg.norm(rec.data[["acc_x", "acc_y", "acc_z"]].to_numpy(float), axis=1)
    norm_s = _lowpass(norm, fs, 5.0)
    ev = Events(times={e: None for e in EVENTS})

    # Climb start: first time the tilt stays above the climb threshold.
    min_len = int(s["climb_min_duration_s"] * fs)
    climb_runs = [r for r in _runs(tilt > s["climb_threshold_deg"])
                  if r[1] - r[0] + 1 >= min_len]
    if not climb_runs:
        ev.notes.append("Climb start not found: the trunk never stays tilted. "
                        "Check the reference posture or the thresholds.")
        return ev
    i = climb_runs[0][0]
    # Go back to where the trunk started to lean (last local minimum of tilt).
    while i > 0 and tilt[i - 1] <= tilt[i]:
        i -= 1
    i_climb = i
    ev.times["climb_start"] = float(t[i_climb])

    # Swing start: first time the tilt leaves the standing posture.
    above = np.where(tilt[:i_climb] > s["swing_threshold_deg"])[0]
    if above.size:
        j = above[0]
        while j > 0 and tilt[j - 1] > s["swing_threshold_deg"] / 2:
            j -= 1
        ev.times["swing_start"] = float(t[j])
    else:
        ev.notes.append("Swing not found before the climb start.")

    # Descent: last unloading episode after the climb start.
    min_unload = int(s["unloading_min_duration_s"] * fs)
    unload = [r for r in _runs(norm_s < s["unloading_threshold_g"])
              if r[1] - r[0] + 1 >= min_unload and r[0] > i_climb + min_len]
    if not unload:
        ev.notes.append("Descent not found: no unloading at the end of the climb. "
                        "Set climb_end and landing by hand.")
        return ev
    k0, k1 = unload[-1]
    # Descent starts where the magnitude began to drop below ~1 g.
    k = k0
    while k > 0 and norm_s[k - 1] < 0.95:
        k -= 1
    ev.times["climb_end"] = float(t[k])
    # Landing: first acceleration peak above 1 g after the unloading
    # (the impact when the feet touch the ground).
    k_min = k0 + int(np.argmin(norm_s[k0:k1 + 1]))
    stop = min(len(t), k_min + int(s["landing_search_s"] * fs))
    peaks = [p for p in range(k_min + 1, stop - 1)
             if norm_s[p] > 1.0 and norm_s[p] >= norm_s[p - 1] and norm_s[p] >= norm_s[p + 1]]
    if peaks:
        ev.times["landing"] = float(t[peaks[0]])
    else:
        ev.notes.append("Landing not found after the descent (no impact above 1 g). "
                        "Set it by hand.")
    if stop == len(t):
        ev.notes.append("The recording ends less than "
                        f"{s['landing_search_s']:g} s after the descent: check the landing.")
    if len(unload) > 1:
        ev.notes.append(f"{len(unload)} unloading episodes found; the last one was "
                        "used as the descent. Check it.")
    return ev


# ---------------------------------------------------------------- table

def events_to_row(rec_id: str, ev: Events) -> dict:
    row = {"id": rec_id}
    row.update({e: (round(ev.times[e], 3) if ev.times.get(e) is not None else None)
                for e in EVENTS})
    row["source"] = ev.source
    row["checked"] = "yes" if ev.checked else "no"
    row["notes"] = " | ".join(ev.notes)
    return row


def save_events_table(rows: list, path) -> None:
    """Save a list of rows (see :func:`events_to_row`) as CSV."""
    pd.DataFrame(rows).to_csv(path, index=False)


def load_events_table(path) -> dict:
    """Read an events table. Returns ``{id: Events}``."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path, sep=None, engine="python")
    result = {}
    for _, row in table.iterrows():
        times = {}
        for e in EVENTS:
            v = row.get(e)
            times[e] = float(str(v).replace(",", ".")) if pd.notna(v) and str(v).strip() else None
        checked = str(row.get("checked", "")).strip().lower() in ("yes", "y", "si", "sí", "1", "true")
        notes = [] if pd.isna(row.get("notes")) else [str(row.get("notes"))]
        result[str(row["id"])] = Events(times=times, source="table", checked=checked,
                                        notes=notes)
    return result


def get_events(rec: Recording, config: dict | None = None, table: dict | None = None) -> Events:
    """Checked events from the table if they exist; otherwise automatic detection."""
    if table is None:
        path = _settings(config).get("events_table")
        table = load_events_table(path) if path and Path(path).exists() else {}
    ev = table.get(rec.id)
    if ev is not None and ev.checked:
        return ev
    return detect_events(rec, config)


# ---------------------------------------------------------------- cutting

def get_segment(rec: Recording, events: Events, name: str) -> Recording:
    """Cut one segment (``"swing"``, ``"climb"`` or ``"descent"``)."""
    if name not in SEGMENTS:
        raise ValueError(f"Unknown segment '{name}'. Options: {list(SEGMENTS)}")
    start_ev, end_ev = SEGMENTS[name]
    t0, t1 = events[start_ev], events[end_ev]
    if t0 is None or t1 is None:
        raise ValueError(f"Segment '{name}' needs '{start_ev}' and '{end_ev}'.")
    if t1 <= t0:
        raise ValueError(f"Segment '{name}': '{end_ev}' is not after '{start_ev}'.")
    out = rec.copy()
    t = out.data["time"]
    out.data = out.data[(t >= t0) & (t <= t1)].reset_index(drop=True)
    out.meta = {**out.meta, "segment": name}
    out.add_log("segment", segment=name, start_event=start_ev, end_event=end_ev,
                start_s=t0, end_s=t1, events_source=events.source,
                events_checked=events.checked)
    return out


# ---------------------------------------------------------------- figure

def plot_events(rec: Recording, events: Events, path=None, config: dict | None = None):
    """Tilt and acceleration magnitude with the events and segments marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = _settings(config)
    t = rec.data["time"].to_numpy()
    tilt = compute_tilt(rec, s["reference_window_s"], s["gravity_cutoff_hz"])
    norm = rec.data["acc_norm"] if "acc_norm" in rec.data else np.linalg.norm(
        rec.data[["acc_x", "acc_y", "acc_z"]].to_numpy(float), axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, tilt, color="C0", lw=1.2)
    ax1.axhline(s["climb_threshold_deg"], color="0.6", ls=":", lw=0.8)
    ax1.set_ylabel("Trunk tilt (deg)")
    ax2.plot(t, norm, color="C1", lw=0.9)
    ax2.axhline(s["unloading_threshold_g"], color="0.6", ls=":", lw=0.8)
    ax2.set_ylabel("Acceleration magnitude (g)")
    ax2.set_xlabel("Time (s)")

    colors = {"swing": "C2", "climb": "C0", "descent": "C3"}
    for seg, (a, b) in SEGMENTS.items():
        if events[a] is not None and events[b] is not None:
            for ax in (ax1, ax2):
                ax.axvspan(events[a], events[b], color=colors[seg], alpha=0.08)
            ax1.text((events[a] + events[b]) / 2, 0.98, seg, transform=ax1.get_xaxis_transform(),
                     ha="center", va="top", fontsize=9, color=colors[seg])
    for n, e in enumerate(EVENTS):
        if events[e] is not None:
            for ax in (ax1, ax2):
                ax.axvline(events[e], color="k", lw=0.8, ls="--")
            # Alternate the label height so close events do not overlap.
            ax2.text(events[e], 0.02 if n % 2 == 0 else 0.55, f" {e}\n {events[e]:.2f} s",
                     rotation=90, transform=ax2.get_xaxis_transform(), fontsize=7,
                     va="bottom")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    status = "checked" if events.checked else "to review"
    title = f"Recording {rec.id}: events ({events.source}, {status})"
    if events.missing():
        title += f" - missing: {', '.join(events.missing())}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig
