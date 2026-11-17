"""Tests of the segmentation with a synthetic attempt. Run them with: pytest"""

import numpy as np
import pandas as pd
import pytest

import climbimu as ci
from climbimu.segment import events_to_row, save_events_table


def synthetic_attempt(fs=100.0):
    """Stand 2 s, swing 2 s, climb 10 s, descent, landing, stand 2 s."""
    t = np.arange(0, 17.5, 1 / fs)
    tilt = np.zeros_like(t)                              # degrees
    swing = (t >= 2) & (t < 4)
    tilt[swing] = 12 * np.sin(np.pi * (t[swing] - 2)) ** 2
    climb = (t >= 4) & (t < 14.5)
    tilt[climb] = np.minimum(40, 40 * (t[climb] - 4)) + 8 * np.sin(2 * np.pi * 0.3 * t[climb])
    down = (t >= 14.5) & (t < 15.3)
    tilt[down] = 40 * (1 - (t[down] - 14.5) / 0.8)
    a = np.radians(tilt)
    mag = np.ones_like(t)
    mag[(t >= 14.6) & (t < 15.0)] = 0.3                  # unloading (drop)
    mag[(t >= 15.0) & (t < 15.08)] = 2.5                 # landing impact
    acc = pd.DataFrame({"time": t, "acc_x": mag * np.sin(a), "acc_y": mag * np.cos(a),
                        "acc_z": 0.0})
    return ci.preprocess(ci.Recording(data=acc, meta={"id": "s1"}), ci.load_config())


def test_detect_events():
    rec = synthetic_attempt()
    ev = ci.detect_events(rec, ci.load_config())
    assert ev.missing() == []
    assert ev["swing_start"] == pytest.approx(2.0, abs=0.3)
    assert ev["climb_start"] == pytest.approx(4.0, abs=0.3)
    assert ev["climb_end"] == pytest.approx(14.6, abs=0.3)
    assert ev["landing"] == pytest.approx(15.0, abs=0.2)


def test_get_segment():
    rec = synthetic_attempt()
    ev = ci.detect_events(rec, ci.load_config())
    climb = ci.get_segment(rec, ev, "climb")
    assert climb.data["time"].iloc[0] >= ev["climb_start"]
    assert climb.data["time"].iloc[-1] <= ev["climb_end"]
    assert climb.meta["segment"] == "climb"
    assert climb.log[-1]["step"] == "segment"
    with pytest.raises(ValueError):
        ci.get_segment(rec, ev, "unknown")


def test_checked_table_wins(tmp_path):
    rec = synthetic_attempt()
    ev = ci.detect_events(rec, ci.load_config())
    row = events_to_row(rec.id, ev)
    row.update(climb_start=5.0, checked="yes")
    path = tmp_path / "events.csv"
    save_events_table([row], path)
    config = ci.load_config()
    config["segmentation"]["events_table"] = str(path)
    final = ci.get_events(rec, config)
    assert final.source == "table" and final["climb_start"] == 5.0


def test_table_from_spanish_excel(tmp_path):
    """Excel in Spanish saves CSV with ';' and decimal commas."""
    path = tmp_path / "events.csv"
    path.write_text("id;swing_start;climb_start;climb_end;landing;source;checked;notes\n"
                    "s1;1,5;4,2;14,6;15;table;sí;\n", encoding="utf-8")
    table = ci.load_events_table(path)
    assert table["s1"].checked
    assert table["s1"]["climb_start"] == 4.2


def test_no_climb_found():
    rec = synthetic_attempt()
    rec.data[["acc_x", "acc_y", "acc_z"]] = [0.0, 1.0, 0.0]   # never tilts
    ev = ci.detect_events(rec, ci.load_config())
    assert ev["climb_start"] is None and ev.notes
