"""Basic tests with synthetic data. Run them with: pytest"""

import numpy as np
import pandas as pd
import pytest

import climbimu as ci


def synthetic_file(tmp_path, name="3_2.csv"):
    """A fake IMU file with irregular sampling, gaps, spikes and angle wrapping."""
    rng = np.random.default_rng(0)
    dt = rng.choice([0.006, 0.007, 0.008], size=2000)
    dt[::80] = 0.065                                  # regular gaps
    t = np.cumsum(dt)
    acc_y = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    acc_y[[100, 500, 502, 504]] = [12, -10, 9, -14]   # spikes
    yaw = (350 + 30 * t) % 360                        # wraps around 360
    df = pd.DataFrame({"time": t, "acc_x": 0.0, "acc_y": acc_y, "acc_z": 0.0,
                       "yaw": yaw, "pitch": 10.0, "roll": 5.0})
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_filename_pattern():
    assert ci.parse_filename("1_2", "{climber}_{attempt}") == {"climber": 1, "attempt": 2}
    assert ci.parse_filename("ana_B3_fatigue", "{climber}_{boulder}_{condition}") == {
        "climber": "ana", "boulder": "B3", "condition": "fatigue"}
    with pytest.warns(UserWarning):
        assert ci.parse_filename("weird", "{climber}_{attempt}") == {}


def test_metadata_table_overrides_name(tmp_path):
    table = tmp_path / "meta.csv"
    pd.DataFrame({"file": ["3_2.csv"], "climber": ["C03"], "grade": ["7a"]}).to_csv(
        table, index=False)
    config = ci.load_config()
    config["metadata"].update(filename_pattern="{climber}_{attempt}", table=str(table))
    meta = ci.get_metadata(tmp_path / "3_2.csv", config)
    assert meta == {"id": "3_2", "climber": "C03", "attempt": 2, "grade": "7a"}


def test_quality_finds_problems(tmp_path):
    rec = ci.load_recording(synthetic_file(tmp_path))
    report = ci.check_quality(rec, ci.load_config())
    assert report.n_gaps == 24  # the first "gap" is before the first sample
    assert report.spikes["acc_y"] == 4
    assert report.angle_wraps["yaw"] > 0


def test_preprocess_cleans(tmp_path):
    config = ci.load_config()
    clean = ci.preprocess(ci.load_recording(synthetic_file(tmp_path)), config)
    assert clean.fs == 100
    assert np.allclose(np.diff(clean.data["time"]), 0.01)
    assert clean.data["acc_norm"].max() < 1.2
    assert np.abs(np.diff(clean.data["yaw"])).max() < 5
    assert [s["step"] for s in clean.log] == [
        "load", "despike", "unwrap_angles", "despike", "resample", "add_acc_norm"]


def test_units_conversion(tmp_path):
    path = synthetic_file(tmp_path)
    df = pd.read_csv(path)
    df["time"] *= 1000
    df[["acc_x", "acc_y", "acc_z"]] *= 9.80665
    df.to_csv(path, index=False)
    config = ci.load_config()
    config["units"].update(time="ms", acc="m/s2")
    rec = ci.load_recording(path, config)
    assert rec.data["time"].iloc[-1] < 100
    assert rec.data["acc_y"].median() == pytest.approx(1, abs=0.1)
