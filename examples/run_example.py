"""Example: quality report, cleaning and figure for one IMU file.

Usage:
    python examples/run_example.py path/to/file.xlsx config/example_config.yaml
"""

import sys
from pathlib import Path

import climbimu as ci
from climbimu.plotting import plot_raw_vs_clean


def main(file_path, config_path=None, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    config = ci.load_config(config_path)
    raw = ci.load_recording(file_path, config)
    print(raw)
    print("Labels:", raw.meta)

    report = ci.check_quality(raw, config)
    print()
    print(report.to_text())
    (out / f"{raw.id}_quality.txt").write_text(report.to_text(), encoding="utf-8")

    clean = ci.preprocess(raw, config)
    print()
    print(clean)
    ci.save_recording(clean, out / f"{raw.id}_clean.csv")
    plot_raw_vs_clean(raw, clean, out / f"{raw.id}_raw_vs_clean.png")
    print(f"\nResults saved in '{out}/'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
