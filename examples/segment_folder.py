"""Example: propose events for every file in a folder, for review.

Usage:
    python examples/segment_folder.py data/ config/my_config.yaml

Workflow:
1. Run this script. It writes results/events.csv and one figure per file.
2. Look at each figure. If an event is wrong, write the right time (in s)
   in events.csv (it opens in Excel). Write "yes" in the "checked" column of
   every row you have reviewed, even if you did not change anything.
3. Put the path of events.csv in your configuration (segmentation: events_table)
   and run the script again: checked rows are kept as they are.
"""

import sys
from pathlib import Path

import climbimu as ci
from climbimu.segment import events_to_row, save_events_table

EXTENSIONS = {".xlsx", ".xls", ".csv"}


def main(folder, config_path=None, out_dir="results"):
    out = Path(out_dir)
    (out / "events").mkdir(parents=True, exist_ok=True)
    config = ci.load_config(config_path)
    table_path = config["segmentation"].get("events_table")
    table = ci.load_events_table(table_path) if table_path and Path(table_path).exists() else {}

    rows = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in EXTENSIONS or path.name.startswith("~"):
            continue
        rec = ci.preprocess(ci.load_recording(path, config), config)
        ev = ci.get_events(rec, config, table)
        ci.plot_events(rec, ev, out / "events" / f"{rec.id}_events.png", config)
        rows.append(events_to_row(rec.id, ev))
        status = "checked" if ev.checked else "to review"
        missing = f", missing: {', '.join(ev.missing())}" if ev.missing() else ""
        print(f"{rec.id}: {ev.source}, {status}{missing}")

    save_events_table(rows, out / "events.csv")
    print(f"\nEvents saved in '{out / 'events.csv'}'. Figures in '{out / 'events'}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
