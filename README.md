# climbimu

Open tools to analyse climbing movement from IMU (inertial measurement unit) data.

The goal of `climbimu` is to go from the raw file of a wearable sensor to
reproducible measures of movement variability and complexity (entropy) in
climbing and bouldering. It is designed by a sport scientist for sport
scientists and coaches: each step is simple, documented and reported.

> \*\*Status:\*\* early development (version 0.1.0). The first modules (reading,
> quality report and cleaning) work. Segmentation and entropy measures are
> the next steps. Feedback is welcome.

## Why

IMU data from climbing are often noisy: irregular sampling, lost samples,
isolated spikes and angle jumps. These problems change variability and
entropy values a lot. For example, in one test recording, sample entropy of
the acceleration magnitude was almost ten times lower with the raw data than
with the cleaned data, because a few spikes inflated the standard deviation.
`climbimu` makes these problems visible and fixes them in a transparent way.

## What it does (v0.2.0)

* **Reads** Excel or CSV files from any IMU. A small YAML file says which
column is which and in which units it is.
* **Labels** each recording (climber, attempt, boulder, condition...) from
the file name and/or from a metadata table. Labels are free: you define them.
* **Quality report**: real sampling frequency, gaps, spikes and angle jumps.
* **Cleaning**: spike removal (iterative Hampel filter), angle unwrapping,
resampling to a fixed frequency, optional low-pass filter.
* **Processing log**: every step and its parameters are saved with the data.
* **Figure**: raw vs clean signals, with the gaps marked.

## Installation

```bash
pip install git+https://github.com/noelmarcencinca/climbimu.git
```

Or, to work on the code:

```bash
git clone https://github.com/noelmarcencinca/climbimu.git
cd climbimu
pip install -e ".\[dev]"
```

## Quick start

1. Copy `config/example\_config.yaml` and adapt the column names and units to
your device.
2. Run the example on one file:

```bash
python examples/run\_example.py path/to/1\_1.xlsx config/my\_config.yaml
```

Or from Python:

```python
import climbimu as ci

config = ci.load\_config("config/my\_config.yaml")
raw = ci.load\_recording("data/1\_1.xlsx", config)

print(ci.check\_quality(raw, config).to\_text())

clean = ci.preprocess(raw, config)
ci.save\_recording(clean, "results/1\_1\_clean.csv")   # also saves the processing log
```

## Segmentation of an attempt

The tool assumes this protocol: the climber stands still, makes a hip
flexion-extension swing (synchronisation movement), climbs and comes down
from the top. It finds four events:

| Event | How it is detected |
|---|---|
| `swing_start` | the trunk tilt leaves the standing posture |
| `climb_start` | the trunk tilt rises and stays high (leaning towards the wall) |
| `climb_end` | start of the last unloading of the body (coming down from the top) |
| `landing` | first impact after the unloading |

The trunk tilt is the angle between the gravity direction measured by the
sensor and the gravity direction during the initial quiet stance. It does not
depend on how the sensor is placed, and it does not use the orientation
angles of the device.

```python
events = ci.detect_events(clean, config)
ci.plot_events(clean, events, "results/1_1_events.png", config)
climb = ci.get_segment(clean, events, "climb")   # also "swing" or "descent"
```

To review a whole folder:

```bash
python examples/segment_folder.py data/ config/my_config.yaml
```

This writes `results/events.csv` and one figure per file. Check each figure,
correct the times in the CSV if needed (it opens in Excel) and write `yes` in
the `checked` column. Then set `events_table` in your configuration: checked
rows are always used instead of the automatic detection, so every analysis is
reproducible.

**Protocol tips** for better detection: start the recording with at least
1-2 s of quiet standing, keep recording 2-3 s after landing, and place the
sensor in the same way for every climber.

## Labels for each recording

The tool does not assume any fixed study design. Three options:

1. **Nothing**: the file name is only an id.
2. **File name pattern**: `filename\_pattern: "{climber}\_{attempt}"` reads
`1\_2.xlsx` as climber 1, attempt 2. Field names are free, for example
`"{climber}\_{boulder}\_{condition}"`.
3. **Metadata table**: a CSV or Excel file with a `file` column and any other
columns (grade, boulder, sent or not...). It overrides the file name.

## Roadmap

* \[x] Reading, labels, quality report, cleaning, processing log (v0.1)
* \[ ] Segmentation: start movement, start of the climb, top and descent
(semi-automatic, with visual confirmation)
* \[ ] Classic measures: RMS, coefficient of variation, normalised jerk,
number of corrections
* \[ ] Entropy: sample entropy, multiscale entropy, permutation entropy
* \[ ] Sensitivity analysis of entropy parameters (m, r, signal length)
* \[ ] Batch processing of a folder and comparison by labels
* \[ ] Automatic draft of the methods section

## Data

No participant data are included in this repository. The `.gitignore` file
excludes `data/`, `results/` and Excel files to avoid uploading them by mistake.

## Citation

If you use `climbimu`, please cite it (see `CITATION.cff`, or use the
"Cite this repository" button on GitHub).

## Author

Noel Marcén-Cinca, Universidad San Jorge (Zaragoza, Spain).

## License

MIT. See `LICENSE`.

