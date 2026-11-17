# Changelog

## 0.2.0 (2026-11-17)

- New `segment` module: trunk tilt from the gravity direction, semi-automatic
  detection of swing start, climb start, climb end (start of the descent) and
  landing.
- Events table (CSV) to review and correct the events. Checked rows are always
  used. Reads CSV files saved by Excel in Spanish (";" and decimal comma).
- Cut a recording into segments (`swing`, `climb`, `descent`), with the cut
  saved in the processing log.
- Figure of the events and segments for visual review.
- New example `examples/segment_folder.py` to process a whole folder.

## 0.1.0 (2026-11-17)

First public version.

- Read IMU files (Excel or CSV) with a YAML configuration that maps device columns and units.
- Labels for each recording from a free file name pattern and/or a metadata table.
- Quality report: sampling frequency, irregular intervals, gaps, acceleration spikes, angle wrapping.
- Cleaning: iterative Hampel despiking (acceleration and angles), angle unwrapping, resampling to a fixed frequency, optional low-pass filter, acceleration magnitude.
- Processing log saved with every cleaned recording.
- Figure of raw vs clean signals.
