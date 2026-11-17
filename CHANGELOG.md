# Changelog

## 0.1.0 (2026-11-17)

First public version.

- Read IMU files (Excel or CSV) with a YAML configuration that maps device columns and units.
- Labels for each recording from a free file name pattern and/or a metadata table.
- Quality report: sampling frequency, irregular intervals, gaps, acceleration spikes, angle wrapping.
- Cleaning: iterative Hampel despiking (acceleration and angles), angle unwrapping, resampling to a fixed frequency, optional low-pass filter, acceleration magnitude.
- Processing log saved with every cleaned recording.
- Figure of raw vs clean signals.
