"""climbimu: open tools to analyse climbing movement from IMU data."""

from .io import load_config, load_recording, save_recording
from .metadata import get_metadata, parse_filename
from .preprocess import (add_acc_norm, despike, lowpass, preprocess, resample,
                         unwrap_angles)
from .quality import check_quality
from .recording import Recording
from .segment import (Events, add_tilt, compute_tilt, detect_events, get_events,
                      get_segment, load_events_table, plot_events)

__version__ = "0.2.0"
