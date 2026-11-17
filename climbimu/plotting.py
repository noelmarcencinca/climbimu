"""Figures to check the data before and after cleaning."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .recording import Recording


def plot_raw_vs_clean(raw: Recording, clean: Recording, path=None, title=None):
    """Acceleration magnitude (raw and clean) and orientation angles (clean).

    Gaps found in the raw recording are marked with grey vertical lines.
    """
    n_panels = 2 if clean.angle_columns() else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    t_raw = raw.data["time"].to_numpy()
    acc_raw = raw.data[raw.acc_columns()].to_numpy(dtype=float)
    norm_raw = np.sqrt((acc_raw ** 2).sum(axis=1))

    dt = np.diff(t_raw)
    gaps = t_raw[1:][dt > 3 * np.median(dt)]

    ax = axes[0]
    ax.plot(t_raw, norm_raw, color="0.75", lw=0.8, label="Raw")
    ax.plot(clean.data["time"], clean.data["acc_norm"], color="C0", lw=1.0,
            label="Clean")
    for g in gaps:
        ax.axvline(g, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel("Acceleration magnitude (g)")
    top = max(2.0, float(np.percentile(clean.data["acc_norm"], 99.9)) * 1.3)
    ax.set_ylim(0, top)
    ax.legend(loc="upper right", frameon=False)
    clipped = int(np.sum(norm_raw > top))
    if clipped:
        ax.text(0.01, 0.95, f"{clipped} raw samples above {top:.1f} g not shown",
                transform=ax.transAxes, va="top", fontsize=8, color="0.4")

    if n_panels == 2:
        ax = axes[1]
        for col in clean.angle_columns():
            ax.plot(clean.data["time"], clean.data[col], lw=1.0, label=col)
        ax.set_ylabel("Angle (deg, unwrapped)")
        ax.legend(loc="upper right", frameon=False, ncol=3)

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title or f"Recording {clean.id}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig
