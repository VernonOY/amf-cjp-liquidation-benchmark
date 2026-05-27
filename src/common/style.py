"""Shared academic plotting style.

Serif fonts (STIX for math), a Tol-muted colour palette, thin black frames
and light dashed grids. Apply once at module import time via `apply()`.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


# Tol muted — colour-blind safe, prints well in grayscale.
PALETTE = [
    "#332288",  # indigo
    "#117733",  # dark green
    "#882255",  # burgundy
    "#88CCEE",  # cyan (light, use sparingly)
    "#DDCC77",  # sand
    "#AA4499",  # plum
    "#44AA99",  # teal
    "#CC6677",  # rose
]

# Darker ordered palette for paired / single-series figures.
MONO_SEQ = ["#000000", "#333333", "#555555", "#777777", "#999999", "#bbbbbb"]

# Continuous dark-to-light mapping for inventory / parameter sweeps.
GRADIENT = plt.get_cmap("cividis")


def apply():
    """Install academic rcParams globally."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman",
                        "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "lines.linewidth": 1.2,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": cycler("color", PALETTE),
        "image.cmap": "cividis",
    })


def gradient_colors(n: int, cmap="cividis", low=0.1, high=0.9):
    """Return a list of n distinguishable colours from a dark->light colormap."""
    cm = plt.get_cmap(cmap)
    if n == 1:
        return [cm(0.5)]
    return [cm(low + (high - low) * i / (n - 1)) for i in range(n)]
