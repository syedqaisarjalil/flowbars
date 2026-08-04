"""Bundled sample data for quickstart and testing.

Provides a few hundred rows of synthetic tick data (deterministic, seeded)
so the README quickstart runs copy-paste without external data files.
"""

from __future__ import annotations

import os

import pandas as pd

_SAMPLE_DIR = os.path.dirname(__file__)
_CSV_PATH = os.path.join(_SAMPLE_DIR, "tick_data.csv")


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample tick data as a pandas DataFrame.

    The data is synthetic (seeded, deterministic) and contains 500 ticks
    with columns ``timestamp`` (Unix ms), ``price``, and ``volume``.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``price``, ``volume``.
    """
    return pd.read_csv(_CSV_PATH)
