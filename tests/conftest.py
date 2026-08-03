"""Shared test fixtures for flowbars."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_tick_data() -> list[dict]:
    """A tiny, hand-computable tick series for deterministic tests.

    5 ticks with known prices — useful for tick rule, bar closing,
    and overflow/rollover verification.
    """
    return [
        {"timestamp": 1000, "price": 100.0, "volume": 10.0},
        {"timestamp": 1001, "price": 101.0, "volume": 5.0},  # up → +1
        {"timestamp": 1002, "price": 101.0, "volume": 8.0},  # flat → carry +1
        {"timestamp": 1003, "price": 99.0, "volume": 12.0},  # down → -1
        {"timestamp": 1004, "price": 102.0, "volume": 6.0},  # up → +1
    ]
