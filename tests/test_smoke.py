"""Smoke tests — verify the package builds, imports, and core types work."""

from __future__ import annotations

import flowbars


class TestPackage:
    """Verify the package is installable and importable."""

    def test_version(self) -> None:
        assert flowbars.__version__ == "0.1.0"

    def test_imports(self) -> None:
        """All public API symbols are importable."""

    def test_bar_registry(self) -> None:
        """BarRegistry has standard bar types registered on import."""
        registered = flowbars.BarRegistry.list()
        assert "tick" in registered
        assert "volume" in registered
        assert "dollar" in registered
        assert "time" in registered


class TestCoreTypes:
    """Core dataclasses and exceptions behave correctly."""

    def test_tick_info_creation(self) -> None:
        from flowbars.core import TickInfo

        tick = TickInfo(timestamp=1000, price=100.0, volume=10.0)
        assert tick.price == 100.0
        assert tick.side is None  # not derived yet

    def test_bar_creation(self) -> None:
        from flowbars.core import Bar

        bar = Bar(
            bar_id=0,
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=41.0,
            dollar_value=4123.0,
            vwap=100.56,
            num_ticks=5,
            open_ts=1000,
            close_ts=1004,
            bar_type="dollar",
        )
        assert bar.bar_type == "dollar"
        assert bar.bar_id == 0

    def test_exception_hierarchy(self) -> None:
        from flowbars.core import (
            FlowbarsError,
            SchemaError,
            StateValidationError,
            ThresholdError,
            TickDataError,
        )

        for exc_cls in [SchemaError, ThresholdError, StateValidationError, TickDataError]:
            assert issubclass(exc_cls, FlowbarsError)
