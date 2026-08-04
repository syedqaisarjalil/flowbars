"""Tests for Phase 8 — public API and registry wiring."""

from __future__ import annotations

import pandas as pd
import pytest

import flowbars
from flowbars.bars.registry import BarRegistry

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _sample_ticks_df(n: int = 20) -> pd.DataFrame:
    """Return a DataFrame with default column names (timestamp, price, volume)."""
    return pd.DataFrame(
        {
            "timestamp": list(range(1000, 1000 + n * 1000, 1000)),
            "price": [100.0 + i * 0.1 for i in range(n)],
            "volume": [1.0] * n,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Registry finalization
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryFinalization:
    def test_all_ten_types_registered(self) -> None:
        """BarRegistry.list() returns all 10 bar types in sorted order."""
        registered = BarRegistry.list()
        assert len(registered) == 10
        expected = [
            "dollar",
            "imbalance_dollar",
            "imbalance_tick",
            "imbalance_volume",
            "run_dollar",
            "run_tick",
            "run_volume",
            "tick",
            "time",
            "volume",
        ]
        assert registered == sorted(expected)

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            BarRegistry.register("tick", object)

    def test_unknown_constructor_raises(self) -> None:
        """get_constructor() with an unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown bar type"):
            BarRegistry.get_constructor("made_up_type")

    def test_unknown_batch_function_raises(self) -> None:
        """get_batch_function() with an unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown bar type"):
            BarRegistry.get_batch_function("made_up_type")

    def test_get_constructor_returns_class(self) -> None:
        """get_constructor() returns the constructor class for each type."""
        for name in BarRegistry.list():
            cls = BarRegistry.get_constructor(name)
            assert isinstance(cls, type)
            assert cls.__name__.endswith("Constructor") or name == "time"

    def test_get_batch_function_returns_callable(self) -> None:
        """get_batch_function() returns a callable for each type."""
        for name in BarRegistry.list():
            fn = BarRegistry.get_batch_function(name)
            assert callable(fn)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience re-exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvenienceReexports:
    """Every ``compute_*_bars`` function is importable from ``flowbars``
    and produces a DataFrame when given valid tick data."""

    def test_compute_tick_bars_importable(self) -> None:
        from flowbars import compute_tick_bars

        df = _sample_ticks_df(10)
        result = compute_tick_bars(df, threshold=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_compute_volume_bars_importable(self) -> None:
        from flowbars import compute_volume_bars

        df = _sample_ticks_df(10)
        result = compute_volume_bars(df, threshold=3.0)
        assert isinstance(result, pd.DataFrame)

    def test_compute_dollar_bars_importable(self) -> None:
        from flowbars import compute_dollar_bars

        df = _sample_ticks_df(10)
        result = compute_dollar_bars(df, threshold=300.0)
        assert isinstance(result, pd.DataFrame)

    def test_compute_time_bars_importable(self) -> None:
        from flowbars import compute_time_bars

        df = _sample_ticks_df(10)
        result = compute_time_bars(df, interval_ms=60_000)
        assert isinstance(result, pd.DataFrame)

    def test_compute_imbalance_tick_bars_importable(self) -> None:
        from flowbars import compute_imbalance_tick_bars

        df = pd.DataFrame(
            {
                "timestamp": list(range(1000, 11000, 1000)),
                "price": [100.0 + i for i in range(10)],
                "volume": [1.0] * 10,
                "side": [1.0] * 10,
            }
        )
        result = compute_imbalance_tick_bars(df, initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_compute_imbalance_volume_bars_importable(self) -> None:
        from flowbars import compute_imbalance_volume_bars

        df = pd.DataFrame(
            {
                "timestamp": list(range(1000, 7000, 1000)),
                "price": [100.0] * 6,
                "volume": [2.0] * 6,
                "side": [1.0] * 6,
            }
        )
        result = compute_imbalance_volume_bars(df, initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_compute_imbalance_dollar_bars_importable(self) -> None:
        from flowbars import compute_imbalance_dollar_bars

        df = pd.DataFrame(
            {
                "timestamp": list(range(1000, 7000, 1000)),
                "price": [100.0] * 6,
                "volume": [2.0] * 6,
                "side": [1.0] * 6,
            }
        )
        result = compute_imbalance_dollar_bars(df, initial_ewa_t=1000.0, initial_ewa_proportion=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_compute_run_tick_bars_importable(self) -> None:
        from flowbars import compute_run_tick_bars

        df = pd.DataFrame(
            {
                "timestamp": list(range(1000, 11000, 1000)),
                "price": [100.0 + i for i in range(10)],
                "volume": [1.0] * 10,
                "side": [1.0] * 10,
            }
        )
        result = compute_run_tick_bars(df, initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_compute_run_volume_bars_importable(self) -> None:
        from flowbars import compute_run_volume_bars

        df = pd.DataFrame(
            {
                "timestamp": list(range(1000, 7000, 1000)),
                "price": [100.0] * 6,
                "volume": [2.0] * 6,
                "side": [1.0] * 6,
            }
        )
        result = compute_run_volume_bars(df, initial_ewa_t=20.0, initial_ewa_proportion=0.5)
        assert isinstance(result, pd.DataFrame)

    def test_compute_run_dollar_bars_importable(self) -> None:
        from flowbars import compute_run_dollar_bars

        df = pd.DataFrame(
            {
                "timestamp": list(range(1000, 7000, 1000)),
                "price": [100.0] * 6,
                "volume": [2.0] * 6,
                "side": [1.0] * 6,
            }
        )
        result = compute_run_dollar_bars(df, initial_ewa_t=2000.0, initial_ewa_proportion=0.5)
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# Constructor imports
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstructorImports:
    """Every ``*BarConstructor`` class is importable and instantiable."""

    def test_tick_bar_constructor_importable(self) -> None:
        from flowbars import TickBarConstructor

        ctor = TickBarConstructor(threshold=5)
        assert ctor is not None
        assert callable(ctor.update)

    def test_volume_bar_constructor_importable(self) -> None:
        from flowbars import VolumeBarConstructor

        ctor = VolumeBarConstructor(threshold=100.0)
        assert ctor is not None

    def test_dollar_bar_constructor_importable(self) -> None:
        from flowbars import DollarBarConstructor

        ctor = DollarBarConstructor(threshold=1000.0)
        assert ctor is not None

    def test_time_bar_constructor_importable(self) -> None:
        from flowbars import TimeBarConstructor

        ctor = TimeBarConstructor(interval_ms=60_000)
        assert ctor is not None

    def test_imbalance_tick_constructor_importable(self) -> None:
        from flowbars import ImbalanceTickBarConstructor

        ctor = ImbalanceTickBarConstructor()
        assert ctor is not None

    def test_imbalance_volume_constructor_importable(self) -> None:
        from flowbars import ImbalanceVolumeBarConstructor

        ctor = ImbalanceVolumeBarConstructor()
        assert ctor is not None

    def test_imbalance_dollar_constructor_importable(self) -> None:
        from flowbars import ImbalanceDollarBarConstructor

        ctor = ImbalanceDollarBarConstructor()
        assert ctor is not None

    def test_run_tick_constructor_importable(self) -> None:
        from flowbars import RunTickBarConstructor

        ctor = RunTickBarConstructor()
        assert ctor is not None

    def test_run_volume_constructor_importable(self) -> None:
        from flowbars import RunVolumeBarConstructor

        ctor = RunVolumeBarConstructor()
        assert ctor is not None

    def test_run_dollar_constructor_importable(self) -> None:
        from flowbars import RunDollarBarConstructor

        ctor = RunDollarBarConstructor()
        assert ctor is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level __all__ completeness
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllCompleteness:
    """flowbars.__all__ includes every public symbol."""

    def test_all_batch_functions_in_all(self) -> None:
        batch_names = {f"compute_{t}_bars" for t in BarRegistry.list()}
        for name in batch_names:
            assert name in flowbars.__all__, f"{name} missing from __all__"

    def test_all_constructors_in_all(self) -> None:
        constructor_names = [
            "DollarBarConstructor",
            "ImbalanceDollarBarConstructor",
            "ImbalanceTickBarConstructor",
            "ImbalanceVolumeBarConstructor",
            "RunDollarBarConstructor",
            "RunTickBarConstructor",
            "RunVolumeBarConstructor",
            "TickBarConstructor",
            "TimeBarConstructor",
            "VolumeBarConstructor",
        ]
        for name in constructor_names:
            assert name in flowbars.__all__, f"{name} missing from __all__"

    def test_core_types_in_all(self) -> None:
        core_names = [
            "BarRegistry",
            "BaseBarConstructor",
            "ContinuousCalendar",
            "EWMAThresholdEstimator",
            "SchemaMapping",
            "SessionCalendar",
            "StaticCalibrationHelper",
            "StaticThresholdEstimator",
            "ThresholdEstimator",
            "TradingCalendar",
            "bar_quality_report",
        ]
        for name in core_names:
            assert name in flowbars.__all__, f"{name} missing from __all__"

    def test_bar_registry_is_importable(self) -> None:
        """BarRegistry is accessible from the top-level package."""
        assert hasattr(flowbars, "BarRegistry")
        assert flowbars.BarRegistry is BarRegistry
