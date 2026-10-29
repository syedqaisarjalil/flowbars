"""Tests for Phase 8 — public API and registry wiring."""

from __future__ import annotations

import pandas as pd
import pytest

import flowbars
from flowbars.bars.numba_backend import _NUMBA_AVAILABLE
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


# ═══════════════════════════════════════════════════════════════════════════════
# Minute path registry and exports
# ═══════════════════════════════════════════════════════════════════════════════


MINUTE_TYPES = [
    "minute_volume",
    "minute_dollar",
    "minute_imbalance_volume",
    "minute_imbalance_dollar",
    "minute_run_volume",
    "minute_run_dollar",
    "minute_volatility",
    "minute_range",
    "minute_renko",
    "minute_hybrid",
]

MINUTE_CONSTRUCTORS = [
    "MinuteVolumeBarConstructor",
    "MinuteDollarBarConstructor",
    "MinuteImbalanceVolumeBarConstructor",
    "MinuteImbalanceDollarBarConstructor",
    "MinuteRunVolumeBarConstructor",
    "MinuteRunDollarBarConstructor",
    "MinuteVolatilityBarConstructor",
    "MinuteRangeBarConstructor",
    "MinuteRenkoBarConstructor",
    "MinuteHybridBarConstructor",
]

MINUTE_COMPUTE_FUNCTIONS = [
    "compute_volume_bars_from_minutes",
    "compute_dollar_bars_from_minutes",
    "compute_imbalance_volume_bars_from_minutes",
    "compute_imbalance_dollar_bars_from_minutes",
    "compute_run_volume_bars_from_minutes",
    "compute_run_dollar_bars_from_minutes",
    "compute_volatility_bars_from_minutes",
    "compute_range_bars_from_minutes",
    "compute_renko_bars_from_minutes",
    "compute_hybrid_bars_from_minutes",
]


def _sample_minutes_df(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [i * 60_000 for i in range(n)],
            "open": [100.0 + (i % 5) for i in range(n)],
            "high": [100.5 + (i % 5) for i in range(n)],
            "low": [99.5 + (i % 5) for i in range(n)],
            "close": [100.0 + (i % 5) + 0.1 for i in range(n)],
            "volume": [10.0] * n,
        }
    )


class TestMinuteRegistry:
    def test_minute_registry_exact_types(self) -> None:
        assert flowbars.MinuteBarRegistry.list() == sorted(MINUTE_TYPES)

    def test_minute_types_not_in_tick_registry(self) -> None:
        tick_registered = set(BarRegistry.list())
        assert all(name not in tick_registered for name in MINUTE_TYPES)

    def test_minute_registry_has_constructors_and_functions(self) -> None:
        for name in MINUTE_TYPES:
            assert isinstance(flowbars.MinuteBarRegistry.get_constructor(name), type)
            assert callable(flowbars.MinuteBarRegistry.get_batch_function(name))


class TestMinuteExports:
    def test_minute_names_in_all(self) -> None:
        for name in MINUTE_CONSTRUCTORS + MINUTE_COMPUTE_FUNCTIONS:
            assert name in flowbars.__all__, f"{name} missing from __all__"
            assert hasattr(flowbars, name)

    def test_minute_helpers_in_all(self) -> None:
        for name in (
            "MinuteBarRegistry",
            "MinuteCalibrationHelper",
            "MinuteSchemaMapping",
            "BaseMinuteBarConstructor",
        ):
            assert name in flowbars.__all__, f"{name} missing from __all__"

    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("MinuteVolumeBarConstructor", {"threshold": 50.0}),
            ("MinuteDollarBarConstructor", {"threshold": 5_000.0}),
            ("MinuteImbalanceVolumeBarConstructor", {}),
            ("MinuteImbalanceDollarBarConstructor", {}),
            ("MinuteRunVolumeBarConstructor", {}),
            ("MinuteRunDollarBarConstructor", {}),
            ("MinuteVolatilityBarConstructor", {"threshold": 0.02}),
            ("MinuteRangeBarConstructor", {"threshold": 0.02}),
            ("MinuteRenkoBarConstructor", {"threshold": 0.01}),
            (
                "MinuteHybridBarConstructor",
                {"dollar_threshold": 5_000.0, "volatility_threshold": 0.02},
            ),
        ],
    )
    def test_minute_constructors_instantiate(self, name: str, kwargs: dict) -> None:
        ctor = getattr(flowbars, name)(**kwargs)
        assert callable(ctor.update)

    def test_minute_compute_functions_produce_frames(self) -> None:
        df = _sample_minutes_df()
        results = {
            "compute_volume_bars_from_minutes": (df, 50.0),
            "compute_dollar_bars_from_minutes": (df, 5_000.0),
            "compute_volatility_bars_from_minutes": (df, 0.02),
            "compute_range_bars_from_minutes": (df, 0.02),
            "compute_renko_bars_from_minutes": (df, 0.01),
        }
        for name, args in results.items():
            out = getattr(flowbars, name)(*args)
            assert isinstance(out, pd.DataFrame)
            assert list(out.columns) == [
                "bar_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dollar_value",
                "vwap",
                "num_ticks",
                "open_ts",
                "close_ts",
                "bar_type",
            ]


# ═══════════════════════════════════════════════════════════════════════════════
# Tick-path signature parity (warmup_bars / backend / stream_id / callbacks)
# ═══════════════════════════════════════════════════════════════════════════════


TICK_CONSTRUCTOR_KWARGS = [
    ("TickBarConstructor", {"threshold": 5}),
    ("TimeBarConstructor", {"interval_ms": 30_000}),
    ("VolumeBarConstructor", {"threshold": 20.0}),
    ("DollarBarConstructor", {"threshold": 2_000.0}),
    ("ImbalanceTickBarConstructor", {"initial_ewa_t": 10.0, "initial_ewa_proportion": 0.5}),
    ("ImbalanceVolumeBarConstructor", {"initial_ewa_t": 20.0, "initial_ewa_proportion": 0.5}),
    (
        "ImbalanceDollarBarConstructor",
        {"initial_ewa_t": 2_000.0, "initial_ewa_proportion": 0.5},
    ),
    ("RunTickBarConstructor", {"initial_ewa_t": 10.0, "initial_ewa_proportion": 0.5}),
    ("RunVolumeBarConstructor", {"initial_ewa_t": 20.0, "initial_ewa_proportion": 0.5}),
    ("RunDollarBarConstructor", {"initial_ewa_t": 2_000.0, "initial_ewa_proportion": 0.5}),
]


def _tick_df(n: int = 200) -> pd.DataFrame:
    """Deterministic tick frame with a side column for information bars."""
    return pd.DataFrame(
        {
            "timestamp": [i * 1_000 for i in range(n)],
            "price": [100.0 + (i % 7) * 0.1 for i in range(n)],
            "volume": [1.0 + (i % 5) for i in range(n)],
            "side": [1.0 if i % 3 else -1.0 for i in range(n)],
        }
    )


class TestTickSignatureParity:
    @pytest.mark.parametrize("name,kwargs", TICK_CONSTRUCTOR_KWARGS)
    def test_accepts_warmup_and_backend(self, name: str, kwargs: dict) -> None:
        for backend in ("python", "numba"):
            ctor = getattr(flowbars, name)(**kwargs, warmup_bars=2, backend=backend)
            assert ctor._warmup_bars == 2
            assert ctor._backend == backend

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            flowbars.VolumeBarConstructor(100.0, backend="rust")

    def test_warmup_ids_and_callbacks_standard(self) -> None:
        seen: list[int] = []
        ctor = flowbars.VolumeBarConstructor(
            10.0, warmup_bars=2, on_bar=lambda bar: seen.append(bar.bar_id)
        )
        bars = ctor.batch(_tick_df(100))
        assert bars["bar_id"].tolist() == list(range(2, len(bars) + 2))
        assert seen == list(range(len(bars) + 2))

    def test_compute_function_forwards_new_kwargs(self) -> None:
        seen: list[int] = []
        df = _tick_df(100)
        out = flowbars.compute_volume_bars(
            df,
            10.0,
            warmup_bars=1,
            backend="python",
            stream_id="s1",
            on_bar=lambda bar: seen.append(bar.bar_id),
        )
        assert out["bar_id"].tolist() == list(range(1, len(out) + 1))
        assert seen == list(range(len(out) + 1))

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
    @pytest.mark.parametrize("name,kwargs", TICK_CONSTRUCTOR_KWARGS)
    def test_python_numba_parity(self, name: str, kwargs: dict) -> None:
        df = _tick_df()
        py = getattr(flowbars, name)(**kwargs, backend="python").batch(df)
        nb = getattr(flowbars, name)(**kwargs, backend="numba").batch(df)
        assert len(py) > 0
        pd.testing.assert_frame_equal(py.reset_index(drop=True), nb.reset_index(drop=True))

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
    def test_numba_warmup_parity(self) -> None:
        df = _tick_df()
        py_seen: list[int] = []
        nb_seen: list[int] = []
        py = flowbars.TickBarConstructor(
            10, warmup_bars=2, on_bar=lambda bar: py_seen.append(bar.bar_id)
        ).batch(df)
        nb = flowbars.TickBarConstructor(
            10,
            warmup_bars=2,
            backend="numba",
            on_bar=lambda bar: nb_seen.append(bar.bar_id),
        ).batch(df)
        pd.testing.assert_frame_equal(py.reset_index(drop=True), nb.reset_index(drop=True))
        assert py["bar_id"].tolist() == list(range(2, len(py) + 2))
        assert nb_seen == py_seen == list(range(len(py) + 2))
