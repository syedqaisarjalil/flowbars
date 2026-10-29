"""Validation of the public bar output contract.

Downstream stores (e.g. ClickHouse) key bars on ``open_ts`` and map
bar-specific numbers into a ``Map(String, Float64)`` column.  This module
checks a produced frame against that contract before it leaves the process:

* the required columns are present (``open_ts``, ``close_ts``, OHLCV,
  ``bar_type``, ``partial``);
* ``partial`` is boolean;
* every column beyond the standard schema is numeric (the "extras");
* ``open_ts`` is unique within the frame (one series);
* each bar's ``close_ts`` is not earlier than its ``open_ts``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.constructor import _OUTPUT_COLUMNS
from flowbars.bars.spec import BarTypeSpec
from flowbars.core import BarContractError

REQUIRED_COLUMNS: tuple[str, ...] = (
    "open_ts",
    "close_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_type",
    "partial",
)

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_value",
    "vwap",
    "num_ticks",
    "bar_id",
    "open_ts",
    "close_ts",
)


def validate_bar_frame(
    bars: pd.DataFrame,
    *,
    spec: BarTypeSpec | None = None,
    require_unique_open_ts: bool = True,
) -> None:
    """Raise :class:`BarContractError` if *bars* violates the output contract.

    Parameters
    ----------
    bars : pd.DataFrame
        Frame returned by ``batch`` / ``update_many``.
    spec : BarTypeSpec, optional
        When given, its ``extras_columns`` are checked for numeric dtype.
    require_unique_open_ts : bool, default True
        Enforce one bar per ``open_ts`` (the downstream series key).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in bars.columns]
    if missing:
        raise BarContractError(f"Bar frame is missing required columns: {missing}")

    if len(bars) == 0:
        return

    if not pd.api.types.is_bool_dtype(bars["partial"]):
        raise BarContractError(f"'partial' must be boolean, got dtype {bars['partial'].dtype}")

    for col in _NUMERIC_COLUMNS:
        if col in bars.columns and not pd.api.types.is_numeric_dtype(bars[col]):
            raise BarContractError(f"Column {col!r} must be numeric, got {bars[col].dtype}")

    extra_columns = [col for col in bars.columns if col not in _OUTPUT_COLUMNS]
    for col in extra_columns:
        if not pd.api.types.is_numeric_dtype(bars[col]):
            raise BarContractError(
                f"Extra column {col!r} must be numeric (Float64), got {bars[col].dtype}"
            )
    if spec is not None:
        for col in spec.extras_columns:
            if col in bars.columns and not pd.api.types.is_numeric_dtype(bars[col]):
                raise BarContractError(
                    f"Declared extra {col!r} must be numeric, got {bars[col].dtype}"
                )

    if require_unique_open_ts:
        duplicated = bars["open_ts"].duplicated(keep=False)
        if bool(duplicated.any()):
            duplicates: Any = sorted(bars.loc[duplicated, "open_ts"].unique().tolist())
            raise BarContractError(
                f"open_ts must be unique within a series; duplicated values: {duplicates[:5]}"
            )

    backwards = bars["close_ts"] < bars["open_ts"]
    if bool(backwards.any()):
        raise BarContractError("close_ts must be >= open_ts for every bar")
