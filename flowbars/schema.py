"""Schema mapping — validates user column names and normalizes input data.

No auto-detection, no fuzzy matching. The user tells us exactly which
columns map to which fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from flowbars.core import SchemaError, TickDataError, TickInfo

# ── Known schema fields ──────────────────────────────────────────────────────

_REQUIRED_FIELDS = frozenset({"timestamp", "price", "volume"})
_OPTIONAL_FIELDS = frozenset({"side"})
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


# ── Value validation (reusable, module-level) ────────────────────────────────


def validate_price(price: float) -> None:
    """Raise TickDataError if price is NaN, Inf, or negative."""
    if np.isnan(price) or np.isinf(price) or price < 0.0:
        raise TickDataError(f"Price must be finite and non-negative, got {price!r}")


def validate_volume(volume: float) -> None:
    """Raise TickDataError if volume is NaN, Inf, or negative.

    Zero-volume trades are accepted.
    """
    if np.isnan(volume) or np.isinf(volume) or volume < 0.0:
        raise TickDataError(f"Volume must be finite and non-negative, got {volume!r}")


def validate_side(side: float) -> None:
    """Raise TickDataError if side is not +1 or -1.

    NaN is accepted — it means tick rule hasn't derived a sign yet.
    """
    if np.isnan(side):
        return
    if side not in (-1.0, 1.0):
        raise TickDataError(f"Side must be +1 or -1 (or NaN for undetermined), got {side!r}")


# ── Array-level validation (batch path) ──────────────────────────────────────


def _first_bad_index(mask: npt.NDArray[np.bool_]) -> int | None:
    """Return the index of the first True in mask, or None if all clean."""
    if len(mask) == 0:
        return None
    idx = np.argmax(mask)  # argmax returns 0 if all False, so check
    return int(idx) if mask[idx] else None


def validate_price_array(prices: npt.NDArray[np.float64]) -> None:
    """Raise TickDataError if any price is NaN, Inf, or negative."""
    bad = np.isnan(prices) | np.isinf(prices) | (prices < 0.0)
    i = _first_bad_index(bad)
    if i is not None:
        raise TickDataError(
            f"Price at row {i} is invalid: {prices[i]!r}. Must be finite and non-negative."
        )


def validate_volume_array(volumes: npt.NDArray[np.float64]) -> None:
    """Raise TickDataError if any volume is NaN, Inf, or negative."""
    bad = np.isnan(volumes) | np.isinf(volumes) | (volumes < 0.0)
    i = _first_bad_index(bad)
    if i is not None:
        raise TickDataError(
            f"Volume at row {i} is invalid: {volumes[i]!r}. Must be finite and non-negative."
        )


def validate_side_array(sides: npt.NDArray[np.float64]) -> None:
    """Raise TickDataError if any side value is not +1, -1, or NaN."""
    nan = np.isnan(sides)
    valid = (sides == -1.0) | (sides == 1.0) | nan
    i = _first_bad_index(~valid)
    if i is not None:
        raise TickDataError(f"Side at row {i} is invalid: {sides[i]!r}. Must be +1, -1, or NaN.")


# ── SchemaMapping ────────────────────────────────────────────────────────────


class SchemaMapping:
    """Maps user-supplied column names to internal field names.

    Parameters
    ----------
    mapping : dict
        Keys are internal field names (``timestamp``, ``price``, ``volume``,
        and optionally ``side``). Values are the corresponding column names
        in the user's DataFrame or tick dict.

    Raises
    ------
    SchemaError
        If required keys are missing or unknown keys are provided.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        keys = set(mapping.keys())
        unknown = keys - _ALLOWED_FIELDS
        if unknown:
            raise SchemaError(
                f"Unknown schema keys: {sorted(unknown)}. Allowed keys: {sorted(_ALLOWED_FIELDS)}"
            )
        missing = _REQUIRED_FIELDS - keys
        if missing:
            raise SchemaError(
                f"Missing required schema keys: {sorted(missing)}. "
                f"Must provide: {sorted(_REQUIRED_FIELDS)}"
            )

        self._mapping: dict[str, str] = dict(mapping)

    # -- read-only accessors --

    @property
    def timestamp_col(self) -> str:
        """The user's column name mapped to timestamp."""
        return self._mapping["timestamp"]

    @property
    def price_col(self) -> str:
        """The user's column name mapped to price."""
        return self._mapping["price"]

    @property
    def volume_col(self) -> str:
        """The user's column name mapped to volume."""
        return self._mapping["volume"]

    @property
    def side_col(self) -> str | None:
        """The user's column name mapped to side, or None if not configured."""
        return self._mapping.get("side")

    @property
    def has_side(self) -> bool:
        """True if a side column was provided in the schema."""
        return "side" in self._mapping

    # -- DataFrame extraction (batch path) --

    def validate_columns(self, columns: set[str]) -> None:
        """Check that all mapped columns exist in the input.

        Raises SchemaError if any mapped column is missing.
        """
        for field, col in self._mapping.items():
            if col not in columns:
                raise SchemaError(
                    f"Column {col!r} (mapped to {field!r}) not found in "
                    f"input. Available columns: {sorted(columns)}"
                )

    def extract_arrays(
        self, df: pd.DataFrame
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
    ]:
        """Extract and validate numpy arrays from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The tick-level input data.

        Returns
        -------
        tuple
            ``(timestamps, prices, volumes, sides)`` where ``sides`` may be
            ``None`` if no side column was configured. Prices and volumes are
            ``float64``. Timestamps preserve their input dtype. Sides are
            ``float64`` (NaN for undetermined sign).

        Raises
        ------
        SchemaError
            If a mapped column is missing.
        TickDataError
            If any value fails validation (NaN/Inf/negative price or volume,
            malformed side).
        """
        self.validate_columns(set(df.columns))

        timestamps: npt.NDArray[Any] = np.asarray(df[self.timestamp_col].to_numpy())
        prices_arr = df[self.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        volumes_arr = df[self.volume_col].to_numpy(dtype=np.float64, na_value=np.nan)

        validate_price_array(prices_arr)
        validate_volume_array(volumes_arr)

        sides: npt.NDArray[np.float64] | None = None
        if self.has_side:
            side_col = self._mapping["side"]
            sides = df[side_col].to_numpy(dtype=np.float64, na_value=np.nan)
            validate_side_array(sides)

        return timestamps, prices_arr, volumes_arr, sides

    # -- Single-tick normalization (streaming path) --

    def normalize_tick(self, row: dict[str, Any]) -> TickInfo:
        """Normalize a single tick (dict row) to a ``TickInfo``.

        Parameters
        ----------
        row : dict
            A single tick as a dict with the user's column names.

        Returns
        -------
        TickInfo
            The normalized tick.

        Raises
        ------
        SchemaError
            If a mapped column is missing from the row.
        TickDataError
            If price, volume, or side fails validation.
        """
        # Check column presence
        for field, col in self._mapping.items():
            if col not in row:
                raise SchemaError(
                    f"Column {col!r} (mapped to {field!r}) not found in tick. "
                    f"Available keys: {sorted(row.keys())}"
                )

        price = float(row[self.price_col])
        volume = float(row[self.volume_col])
        timestamp = int(row[self.timestamp_col])

        validate_price(price)
        validate_volume(volume)

        side: int | None = None
        if self.has_side:
            side_col = self._mapping["side"]
            raw = row[side_col]
            if raw is None:
                side = None
            else:
                side = int(raw)
                validate_side(float(side))

        return TickInfo(timestamp=timestamp, price=price, volume=volume, side=side)
