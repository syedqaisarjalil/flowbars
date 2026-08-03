"""Tick rule — derive trade direction from consecutive price moves.

Per AFML: sign(price_t − price_{t-1}), with equal-price carry-forward.
First tick has no sign (NaN). This is a well-known subtlety that existing
open implementations get wrong.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def derive_tick_sign(prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Derive trade signs from a price series using the tick rule.

    Parameters
    ----------
    prices : ndarray of float64
        1-D array of trade prices in chronological order.

    Returns
    -------
    ndarray of float64
        Array of signs: ``+1.0`` (buy / uptick), ``-1.0`` (sell / downtick),
        or ``NaN`` (undetermined — first tick, or carry-forward from first
        tick when all early prices are equal).

    Rule
    ----
    - First tick: ``NaN`` (no previous price to compare against).
    - ``price_t > price_{t-1}`` → ``+1.0``
    - ``price_t < price_{t-1}`` → ``-1.0``
    - ``price_t == price_{t-1}`` → carry forward the previous tick's sign.
    """
    n = len(prices)
    signs = np.empty(n, dtype=np.float64)

    if n == 0:
        return signs

    # First tick has no sign
    signs[0] = np.nan

    if n == 1:
        return signs

    # Compute differences and assign signs
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0.0:
            signs[i] = 1.0
        elif diff < 0.0:
            signs[i] = -1.0
        else:
            # Equal price → carry forward previous sign (which may be NaN)
            signs[i] = signs[i - 1]

    return signs


def resolve_tick_signs(
    prices: npt.NDArray[np.float64],
    supplied_sides: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    """Resolve trade signs: use supplied side column if provided, otherwise derive.

    Parameters
    ----------
    prices : ndarray of float64
        1-D array of trade prices in chronological order.
    supplied_sides : ndarray of float64 or None
        User-supplied side values (+1, -1, NaN). If provided, returned as-is.
        Values must already be validated (SchemaMapping handles this).

    Returns
    -------
    ndarray of float64
        Resolved signs array (same length as ``prices``).
    """
    if supplied_sides is not None:
        if len(supplied_sides) != len(prices):
            raise ValueError(
                f"supplied_sides length ({len(supplied_sides)}) "
                f"does not match prices length ({len(prices)})"
            )
        return supplied_sides.copy()

    return derive_tick_sign(prices)
