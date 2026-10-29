"""Registry for minute bar types.

Separate from :class:`~flowbars.bars.registry.BarRegistry` because several
minute types share names with tick types (``volume``, ``dollar``,
``imbalance_volume``, …).  Keys are prefixed with ``minute_`` so the two
registries cannot collide.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

from flowbars.bars.spec import BarTypeSpec, BarTypeSpecRegistry


class MinuteBarRegistry(BarTypeSpecRegistry):
    """Mapping from minute bar-type names to constructors and batch functions."""

    _label: ClassVar[str] = "Minute bar type"
    _constructors: ClassVar[dict[str, type]] = {}
    _batch_functions: ClassVar[dict[str, Callable[..., Any]]] = {}
    _specs: ClassVar[dict[str, BarTypeSpec]] = {}
