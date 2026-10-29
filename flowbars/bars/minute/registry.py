"""Registry for minute bar types.

Separate from :class:`~flowbars.bars.registry.BarRegistry` because several
minute types share names with tick types (``volume``, ``dollar``,
``imbalance_volume``, …).  Keys are prefixed with ``minute_`` so the two
registries cannot collide.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar


class MinuteBarRegistry:
    """Mapping from minute bar-type names to constructors and batch functions."""

    _constructors: ClassVar[dict[str, type]] = {}
    _batch_functions: ClassVar[dict[str, Callable[..., Any]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        constructor_cls: type,
        batch_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Register a minute bar type.

        Args:
            name: Minute bar-type name (e.g. ``"minute_volume"``).
            constructor_cls: The ``Minute*BarConstructor`` class.
            batch_fn: The ``compute_*_bars_from_minutes`` batch function.
        """
        if name in cls._constructors:
            raise ValueError(f"Minute bar type {name!r} is already registered.")
        cls._constructors[name] = constructor_cls
        if batch_fn is not None:
            cls._batch_functions[name] = batch_fn

    @classmethod
    def get_constructor(cls, name: str) -> type:
        """Return the constructor class for a registered minute bar type."""
        if name not in cls._constructors:
            raise KeyError(f"Unknown minute bar type: {name!r}. Available: {cls.list()}")
        return cls._constructors[name]

    @classmethod
    def get_batch_function(cls, name: str) -> Callable[..., Any]:
        """Return the batch function for a registered minute bar type."""
        if name not in cls._batch_functions:
            raise KeyError(f"Unknown minute bar type: {name!r}. Available: {cls.list()}")
        return cls._batch_functions[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered minute bar-type names."""
        return sorted(cls._constructors.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations. For testing only — not part of the public API."""
        cls._constructors.clear()
        cls._batch_functions.clear()
