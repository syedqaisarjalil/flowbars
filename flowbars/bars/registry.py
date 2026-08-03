"""Bar type registry. Each bar type registers itself so the public API
doesn't hardcode bar types in if/elif chains.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

# Type aliases for what gets registered
ConstructorFactory = Callable[..., Any]
BatchFunction = Callable[..., Any]


class BarRegistry:
    """Lightweight registry mapping bar type names to their implementations.

    Adding a new bar type = write a module + call register(). No edits to
    existing bar-type code or to the core engine.
    """

    _constructors: ClassVar[dict[str, type]] = {}
    _batch_functions: ClassVar[dict[str, BatchFunction]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        constructor_cls: type,
        batch_fn: BatchFunction | None = None,
    ) -> None:
        """Register a bar type.

        Args:
            name: Bar type name (e.g. ``"dollar"``, ``"imbalance_tick"``).
            constructor_cls: The ``*BarConstructor`` class for streaming.
            batch_fn: The ``compute_*_bars()`` batch function.
        """
        if name in cls._constructors:
            raise ValueError(f"Bar type {name!r} is already registered.")
        cls._constructors[name] = constructor_cls
        if batch_fn is not None:
            cls._batch_functions[name] = batch_fn

    @classmethod
    def get_constructor(cls, name: str) -> type:
        """Return the constructor class for a registered bar type."""
        if name not in cls._constructors:
            raise KeyError(f"Unknown bar type: {name!r}. Available: {cls.list()}")
        return cls._constructors[name]

    @classmethod
    def get_batch_function(cls, name: str) -> BatchFunction:
        """Return the batch function for a registered bar type."""
        if name not in cls._batch_functions:
            raise KeyError(f"Unknown bar type: {name!r}. Available: {cls.list()}")
        return cls._batch_functions[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered bar type names."""
        return sorted(cls._constructors.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations. For testing only — not part of the public API."""
        cls._constructors.clear()
        cls._batch_functions.clear()


def register_bar(name: str) -> Callable[[type], type]:
    """Decorator that registers a bar constructor class.

    Usage::

        @register_bar("dollar")
        class DollarBarConstructor(BaseBarConstructor):
            ...
    """

    def decorator(cls: type) -> type:
        BarRegistry.register(name, cls)  # batch_fn registered separately
        return cls

    return decorator
