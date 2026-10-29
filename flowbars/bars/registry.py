"""Bar type registry. Each bar type registers itself so the public API
doesn't hardcode bar types in if/elif chains.

Registration also derives a :class:`~flowbars.bars.spec.BarTypeSpec` from the
constructor, so discovery (``list_specs`` / ``describe``) needs no extra
bookkeeping when a new bar type is added.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

from flowbars.bars.spec import BarTypeSpec, BarTypeSpecRegistry

# Type aliases for what gets registered
ConstructorFactory = Callable[..., Any]
BatchFunction = Callable[..., Any]


class BarRegistry(BarTypeSpecRegistry):
    """Registry mapping bar type names to their implementations.

    Adding a new bar type = write a module + call register(). No edits to
    existing bar-type code or to the core engine.
    """

    _label: ClassVar[str] = "Bar type"
    _constructors: ClassVar[dict[str, type]] = {}
    _batch_functions: ClassVar[dict[str, BatchFunction]] = {}
    _specs: ClassVar[dict[str, BarTypeSpec]] = {}


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
