"""Machine-readable bar-type specifications.

Every registered bar type exposes a :class:`BarTypeSpec` describing how to
call it, what it emits, and which engine features it supports.  Specs are
derived automatically from the constructor signature (JSON Schema for the
JSON-safe parameters) plus a small set of class-level ``spec_overrides`` for
metadata that cannot be inferred (e.g. ``approximate``).

The point is discovery: a downstream service can list bar types, validate
user-supplied params against the schema, pick the right constructor, and
know the output columns — without hardcoding any bar-type knowledge.
"""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, cast

from flowbars.bars.constructor import (
    _OUTPUT_COLUMNS,
    _OUTPUT_CONTRACT_VERSION,
    _STATE_SCHEMA_VERSION,
)

# Parameters every constructor accepts that can be expressed as JSON.
_COMMON_JSON_PARAMS: dict[str, dict[str, Any]] = {
    "warmup_bars": {"type": "integer", "minimum": 0, "default": 0},
    "warmup_minutes": {"type": ["integer", "null"], "minimum": 0, "default": None},
    "watermark": {"type": ["string", "null"], "default": "timestamp"},
    "stream_id": {"type": "string", "default": "default"},
    "strict_ordering": {"type": "boolean", "default": False},
    "backend": {"type": "string", "enum": ["python", "numba"], "default": "python"},
}

# Constructor parameters that are not JSON-serializable configuration.
_NON_JSON_PARAMS = {
    "self",
    "calendar",
    "schema",
    "on_bar",
    "on_threshold_update",
}

_SCALAR_SCHEMAS: dict[Any, dict[str, Any]] = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
}


@dataclass(frozen=True)
class BarTypeSpec:
    """Description of one registered bar type.

    Attributes
    ----------
    bar_type : str
        Registry key (e.g. ``"volume"``, ``"minute_renko"``).
    input_kinds : tuple[str, ...]
        Accepted input kinds (``"tick"`` and/or ``"minute"``).
    params_schema : dict
        JSON Schema (draft-compatible subset) for constructor parameters.
    output_columns : tuple[str, ...]
        Columns of the frame returned by ``batch`` / ``update_many``.
        Always equal to ``core_columns + tuple(extras_columns)``.
    core_columns : tuple[str, ...]
        The contract columns every bar type emits (``dollar_value``,
        ``vwap``, and ``num_ticks`` are core).
    extras_columns : dict[str, str]
        Bar-type-specific numeric extras and their numeric types
        (name → e.g. ``"float64"``).  Empty for all built-in types.
    streaming : bool
        Whether the constructor supports single-item ``update()``.
    state_schema_version : int
        Version of the JSON state contract written by this bar type.
    spec_version : int
        Version of the public output contract (columns + frame attrs).
        Consumers can pin ``(bar_type, spec_version, params_hash)``.
    session_policies : tuple[str, ...]
        Session-boundary policies this bar type supports.
    warmup_minutes : int
        Recommended warmup duration in minutes before bars are trusted
        (from the constructor's ``recommended_warmup_minutes`` class
        attribute; distinct from the ``warmup_minutes`` runtime parameter,
        which appears in ``params_schema``).  ``0`` means "no warmup
        required" — the bar type works on the first emitted bar; it does
        not mean "warm up for zero minutes by default".
    volume_types : tuple[str, ...]
        Volume semantics supported for this bar type.
    approximate : bool
        True when the bar's notional/volume mapping is a proxy rather than an
        exact measurement of the underlying data.
    description : str
        Human-readable summary.
    batch_function : str or None
        Name of the registered batch function, if any.
    """

    bar_type: str
    input_kinds: tuple[str, ...]
    params_schema: dict[str, Any]
    output_columns: tuple[str, ...] = tuple(_OUTPUT_COLUMNS)
    core_columns: tuple[str, ...] = tuple(_OUTPUT_COLUMNS)
    extras_columns: dict[str, str] = field(default_factory=dict)
    streaming: bool = True
    state_schema_version: int = _STATE_SCHEMA_VERSION
    spec_version: int = _OUTPUT_CONTRACT_VERSION
    session_policies: tuple[str, ...] = ("close",)
    warmup_minutes: int = 0
    volume_types: tuple[str, ...] = ("base", "quote", "shares", "contracts", "tick")
    approximate: bool = False
    description: str = ""
    batch_function: str | None = None

    def __post_init__(self) -> None:
        if tuple(self.output_columns) != tuple(self.core_columns) + tuple(self.extras_columns):
            raise ValueError(
                "output_columns must equal core_columns + extras_columns; got "
                f"output_columns={tuple(self.output_columns)!r}, "
                f"core_columns={tuple(self.core_columns)!r}, "
                f"extras_columns={tuple(self.extras_columns)!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the spec."""
        return {
            "bar_type": self.bar_type,
            "input_kinds": list(self.input_kinds),
            "params_schema": self.params_schema,
            "output_columns": list(self.output_columns),
            "core_columns": list(self.core_columns),
            "extras_columns": dict(self.extras_columns),
            "streaming": self.streaming,
            "state_schema_version": self.state_schema_version,
            "spec_version": self.spec_version,
            "session_policies": list(self.session_policies),
            "warmup_minutes": self.warmup_minutes,
            "volume_types": list(self.volume_types),
            "approximate": self.approximate,
            "description": self.description,
            "batch_function": self.batch_function,
        }


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a JSON Schema fragment (best effort)."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _SCALAR_SCHEMAS:
        return dict(_SCALAR_SCHEMAS[annotation])

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _schema_for_annotation(non_none[0])
            base = inner.get("type")
            if base is None:
                return {}
            if isinstance(base, list):
                return {**inner, "type": [*base, "null"]}
            return {**inner, "type": [base, "null"]}
        return {}

    if origin is typing.Literal:
        values = list(args)
        return {"enum": values}

    return {}


def _init_function(constructor_cls: type) -> Any:
    """Return the first ``__init__`` defined in the class hierarchy."""
    for klass in constructor_cls.__mro__:
        if "__init__" in vars(klass):
            return vars(klass)["__init__"]
    return object.__init__  # pragma: no cover - every class has one


def _constructor_params_schema(constructor_cls: type) -> dict[str, Any]:
    """Build a JSON Schema for the JSON-safe constructor parameters."""
    init_fn = _init_function(constructor_cls)
    try:
        hints = typing.get_type_hints(init_fn)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}

    signature = inspect.signature(init_fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in _NON_JSON_PARAMS:
            continue
        schema = _schema_for_annotation(hints.get(name, param.annotation))
        if name in _COMMON_JSON_PARAMS:
            properties[name] = dict(_COMMON_JSON_PARAMS[name])
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif _is_json_scalar(param.default):
            schema["default"] = param.default
        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def build_bar_type_spec(
    name: str,
    constructor_cls: type,
    batch_fn: Callable[..., Any] | None = None,
) -> BarTypeSpec:
    """Derive a :class:`BarTypeSpec` from a constructor class.

    ``approximate`` is read from the constructor's class attribute (False by
    default), so a proxy-notional bar type just declares
    ``approximate = True``.  Other metadata can be tuned with a
    ``spec_overrides`` class attribute (read from the class ``__dict__`` so
    subclasses do not inherit it)::

        spec_overrides = {"description": "..."}
    """
    input_kind = getattr(constructor_cls, "input_kind", "tick")
    doc = inspect.getdoc(constructor_cls) or ""
    spec = BarTypeSpec(
        bar_type=name,
        input_kinds=(input_kind,),
        params_schema=_constructor_params_schema(constructor_cls),
        approximate=bool(getattr(constructor_cls, "approximate", False)),
        session_policies=tuple(getattr(constructor_cls, "session_policies", ("close",))),
        warmup_minutes=int(getattr(constructor_cls, "recommended_warmup_minutes", 0)),
        description=doc.split("\n\n", 1)[0].replace("\n", " ").strip(),
        batch_function=batch_fn.__name__ if batch_fn is not None else None,
    )

    overrides = constructor_cls.__dict__.get("spec_overrides")
    if overrides:
        spec = replace(spec, **overrides)
    return spec


class BarTypeSpecRegistry:
    """Shared registry behavior for tick and minute bar types.

    Subclasses must redeclare the class-level dicts so the two registries keep
    independent contents.
    """

    _label: ClassVar[str] = "Bar type"
    _constructors: ClassVar[dict[str, type]] = {}
    _batch_functions: ClassVar[dict[str, Callable[..., Any]]] = {}
    _specs: ClassVar[dict[str, BarTypeSpec]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        constructor_cls: type,
        batch_fn: Callable[..., Any] | None = None,
        spec: BarTypeSpec | None = None,
    ) -> None:
        """Register a bar type, deriving its spec when none is supplied."""
        if name in cls._constructors:
            raise ValueError(f"{cls._label} {name!r} is already registered.")
        cls._constructors[name] = constructor_cls
        if batch_fn is not None:
            cls._batch_functions[name] = batch_fn
        # Canonical identity used in frame attrs and saved state.  The emitted
        # ``bar_type`` label may be shorter and is not unique across registries.
        cast(Any, constructor_cls).registry_name = name
        cls._specs[name] = (
            spec if spec is not None else build_bar_type_spec(name, constructor_cls, batch_fn)
        )

    @classmethod
    def get_constructor(cls, name: str) -> type:
        """Return the constructor class for a registered bar type."""
        if name not in cls._constructors:
            raise KeyError(f"Unknown {cls._label.lower()}: {name!r}. Available: {cls.list()}")
        return cls._constructors[name]

    @classmethod
    def get_batch_function(cls, name: str) -> Callable[..., Any]:
        """Return the batch function for a registered bar type."""
        if name not in cls._batch_functions:
            raise KeyError(f"Unknown {cls._label.lower()}: {name!r}. Available: {cls.list()}")
        return cls._batch_functions[name]

    @classmethod
    def get_spec(cls, name: str) -> BarTypeSpec:
        """Return the :class:`BarTypeSpec` for a registered bar type."""
        if name not in cls._specs:
            raise KeyError(f"Unknown {cls._label.lower()}: {name!r}. Available: {cls.list()}")
        return cls._specs[name]

    @classmethod
    def constructor_from_state(
        cls,
        name: str,
        state: dict[str, Any],
        *,
        schema: Any = None,
        calendar: Any = None,
        verify_identity: bool = True,
    ) -> Any:
        """Rebuild the constructor registered under *name* from saved *state*.

        Uniform entry point — callers do not need to know the bar type's
        concrete constructor or its ``from_state`` signature.

        Parameters
        ----------
        name : str
            Registry key (e.g. ``"volume"`` or ``"minute_renko"``).
        state : dict
            State previously returned by ``get_state``.
        schema : SchemaMapping or MinuteSchemaMapping, optional
            Input mapping; must match the one recorded in the state.
        calendar : TradingCalendar, optional
            Must match the one recorded in the state (default: continuous).
        verify_identity : bool, default True
            Pass ``False`` to resume despite a calendar/schema mismatch.
        """
        constructor_cls = cast(Any, cls.get_constructor(name))
        return constructor_cls.from_state(
            state,
            calendar=calendar,
            schema=schema,
            verify_identity=verify_identity,
        )

    @classmethod
    def list_specs(cls) -> list[BarTypeSpec]:
        """Return every registered spec, sorted by bar-type name."""
        return [cls._specs[name] for name in cls.list()]

    @classmethod
    def describe(cls, name: str) -> dict[str, Any]:
        """Return the JSON-serializable spec for a registered bar type."""
        return cls.get_spec(name).to_dict()

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered bar type names."""
        return sorted(cls._constructors.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations. For testing only — not part of the public API."""
        cls._constructors.clear()
        cls._batch_functions.clear()
        cls._specs.clear()
