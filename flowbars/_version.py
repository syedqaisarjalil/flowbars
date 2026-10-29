"""Single source of truth for the flowbars package version.

Kept in its own module so that internal code (state serialization, registry
metadata) can import it without triggering the package ``__init__`` and its
circular imports.
"""

__version__ = "0.1.0"
