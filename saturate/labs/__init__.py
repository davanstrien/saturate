"""Lab helpers: not part of the contract; may change or be removed in any release, without
deprecation.

Nothing here is exported from `saturate` or covered by CONTRACT.md. Import a lab module by its
full name, e.g. `from saturate.labs.compile_cache import compile_cache`. The first import of
anything under `saturate.labs` gives one FutureWarning; a new lab module needs no warning of
its own.
"""

import warnings

warnings.warn(
    "saturate.labs holds lab helpers: not part of the contract; they may change or be removed "
    "in any release, without deprecation",
    FutureWarning,
    stacklevel=2,
)
