"""saturate's default log output: `[pump] message` lines on stderr while nobody else has
configured logging.

Every module logs to a logger under `saturate` (logging.getLogger(__name__)). When saturate is
imported, the `saturate` logger gets one fallback handler and, unless the caller has already
set one, the level INFO. The fallback handler decides at each record: while neither the root
logger nor `saturate` has another handler, it prints the record to stderr; once the caller has
configured logging, it prints nothing and the caller's handlers get the record. The library
never configures the root logger. To show only problems:
`logging.getLogger("saturate").setLevel(logging.WARNING)`.
"""

from __future__ import annotations

import logging
import sys

PACKAGE = "saturate"


class _Fallback(logging.Handler):
    """Prints to stderr only while nobody has configured logging. Writes to sys.stderr as it is
    at each record: pytest, notebooks and agents replace sys.stderr after import."""

    def __init__(self, package: logging.Logger) -> None:
        super().__init__()
        # held here, not looked up per record: logging.getLogger takes logging's module lock,
        # and emit runs under this handler's lock, while dictConfig takes them the other way
        self.package = package
        self.setFormatter(logging.Formatter("[pump] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if self.configured():
            return
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)

    def configured(self) -> bool:
        """True once the caller has put a handler on the root logger or on `saturate`."""
        theirs = [handler for handler in self.package.handlers if handler is not self]
        return bool(theirs or logging.root.handlers)


def install() -> None:
    """Attach the fallback handler once (at import, under the import lock)."""
    package = logging.getLogger(PACKAGE)
    if any(isinstance(handler, _Fallback) for handler in package.handlers):
        return
    package.addHandler(_Fallback(package))
    if package.level == logging.NOTSET:  # progress lines are INFO; a caller's own level wins
        package.setLevel(logging.INFO)
