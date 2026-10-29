"""saturate's default log output: `[pump] message` lines on stderr, only while nobody else has
configured logging.

Every module logs to a logger under `saturate` (logging.getLogger(__name__)). The library never
configures the root logger. pump() and Engine call `default_output()` when they start: with no
handler on `saturate` or on the root logger, it adds one stderr handler and sets `saturate` to
INFO. Once the caller configures logging (a handler on either), it removes that handler and
restores the level, so the caller's configuration decides what is shown and where.
"""

from __future__ import annotations

import logging
import sys

PACKAGE = "saturate"


class _Stderr(logging.StreamHandler):
    """Writes to sys.stderr as it is at each record, not as it was when the handler was made:
    pytest, notebooks and agents replace sys.stderr after import."""

    def __init__(self, level_before: int) -> None:
        super().__init__()
        self.level_before = level_before  # the logger's level before this handler set INFO
        self.setFormatter(logging.Formatter("[pump] %(message)s"))

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__ sets it; sys.stderr is always used
        pass


def default_output() -> None:
    """Show saturate's INFO and above on stderr, unless the caller has configured logging."""
    logger = logging.getLogger(PACKAGE)
    ours = [handler for handler in logger.handlers if isinstance(handler, _Stderr)]
    theirs = [handler for handler in logger.handlers if not isinstance(handler, _Stderr)]
    configured = bool(theirs or logging.getLogger().handlers)
    if configured:
        for handler in ours:
            logger.removeHandler(handler)
            logger.setLevel(handler.level_before)
    elif not ours:
        handler = _Stderr(logger.level)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
