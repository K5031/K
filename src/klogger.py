"""
Shared logging setup for all modules.

Usage in any module's __init__:
    from klogger import get_logger
    self.log = get_logger("Remis")

Then replace prints with:
    self.log.info("stored: %s", facts)
    self.log.warning("store failed: %s", e)
    self.log.error("conflict check failed: %s", e)

Same [Name] prefix as before, but now with levels and consistent formatting —
and any module can be silenced independently, e.g.:
    logging.getLogger("Remis").setLevel(logging.WARNING)
to hide its INFO-level chatter (like every stored fact) while still seeing
warnings/errors, without touching remis.py at all.
"""

import logging
import sys

_CONFIGURED = False


def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Returns a logger that prints as '[name] message', same style as
    the old print(f"[{self.name}] ...") calls, but as a real logger."""
    _configure_root()
    return logging.getLogger(name)