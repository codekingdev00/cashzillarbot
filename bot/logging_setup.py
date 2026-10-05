from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bot.config import LoggingConfig


def setup_logging(cfg: LoggingConfig) -> None:
    Path(cfg.file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(cfg.level.upper())
    root.handlers.clear()
    fh = RotatingFileHandler(cfg.file, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding="utf-8")
    fh.setFormatter(fmt)
    try:  # Windows consoles default to a legacy code page that cannot print emoji
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    for noisy in ("httpx", "telegram", "ccxt", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
