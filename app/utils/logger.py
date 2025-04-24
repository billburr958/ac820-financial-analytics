# -------------------------------------------------------------
# utils/logger.py  ·  single-source rotating logger (QUIET)
# -------------------------------------------------------------
import logging, pathlib
from logging.handlers import RotatingFileHandler

# ── paths ------------------------------------------------------
ROOT     = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ── formatter --------------------------------------------------
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"
_fmt  = logging.Formatter(_FMT, datefmt=_DATE)

# ── root logger -----------------------------------------------
root = logging.getLogger()
root.setLevel(logging.INFO)                  # quiet by default

# console -------------------------------------------------------
if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
    sh = logging.StreamHandler()
    sh.setFormatter(_fmt)
    root.addHandler(sh)

# file (rotating) ----------------------------------------------
if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000,
                             backupCount=5, encoding="utf-8")
    fh.setFormatter(_fmt)
    root.addHandler(fh)

# silence noisy libraries --------------------------------------
for noisy in ("fsevents", "watchdog", "urllib3", "PIL"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

root.debug("LOGGER ready – handlers attached")

# helper --------------------------------------------------------
def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    lg = logging.getLogger(name or "app")
    lg.setLevel(level)
    return lg
