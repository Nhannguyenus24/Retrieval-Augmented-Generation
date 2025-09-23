import logging

RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[41m",   # Red background
}

# Màu cho các field
FIELD_COLORS = {
    "asctime": "\033[92m",    # Light green
    "threadName": "\033[34m", # Blue
    "name": "\033[95m",       # Light magenta (pinkish)
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Màu level
        if record.levelname in COLORS:
            record.levelname = f"{COLORS[record.levelname]}{record.levelname}{RESET}"
        # Màu timestamp
        if "asctime" in self._fmt and hasattr(record, "asctime"):
            record.asctime = f"{FIELD_COLORS['asctime']}{record.asctime}{RESET}"
        # Màu thread
        record.threadName = f"{FIELD_COLORS['threadName']}{record.threadName}{RESET}"
        # Màu logger name
        record.name = f"{FIELD_COLORS['name']}{record.name}{RESET}"
        return super().format(record)

class MillisecondFormatter(ColorFormatter):
    def formatTime(self, record, datefmt=None):
        from datetime import datetime
        ct = datetime.fromtimestamp(record.created)
        s = ct.strftime(datefmt)
        return s[:-2]

handler = logging.StreamHandler()
formatter = MillisecondFormatter(
    "|%(levelname)s|%(asctime)s|%(threadName)s|%(name)s|%(message)s|",
    "%Y-%m-%d %H:%M:%S.%f"
)
handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[handler])
