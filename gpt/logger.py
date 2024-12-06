import logging
import json
from datetime import datetime, timezone
from rich.logging import RichHandler


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for log records."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "mode": record.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": record.step,
            "train_loss": record.train_loss,
            "val_loss": record.val_loss,
            "learning_rate": record.learning_rate,
            "step_time": record.step_time,
            "tokens_seen": record.tokens_seen,
            # "message": record.getMessage(),
        }
        return json.dumps(log_record)


class StepFilter(logging.Filter):
    """Custom filter to allow only log records that start with 'Step '."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "step")


def setup_logger(log_file: str = "training.log") -> logging.Logger:
    """Sets up a logger that outputs to the console using Rich and writes specific logs to a file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        # Console handler with Rich
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setLevel(logging.INFO)
        # rich_formatter = logging.Formatter("%(message)s")
        # rich_handler.setFormatter(rich_formatter)
        logger.addHandler(rich_handler)

        # File handler for logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.addFilter(StepFilter())

        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    return logger
