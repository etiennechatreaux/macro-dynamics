import sys
from loguru import logger


def get_logger(name: str = "macrostate") -> "logger":
    """Get configured logger instance."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
        level="INFO",
        colorize=True,
    )
    return logger.bind(name=name)

