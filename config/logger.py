import os
from loguru import logger
import sys

# Remover el handler por defecto
logger.remove()

# Configurar formato
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Handler para consola
logger.add(
    sys.stdout,
    format=log_format,
    level="INFO",
    colorize=True
)

__all__ = ["logger"]