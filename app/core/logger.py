import logging
import sys
from app.core.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Creates a unified structured logger for the application."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(settings.LOG_LEVEL)
        
        # Create console handler with format
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(settings.LOG_LEVEL)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
    return logger
