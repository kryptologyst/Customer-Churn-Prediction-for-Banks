"""
Utility functions for reproducible experiments and device management.
"""

import random
import numpy as np
import torch
from typing import Optional, Union
import logging


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def log_system_info() -> None:
    """Log system information for reproducibility."""
    import platform
    import sys
    
    logging.info("=" * 50)
    logging.info("SYSTEM INFORMATION")
    logging.info("=" * 50)
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Architecture: {platform.architecture()}")
    logging.info(f"Processor: {platform.processor()}")
    
    # PyTorch info
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # NumPy info
    logging.info(f"NumPy version: {np.__version__}")
    
    logging.info("=" * 50)


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Ensure reproducible results across runs.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed)
    setup_logging()
    log_system_info()
    
    logging.info(f"Reproducibility setup complete with seed: {seed}")


def format_number(value: Union[int, float], precision: int = 4) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float):
        if abs(value) < 0.001:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.{precision}f}"
        elif abs(value) < 1000:
            return f"{value:.{precision}f}"
        else:
            return f"{value:,.{precision}f}"
    else:
        return str(value)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
