# econ_models/io/file_utils.py
"""
File I/O utilities for loading and saving data.

This module provides safe file operations with proper error handling
and logging for configuration files and numerical artifacts.

Example:
    >>> from econ_models.io.file_utils import load_json_file, save_json_file
    >>> data = load_json_file("config.json")
    >>> save_json_file(data, "config_backup.json")
"""

import json
import os
import sys
import logging
from typing import Dict, Any

from econ_models.config.economic_params import EconomicParams

logger = logging.getLogger(__name__)


def load_json_file(filename: str) -> Dict[str, Any]:
    """
    Safely load a JSON file with comprehensive error handling.

    Args:
        filename: Path to the JSON file.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        SystemExit: If file not found, invalid JSON, or read error.
    """
    if not os.path.exists(filename):
        logger.error(f"File '{filename}' not found.")
        sys.exit(1)

    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        sys.exit(1)


def save_json_file(data: Dict[str, Any], filename: str) -> None:
    """
    Save data to a JSON file with directory creation.

    Args:
        data: Dictionary to serialize to JSON.
        filename: Target file path.

    Raises:
        IOError: If write operation fails.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved data to {filename}")
    except IOError as e:
        logger.error(f"Failed to save to {filename}: {e}")
        raise

def save_boundary_to_json(
    filename: str,
    bounds_data: Dict[str, float],
    params: EconomicParams
) -> None:
    """
    Save boundary data and source parameters to JSON.

    Args:
        filename: Target file path.
        bounds_data: Dictionary of boundary values.
        params: Economic parameters used to generate bounds.
    """
    export_data = {
        "bounds": bounds_data,
        "source_params": params.__dict__
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=4)
        logger.info(f"Boundaries saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save boundaries to {filename}: {e}")