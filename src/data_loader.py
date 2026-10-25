"""Data loading utilities for Northstar SaaS Churn & LTV Predictor."""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    REMOTE_DATASET_URL,
    ID_COLUMN,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def download_raw_dataset(
    source_url: str = REMOTE_DATASET_URL,
    destination_path: Union[str, Path] = RAW_DATA_PATH,
    force: bool = False,
) -> Path:
    """Download the raw dataset from a remote URL to the specified local path.

    Args:
        source_url: Remote URL pointing to the raw dataset CSV.
        destination_path: Local destination path.
        force: If True, re-download even if the file already exists.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If download fails due to network error or invalid URL.
    """
    dest = Path(destination_path)
    if dest.exists() and not force:
        logger.info(f"Dataset already exists at {dest}. Skipping download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset from {source_url} to {dest}...")

    try:
        urllib.request.urlretrieve(source_url, dest)
        size_bytes = dest.stat().st_size
        logger.info(f"Download complete: {dest} ({size_bytes} bytes).")
        return dest
    except Exception as e:
        logger.error(f"Failed to download dataset from {source_url}: {e}")
        raise RuntimeError(
            f"Could not download dataset from '{source_url}'. Error: {e}"
        ) from e


def load_raw_dataset(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the raw churn CSV into a pandas DataFrame without mutating the file.

    Args:
        filepath: Path to the raw CSV file. Defaults to config.RAW_DATA_PATH.

    Returns:
        pd.DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or missing expected basic structure.
    """
    target_path = Path(filepath) if filepath else RAW_DATA_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at '{target_path}'. "
            f"Please run download_raw_dataset() or place the CSV in data/raw/."
        )

    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file at '{target_path}': {e}") from e

    if df.empty:
        raise ValueError(f"The loaded dataset from '{target_path}' is empty.")

    logger.info(f"Loaded raw dataset from {target_path} with shape {df.shape}.")
    return df
