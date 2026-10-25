"""Tests for data loading module."""

from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch

from src.data_loader import load_raw_dataset, download_raw_dataset


def test_load_raw_dataset_missing_file(tmp_path: Path):
    """Test that load_raw_dataset raises FileNotFoundError for non-existent file."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset file not found"):
        load_raw_dataset(non_existent)


def test_load_raw_dataset_empty_file(tmp_path: Path):
    """Test that load_raw_dataset raises ValueError on empty CSV file."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_dataset(empty_file)


def test_load_raw_dataset_valid_synthetic(tmp_path: Path):
    """Test loading a small synthetic CSV file."""
    synth_file = tmp_path / "synth.csv"
    content = "customerID,tenure,MonthlyCharges,Churn\nC001,12,50.0,No\nC002,1,75.0,Yes\n"
    synth_file.write_text(content, encoding="utf-8")

    df = load_raw_dataset(synth_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["customerID", "tenure", "MonthlyCharges", "Churn"]


def test_download_raw_dataset_existing_skip(tmp_path: Path):
    """Test that download_raw_dataset does not re-download if file exists and force=False."""
    existing_file = tmp_path / "existing.csv"
    existing_file.write_text("dummy content", encoding="utf-8")

    # Should immediately return path without making HTTP request
    result = download_raw_dataset(
        source_url="http://invalid.nonexistent.domain/fake.csv",
        destination_path=existing_file,
        force=False,
    )
    assert result == existing_file
    assert result.read_text(encoding="utf-8") == "dummy content"
