"""
Production reporting utilities for dataset processing and verification.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union


def save_json_report(data: Dict[str, Any], output_path: Path) -> None:
    """
    Saves a dictionary report as formatted JSON with timestamp metadata.

    Args:
        data (Dict[str, Any]): Dictionary containing execution metrics and results.
        output_path (Path): Path to output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, default=str)


def save_csv_report(
    rows: List[Dict[str, Any]], fieldnames: List[str], output_path: Path
) -> None:
    """
    Saves tabular data to a CSV file.

    Args:
        rows (List[Dict[str, Any]]): List of row dictionaries.
        fieldnames (List[str]): List of column names in order.
        output_path (Path): Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_report(content: str, output_path: Path) -> None:
    """
    Saves a markdown formatted report string to a file.

    Args:
        content (str): Markdown text.
        output_path (Path): Path to output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
