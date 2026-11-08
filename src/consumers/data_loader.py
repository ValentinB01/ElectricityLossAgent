"""Chunked CSV validation and chronological calendar alignment."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ConsumerData:
    ids: np.ndarray
    labels: np.ndarray
    values: np.ndarray
    dates: pd.DatetimeIndex
    report: dict[str, Any]

    @property
    def mask(self) -> np.ndarray:
        """Usable original measurements, before clipping, scaling or imputation."""
        return np.isfinite(self.values)


def inspect_header(path: Path, dates_config: dict) -> tuple[list[str], pd.DatetimeIndex, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names (including duplicate dates) detected")
    if not {"CONS_NO", "FLAG"}.issubset(header):
        raise ValueError("CSV must contain CONS_NO and FLAG")
    date_columns = [c for c in header if c not in ("CONS_NO", "FLAG")]
    if not date_columns:
        raise ValueError("CSV contains no daily measurement columns")
    parsed = []
    for name in date_columns:
        if not re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", name):
            raise ValueError(f"Unexpected column or invalid date format: {name!r}; expected YYYY/M/D")
        try:
            parsed.append(pd.Timestamp(datetime.strptime(name, "%Y/%m/%d")))
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date: {name}") from exc
    if len(parsed) != len(set(parsed)):
        raise ValueError("Duplicate dates after parsing column names")
    calendar = pd.date_range(dates_config["start"], dates_config["end"], freq="D")
    positions = calendar.get_indexer(parsed)
    if (positions < 0).any():
        raise ValueError("CSV has dates outside the configured calendar; refusing to discard data")
    return header, calendar, positions


def load_dataset(path: str | Path, config: dict, audit_path: Path | None = None) -> ConsumerData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    header, calendar, positions = inspect_header(path, config["dates"])
    # A narrow first pass allocates one numeric matrix without retaining all string chunks.
    identity = pd.read_csv(path, usecols=["CONS_NO", "FLAG"], dtype=str,
                           keep_default_na=False, encoding="utf-8-sig")
    if identity.empty:
        raise ValueError("The input CSV contains no consumers")
    if identity.CONS_NO.str.strip().eq("").any():
        raise ValueError("CONS_NO must not be empty")
    if identity.CONS_NO.duplicated().any():
        raise ValueError("CONS_NO must be unique; duplicate consumers detected")
    label_values = pd.to_numeric(identity.FLAG, errors="coerce")
    if not label_values.isin([0, 1]).all():
        raise ValueError("FLAG must contain only the binary labels 0 and 1")
    ids = np.asarray(identity.CONS_NO.tolist(), dtype=str)
    labels = label_values.to_numpy(dtype=np.int8)
    values = np.full((len(ids), len(calendar)), np.nan, dtype=np.float64)
    date_columns = [c for c in header if c not in ("CONS_NO", "FLAG")]
    report: dict[str, Any] = {
        "raw_dimensions": [len(ids), len(header)], "unique_consumers": len(ids),
        "original_date_columns": len(date_columns), "expected_calendar_days": len(calendar),
        "missing_calendar_dates": [d.strftime("%Y-%m-%d") for d in calendar
                                   if d not in set(calendar[positions])],
        "raw_missing_measurements": 0, "invalid_numeric_values": 0,
        "infinite_values": 0, "negative_measurements": 0, "suspiciously_large_values": 0,
        "raw_finite_min": None, "raw_finite_max": None, "issue_examples": [],
        "invalid_policy": config["data"]["invalid_policy"],
        "negative_policy": config["data"]["negative_policy"],
        "suspicious_value_threshold": config["data"]["suspicious_value_threshold"],
        "warnings": [],
    }
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["CONS_NO", "date", "kind", "raw_value"])
    offset = 0
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False,
                             chunksize=config["data"]["chunksize"], encoding="utf-8-sig"):
        raw = chunk[date_columns]
        missing = raw.apply(lambda s: s.str.strip().str.lower().isin(
            ["", "nan", "na", "null", "none", "n/a"])).to_numpy(dtype=bool)
        numeric = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        invalid = np.isnan(numeric) & ~missing
        infinite = np.isinf(numeric)
        negative = np.isfinite(numeric) & (numeric < 0)
        suspicious = np.isfinite(numeric) & (numeric > config["data"]["suspicious_value_threshold"])
        report["raw_missing_measurements"] += int(missing.sum())
        issues = []
        for kind, bad in [("invalid_numeric_values", invalid), ("infinite_values", infinite),
                          ("negative_measurements", negative), ("suspiciously_large_values", suspicious)]:
            report[kind] += int(bad.sum())
            for row, col in zip(*np.nonzero(bad)):
                issues.append([str(chunk.CONS_NO.iloc[row]), calendar[positions[col]].strftime("%Y-%m-%d"),
                               kind, str(raw.iloc[row, col])])
        if audit_path is not None and issues:
            with audit_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(issues)
        report["issue_examples"].extend(issues[:max(0, 20 - len(report["issue_examples"]))])
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            for key, val, reduce in [("raw_finite_min", float(finite.min()), min),
                                     ("raw_finite_max", float(finite.max()), max)]:
                report[key] = val if report[key] is None else reduce(report[key], val)
        if (invalid.any() or infinite.any()) and config["data"]["invalid_policy"] == "error":
            raise ValueError(f"Invalid/infinite numeric measurements near CSV row {offset + 2}; see audit")
        if negative.any() and config["data"]["negative_policy"] == "error":
            raise ValueError(f"Negative measurements near CSV row {offset + 2}; see audit")
        numeric[missing | invalid | infinite] = np.nan
        numeric[negative] = 0 if config["data"]["negative_policy"] == "clip_zero" else np.nan
        values[offset:offset + len(chunk), positions] = numeric
        offset += len(chunk)
    if offset != len(ids):
        raise ValueError("CSV row count changed while reading")
    for key in ("invalid_numeric_values", "infinite_values", "negative_measurements", "suspiciously_large_values"):
        if report[key]:
            report["warnings"].append(f"{key}: {report[key]}; see measurement_issues.csv and handling policies")
    if report["missing_calendar_dates"]:
        report["warnings"].append("Absent calendar columns were inserted as entirely missing measurements")
    return ConsumerData(ids, labels, values, calendar, report)
