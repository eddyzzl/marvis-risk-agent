from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd


_YYYYMM_RE = re.compile(r"^\d{6}$")
_YYYYMMDD_RE = re.compile(r"^\d{8}$")
_INTEGER_FLOAT_RE = re.compile(r"^(\d+)\.0$")
_EXPLICIT_TIME_FORMATS = (
    "%Y%m%d",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
)


def date_key_series(values, *, column_name: str) -> pd.Series:
    dates = parse_time_series(values, column_name=column_name)
    if dates.empty:
        return pd.Series([], index=dates.index, dtype="object")
    return dates.dt.strftime("%Y%m%d")


def month_key_series(values, *, column_name: str) -> pd.Series:
    dates = parse_time_series(values, column_name=column_name)
    if dates.empty:
        return pd.Series([], index=dates.index, dtype="object")
    return dates.dt.strftime("%Y%m")


def parse_time_series(values, *, column_name: str) -> pd.Series:
    source = pd.Series(values, copy=False)
    if source.empty:
        return pd.Series([], index=source.index, dtype="datetime64[ns]")

    parsed = pd.Series(
        [_parse_normalized_time_value(value) for value in source.map(_normalize_time_value)],
        index=source.index,
        dtype="datetime64[ns]",
    )
    invalid_mask = parsed.isna()
    if invalid_mask.any():
        bad_values = [
            repr(value)
            for value in source[invalid_mask].head(5).tolist()
        ]
        raise ValueError(
            f"time column {column_name!r} contains unparseable date values: "
            + ", ".join(bad_values)
        )
    return pd.Series(parsed, index=source.index).dt.normalize()


def _normalize_time_value(value) -> str | datetime | date:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value
    if pd.isna(value):
        return ""

    text = str(value).strip()
    integer_float_match = _INTEGER_FLOAT_RE.match(text)
    if integer_float_match:
        text = integer_float_match.group(1)
    if _YYYYMM_RE.match(text):
        return f"{text}01"
    if _YYYYMMDD_RE.match(text):
        return text
    return text


def _parse_normalized_time_value(value) -> pd.Timestamp | pd.NaT:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return _timestamp_without_timezone(value)

    text = str(value).strip()
    if not text:
        return pd.NaT

    for date_format in _EXPLICIT_TIME_FORMATS:
        try:
            return _timestamp_without_timezone(datetime.strptime(text, date_format))
        except ValueError:
            pass

    try:
        iso_text = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        return _timestamp_without_timezone(datetime.fromisoformat(iso_text))
    except ValueError:
        pass

    parsed = pd.to_datetime(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return pd.NaT
    return _timestamp_without_timezone(parsed)


def _timestamp_without_timezone(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp
