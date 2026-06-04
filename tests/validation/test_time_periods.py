import pandas as pd
import pytest

from riskmodel_checker.validation import time_periods


def test_date_keys_do_not_depend_on_pandas_mixed_format_support(monkeypatch):
    original_to_datetime = pd.to_datetime

    def old_pandas_to_datetime(values, *args, **kwargs):
        if kwargs.get("format") == "mixed":
            return pd.Series(pd.NaT, index=pd.Series(values).index)
        return original_to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(time_periods.pd, "to_datetime", old_pandas_to_datetime)

    values = pd.Series([
        "2025-04-21",
        "2025/05/28",
        "20250625",
        "2025-04-03 13:20:00",
        20250302,
    ])

    assert time_periods.date_key_series(values, column_name="apply_dt").tolist() == [
        "20250421",
        "20250528",
        "20250625",
        "20250403",
        "20250302",
    ]
    assert time_periods.month_key_series(values, column_name="apply_dt").tolist() == [
        "202504",
        "202505",
        "202506",
        "202504",
        "202503",
    ]


def test_unparseable_time_values_still_raise_clear_column_error():
    with pytest.raises(ValueError, match=r"time column 'apply_dt' contains unparseable date values: 'not-a-date'"):
        time_periods.date_key_series(pd.Series(["2025-04-21", "not-a-date"]), column_name="apply_dt")


def test_timezone_aware_business_dates_preserve_local_wall_clock_day():
    values = pd.Series(["2025-04-01T00:30:00+08:00", "2025-03-31T23:30:00-04:00"])

    assert time_periods.date_key_series(values, column_name="apply_dt").tolist() == [
        "20250401",
        "20250331",
    ]
    assert time_periods.month_key_series(values, column_name="apply_dt").tolist() == [
        "202504",
        "202503",
    ]
