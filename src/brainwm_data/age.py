"""Conservative conversion of heterogeneous developmental ages.

The source string is always retained. Numeric values use birth as day zero;
negative values are prenatal. Censored/unknown values do not receive an
invented point estimate.
"""

from __future__ import annotations

import math
import re
from typing import Any


DAYS_PER_YEAR = 365.2425
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0
DAYS_FROM_CONCEPTION_TO_BIRTH = 266.0
DAYS_FROM_LMP_TO_BIRTH = 280.0

# Approximate midpoint in days post conception. The explicit basis flag keeps
# these values distinguishable from directly reported pcw values.
CARNEGIE_MIDPOINT_DPC = {
    1: 0.5, 2: 2.0, 3: 3.0, 4: 5.0, 5: 8.0, 6: 13.0, 7: 15.5,
    8: 18.0, 9: 20.5, 10: 22.5, 11: 24.5, 12: 26.5, 13: 28.5,
    14: 32.0, 15: 35.0, 16: 38.0, 17: 41.0, 18: 44.0,
    19: 47.0, 20: 50.0, 21: 52.0, 22: 54.0, 23: 57.0,
}

ORDINAL_MONTH = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10,
}


def _empty(raw: str, status: str = "unknown") -> dict[str, Any]:
    return {
        "age_raw": raw,
        "time_point_days": None,
        "time_lower_days": None,
        "time_upper_days": None,
        "time_basis": None,
        "time_status": status,
        "life_stage": "unknown",
    }


def _result(raw: str, point: float | None, lower: float | None,
            upper: float | None, basis: str, status: str,
            life_stage: str) -> dict[str, Any]:
    return {
        "age_raw": raw,
        "time_point_days": point,
        "time_lower_days": lower,
        "time_upper_days": upper,
        "time_basis": basis,
        "time_status": status,
        "life_stage": life_stage,
    }


def parse_age(value: Any) -> dict[str, Any]:
    """Parse an age without silently treating plain ``w`` as prenatal.

    ``pcw`` is interpreted as weeks post conception. Plain day/week/month/year
    units are postnatal, matching the conservative policy in the source audit.
    Ranges expose lower/upper bounds; inequality and unknown values have no
    point estimate.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _empty("")
    raw = str(value).strip()
    text = raw.lower().strip()
    if not text or text in {"unknown", "nan", "na", "n/a", "none"}:
        return _empty(raw)
    if text == "adult stage":
        return _empty(raw, "categorical_only") | {"life_stage": "adult"}
    if text == "infant stage":
        return _empty(raw, "categorical_only") | {"life_stage": "postnatal"}
    if text == "newborn stage (0-28 days)":
        return _result(raw, 14.0, 0.0, 28.0, "postnatal_day_range",
                       "range_midpoint", "postnatal")

    m = re.fullmatch(r"carnegie\s+stage\s+(\d+)", text)
    if m:
        stage = int(m.group(1))
        if stage in CARNEGIE_MIDPOINT_DPC:
            point = CARNEGIE_MIDPOINT_DPC[stage] - DAYS_FROM_CONCEPTION_TO_BIRTH
            return _result(raw, point, point, point,
                           "carnegie_midpoint_days_post_conception",
                           "approximate", "prenatal")
        return _empty(raw, "unsupported_stage") | {"life_stage": "prenatal"}

    m = re.fullmatch(r"(" + "|".join(ORDINAL_MONTH) + r")\s+lmp\s+month\s+stage", text)
    if m:
        month = ORDINAL_MONTH[m.group(1)]
        point = month * DAYS_PER_MONTH - DAYS_FROM_LMP_TO_BIRTH
        return _result(raw, point, point, point, "lmp_month", "approximate", "prenatal")

    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*pcw", text)
    if m:
        point = float(m.group(1)) * 7.0 - DAYS_FROM_CONCEPTION_TO_BIRTH
        return _result(raw, point, point, point, "weeks_post_conception", "exact", "prenatal")

    # Postnatal unit conversion. A plain `w` is deliberately postnatal.
    units = {"d": 1.0, "w": 7.0, "m": DAYS_PER_MONTH,
             "mo": DAYS_PER_MONTH, "y": DAYS_PER_YEAR}
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(d|w|m|mo|y)", text)
    if m:
        point = float(m.group(1)) * units[m.group(2)]
        stage = "adult" if point >= 18 * DAYS_PER_YEAR else "postnatal"
        return _result(raw, point, point, point, f"postnatal_{m.group(2)}", "exact", stage)

    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*(d|w|m|mo|y)", text)
    if m:
        lower = float(m.group(1)) * units[m.group(3)]
        upper = float(m.group(2)) * units[m.group(3)]
        stage = "adult" if lower >= 18 * DAYS_PER_YEAR else "postnatal"
        return _result(raw, (lower + upper) / 2.0, lower, upper,
                       f"postnatal_{m.group(3)}_range", "range_midpoint", stage)

    m = re.fullmatch(r"([<>])\s*([0-9]+(?:\.[0-9]+)?)\s*(d|w|m|mo|y)", text)
    if m:
        boundary = float(m.group(2)) * units[m.group(3)]
        lower = boundary if m.group(1) == ">" else None
        upper = boundary if m.group(1) == "<" else None
        stage = "adult" if boundary >= 18 * DAYS_PER_YEAR else "postnatal"
        return _result(raw, None, lower, upper, f"postnatal_{m.group(3)}_censored",
                       "right_censored" if m.group(1) == ">" else "left_censored", stage)

    return _empty(raw, "unparsed")
