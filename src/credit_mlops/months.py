"""Helpers for month-granular dates (always `date` on day 1)."""

from datetime import date, datetime


def parse_month(s: str) -> date:
    """Parse a `YYYY-MM` string into the first day of that month."""
    return datetime.strptime(s, "%Y-%m").date().replace(day=1)


def month_label(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def add_months(d: date, n: int) -> date:
    index = d.year * 12 + (d.month - 1) + n
    return date(index // 12, index % 12 + 1, 1)


def months_between(a: date, b: date) -> int:
    """Number of months from `a` to `b` (b - a)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def month_range(start: date, end: date) -> list[date]:
    """All months from `start` to `end`, both inclusive."""
    return [add_months(start, i) for i in range(months_between(start, end) + 1)]
