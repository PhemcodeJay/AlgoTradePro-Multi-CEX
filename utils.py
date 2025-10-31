# utils.py
import math
from typing import Union

Number = Union[int, float, str]


def round_to_precision(value: Number, precision: Union[int, float] = 0.0001) -> float:
    """
    Round a price/quantity to the nearest step size (precision).

    Parameters
    ----------
    value : int, float, str
        The number to round.
    precision : int, float
        The step size (e.g. 0.01 for 2-decimal tick, 0.0001 for 4-decimal).

    Returns
    -------
    float
        The rounded value.

    Examples
    --------
    >>> round_to_precision(12.3456, 0.01)
    12.35
    >>> round_to_precision(0.12345, 0.0001)
    0.1235
    """
    try:
        num = float(value)
        if precision <= 0:
            raise ValueError("precision must be > 0")
        return round(num / precision) * precision
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input for round_to_precision: {e}")


def format_price(value: Number, decimals: int = 2) -> str:
    """Format large numbers with K, M, B suffixes (with $ sign)"""
    try:
        num = float(value)
        if abs(num) >= 1_000_000_000:
            return f"${num / 1_000_000_000:.{decimals}f}B"
        elif abs(num) >= 1_000_000:
            return f"${num / 1_000_000:.{decimals}f}M"
        elif abs(num) >= 1_000:
            return f"${num / 1_000:.{decimals}f}K"
        else:
            return f"${num:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_number(value: Number, decimals: int = 2) -> str:
    """Format large numbers with K, M, B suffixes (without $ sign)"""
    try:
        num = float(value)
        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:.{decimals}f}B"
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.{decimals}f}M"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:.{decimals}f}K"
        else:
            return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)