
def format_price(value, decimals=2):
    """Format large numbers with K, M, B suffixes"""
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

def format_number(value, decimals=2):
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
