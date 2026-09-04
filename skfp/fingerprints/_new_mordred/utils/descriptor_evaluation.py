from collections.abc import Callable
from typing import Any

import numpy as np

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def safe_value(func: Callable[..., float | int], *args: Any, **kwargs: Any) -> float:
    """
    Execute a descriptor function and return NaN for known calculation failures.
    """
    try:
        return float(func(*args, **kwargs))
    except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
        return np.nan
