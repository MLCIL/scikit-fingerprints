import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from rdkit.Chem import Mol, MolToSmiles

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


# seconds an exhaustive search gets per molecule, as in Mordred
DEFAULT_TIMEOUT = 60.0


class SearchTimeoutError(Exception):
    """
    Raised by an exhaustive search that ran past the time it was given, or that is
    known in advance not to fit in it. Its message is the reason, as
    :func:`warn_search_given_up` reports it.
    """


def warn_search_given_up(descriptor: str, mol: Mol, error: SearchTimeoutError) -> None:
    """
    Report that ``descriptor`` abandoned its search of ``mol``.

    A timeout and a molecule too tangled to try both come out as NaN, as do a
    disconnected molecule and a failed calculation, so the reason is warned about
    rather than left for the caller to guess at.
    """
    warnings.warn(
        f"{descriptor} gave up on {MolToSmiles(mol)}: {error}. "
        f"Its features are returned as NaN.",
        stacklevel=3,
    )


def safe_value(func: Callable[..., float | int], *args: Any, **kwargs: Any) -> float:
    """
    Execute a descriptor function and return NaN for known calculation failures.
    """
    try:
        return float(func(*args, **kwargs))
    except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
        return np.nan
