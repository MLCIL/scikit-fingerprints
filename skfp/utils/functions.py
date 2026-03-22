from collections.abc import Sequence
from importlib.metadata import version

import pandas as pd
from rdkit import rdBase


def get_data_from_indices(data: Sequence, indices: Sequence[int]) -> list:
    """
    Retrieve elements from ``data`` at specified ``indices``. Works not only
    for Python lists but also for e.g. NumPy arrays and Pandas DataFrames and
    Series.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return [data.iloc[idx] for idx in indices]
    else:
        return [data[idx] for idx in indices]


def _get_sklearn_version():
    sklearn_ver = version("scikit-learn")  # e.g. 1.6.0
    sklearn_ver = ".".join(sklearn_ver.split(".")[:2])  # e.g. 1.6
    return float(sklearn_ver)


def _get_rdkit_version() -> tuple[int, int, int]:
    # Unlike scikit-learn which uses float (broken for minor >= 10, e.g. 2025.1 == 2025.10),
    # we return a tuple for correct ordering.
    rdkit_ver = rdBase.rdkitVersion  # e.g. "2025.09.3"
    parts = rdkit_ver.split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Cannot parse RDKit version: {rdkit_ver}")
    return int(parts[0]), int(parts[1]), int(parts[2])
