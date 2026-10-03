import numpy as np

"""
Helpers for working with rows of unequal length, kept flat and described by their
lengths, as they arise from per-atom and per-bond neighborhoods.
"""


def run_starts(counts: np.ndarray) -> np.ndarray:
    """
    Start offset of every row with the given lengths.
    """
    return np.cumsum(counts) - counts


def ragged_indices(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Index arrays for the flattened concatenation of rows with the given lengths:
    for every element, the row it belongs to and its position within that row.
    """
    owner = np.repeat(np.arange(counts.size), counts)
    within = np.arange(int(counts.sum())) - np.repeat(run_starts(counts), counts)
    return owner, within
