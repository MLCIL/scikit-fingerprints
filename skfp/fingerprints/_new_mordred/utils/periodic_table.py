from collections.abc import Callable
from pathlib import Path

import numpy as np

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


class PeriodicTable:
    """
    Periodic-table lookup table.

    Each instance maps a 1-based atomic number to a scalar property value.
    Data files are loaded once at module import time.

    Scalar lookups use ``table[atomic_num]`` and are backed by a plain list, which
    is faster than NumPy for single elements. Whole-molecule lookups should use
    :meth:`lookup`, which indexes a padded array with an atomic number vector.
    """

    __slots__ = ("_data", "_padded")
    _datadir = Path(__file__).parent / "data"

    def __init__(self, data: list[float]):
        self._data = data
        # index 0 and any atomic number past the end of the table read as NaN, so
        # that lookup() can index without bounds checks and match __getitem__
        self._padded = np.full(len(data) + 2, np.nan, dtype=np.float64)
        self._padded[1 : len(data) + 1] = data

    @classmethod
    def from_file(cls, name: str) -> "PeriodicTable":
        values: list[float] = []
        # example lines: "2.592 #   1 H", "-     #   2 He", "# comment"
        with open(cls._datadir / name) as file:
            for line in file:
                raw = line.split("#")[0].strip()
                if "-" in raw:
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(raw))
                    except ValueError:
                        continue
        return cls(values)

    def __getitem__(self, atomic_num: int) -> float:
        if atomic_num < 1:
            return np.nan
        try:
            return self._data[atomic_num - 1]
        except IndexError:
            return np.nan

    def lookup(self, atomic_nums: np.ndarray) -> np.ndarray:
        """
        Vectorized variant of ``__getitem__`` for an array of atomic numbers.
        Atomic numbers outside the table map to NaN.
        """
        idxs = np.where(atomic_nums < len(self._padded) - 1, atomic_nums, -1)
        return self._padded[idxs]

    def map(self, func: Callable[[float], float]) -> "PeriodicTable":
        """
        Build a new table by applying ``func`` to every value.
        """
        return PeriodicTable([func(value) for value in self._data])


ALLRED_ROCHOW_ELECTRONEGATIVITY = PeriodicTable.from_file(
    "allred_rocow_electron_negativity.txt"
)
IONIZATION_POTENTIAL = PeriodicTable.from_file("ionization_potential.txt")
MASS = PeriodicTable.from_file("mass.txt")
MC_GOWAN_VOLUME = PeriodicTable.from_file("mc_gowan_volume.txt")
PAULING_ELECTRONEGATIVITY = PeriodicTable.from_file("pauling_electron_negativity.txt")
POLARIZABILITY_78 = PeriodicTable.from_file("polarizalibity78.txt")
POLARIZABILITY_94 = PeriodicTable.from_file("polarizalibity94.txt")
SANDERSON_ELECTRONEGATIVITY = PeriodicTable.from_file(
    "sanderson_electron_negativity.txt"
)
VAN_DER_WAALS_RADII = PeriodicTable.from_file("van_der_waals_radii.txt")

# derived tables, so that every element property can be read the same way
ATOMIC_NUMBER = PeriodicTable([float(atomic_num) for atomic_num in range(1, 119)])
VAN_DER_WAALS_VOLUME = VAN_DER_WAALS_RADII.map(
    lambda radius: 4.0 / 3.0 * np.pi * radius**3
)

PERIOD = PeriodicTable(
    [1.0] * 2
    + [2.0] * 8
    + [3.0] * 8
    + [4.0] * 18
    + [5.0] * 18
    + [6.0] * 32
    + [7.0] * 32
)

HALOGEN_ATOMIC_NUMS: frozenset[int] = frozenset({9, 17, 35, 53, 85, 117})
