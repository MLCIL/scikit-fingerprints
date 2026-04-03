import os

import numpy as np

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


class PeriodicTable:
    """Periodic-table lookup tables.

    Data files are loaded once at module import time.
    Each PeriodicTable instance maps atomic_number (1-based) -> value.
    """

    __slots__ = ("_data",)
    _datadir = os.path.join(os.path.dirname(__file__), "data")

    def __init__(self, data: list[float]):
        self._data = data

    @classmethod
    def from_file(cls, name: str, conv: type = float) -> "PeriodicTable":
        values: list[float] = []
        with open(os.path.join(cls._datadir, name)) as f:
            for line in f:
                raw = line.split("#")[0].strip()
                if "-" in raw:
                    values.append(np.nan)
                else:
                    try:
                        values.append(conv(raw))
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


MASS = PeriodicTable.from_file("mass.txt")
VDW_RADII = PeriodicTable.from_file("van_der_waals_radii.txt")
VDW_VOLUME = PeriodicTable([4.0 / 3.0 * np.pi * r**3 for r in VDW_RADII._data])
SANDERSON_EN = PeriodicTable.from_file("sanderson_electron_negativity.txt")
PAULING_EN = PeriodicTable.from_file("pauling_electron_negativity.txt")
ALLRED_ROCOW_EN = PeriodicTable.from_file("allred_rocow_electron_negativity.txt")
POLARIZABILITY_94 = PeriodicTable.from_file("polarizalibity94.txt")
POLARIZABILITY_78 = PeriodicTable.from_file("polarizalibity78.txt")
IONIZATION_POTENTIAL = PeriodicTable.from_file("ionization_potential.txt")
MC_GOWAN_VOLUME = PeriodicTable.from_file("mc_gowan_volume.txt")

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
