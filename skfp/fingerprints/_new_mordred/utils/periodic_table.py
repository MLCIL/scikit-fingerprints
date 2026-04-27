from pathlib import Path

import numpy as np
from rdkit.Chem import GetPeriodicTable

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
    """

    __slots__ = ("_data",)
    _datadir = Path(__file__).parent / "data"

    def __init__(self, data: list[float]):
        self._data = data

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


SANDERSON_EN = PeriodicTable.from_file("sanderson_electron_negativity.txt")
PAULING_EN = PeriodicTable.from_file("pauling_electron_negativity.txt")
ALLRED_ROCOW_EN = PeriodicTable.from_file("allred_rocow_electron_negativity.txt")
POLARIZABILITY_94 = PeriodicTable.from_file("polarizalibity94.txt")
POLARIZABILITY_78 = PeriodicTable.from_file("polarizalibity78.txt")
IONIZATION_POTENTIAL = PeriodicTable.from_file("ionization_potential.txt")
MASS = PeriodicTable.from_file("mass.txt")
MORDRED_VDW_RADII = PeriodicTable.from_file("van_der_waals_radii.txt")
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

_RDKIT_PT = GetPeriodicTable()


def mass(atomic_num: int) -> float:
    return MASS[atomic_num]


def vdw_radii(atomic_num: int) -> float:
    return _RDKIT_PT.GetRvdw(atomic_num)


def vdw_volume(atomic_num: int) -> float:
    return 4.0 / 3.0 * np.pi * MORDRED_VDW_RADII[atomic_num] ** 3
