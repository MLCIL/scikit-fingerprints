from collections.abc import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func

from .periodic_table import (
    ALLRED_ROCHOW_ELECTRONEGATIVITY,
    IONIZATION_POTENTIAL,
    MASS,
    MC_GOWAN_VOLUME,
    PAULING_ELECTRONEGATIVITY,
    PERIOD,
    POLARIZABILITY_94,
    SANDERSON_ELECTRONEGATIVITY,
    VAN_DER_WAALS_RADII,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


_RDKIT_PERIODIC_TABLE = Chem.GetPeriodicTable()


def get_element_symbol(atomic_num: int) -> str:
    return _RDKIT_PERIODIC_TABLE.GetElementSymbol(atomic_num)


def get_atomic_number_from_symbol(symbol: str) -> int:
    return _RDKIT_PERIODIC_TABLE.GetAtomicNumber(symbol)


def get_atomic_number(atom: Atom) -> int:
    return atom.GetAtomicNum()


def get_mass(atom: Atom) -> float:
    return MASS[atom.GetAtomicNum()]


def get_van_der_waals_radius_rdkit(atom: Atom) -> float:
    # radius used by RDKit
    return _RDKIT_PERIODIC_TABLE.GetRvdw(atom.GetAtomicNum())


def get_van_der_waals_radius(atom: Atom) -> float:
    # radius used by Mordred & PaDEL-Descriptor
    return VAN_DER_WAALS_RADII[atom.GetAtomicNum()]


def get_van_der_waals_volume(atom: Atom) -> float:
    return 4.0 / 3.0 * np.pi * VAN_DER_WAALS_RADII[atom.GetAtomicNum()] ** 3


def get_sanderson_electronegativity(atom: Atom) -> float:
    return SANDERSON_ELECTRONEGATIVITY[atom.GetAtomicNum()]


def get_pauling_electronegativity(atom: Atom) -> float:
    return PAULING_ELECTRONEGATIVITY[atom.GetAtomicNum()]


def get_allred_rochow_electronegativity(atom: Atom) -> float:
    return ALLRED_ROCHOW_ELECTRONEGATIVITY[atom.GetAtomicNum()]


def get_polarizability(atom: Atom) -> float:
    return POLARIZABILITY_94[atom.GetAtomicNum()]


def get_ionization_potential(atom: Atom) -> float:
    return IONIZATION_POTENTIAL[atom.GetAtomicNum()]


def get_mc_gowan_volume(atom: Atom) -> float:
    return MC_GOWAN_VOLUME[atom.GetAtomicNum()]


def get_gasteiger_charge(atom: Atom) -> float:
    return (
        atom.GetDoubleProp("_GasteigerCharge") + atom.GetDoubleProp("_GasteigerHCharge")
        if atom.HasProp("_GasteigerHCharge")
        else 0.0
    )


def gasteiger_charges(mol: Mol) -> np.ndarray:
    """
    Per-atom Gasteiger partial charges, shared by the 2D and 3D descriptors.

    Each value includes the charge of the atom's implicit hydrogens folded back
    onto the heavy atom (see :func:`get_gasteiger_charge`). Charges depend only on
    connectivity, so the same array is valid regardless of 3D coordinates.
    """
    ComputeGasteigerCharges(mol)
    return atoms_apply_func(get_gasteiger_charge, mol)


def get_sigma_electrons(atom: Atom) -> int:
    """
    Return the number of sigma (single-bond framework) electrons on an atom,
    approximated as the count of its non-hydrogen neighbors.
    """
    return sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() != 1)


def get_valence_electrons(atom: Atom) -> float:
    """
    Valence delta-value used in molecular connectivity indices.

    Based on Kier, L. B., & Hall, L. H. (1983). General definition of
    valence delta-values for molecular connectivity. Journal of
    Pharmaceutical Sciences, 72(10), 1170-1173.
    https://doi.org/10.1002/jps.2600721016
    """
    N = atom.GetAtomicNum()
    if N == 1:
        return 0.0
    Zv = _RDKIT_PERIODIC_TABLE.GetNOuterElecs(N) - atom.GetFormalCharge()
    Z = N - atom.GetFormalCharge()
    h = atom.GetTotalNumHs() + sum(
        1 for a in atom.GetNeighbors() if a.GetAtomicNum() == 1
    )
    return (Zv - h) / (Z - Zv - 1)


def get_intrinsic_state(atom: Atom) -> float:
    """
    Intrinsic state value used in electrotopological-state (E-state) indices.

    See the Molconn-Z 4.00 manual, chapter 2, p. 283:
    http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html.
    """
    d = get_sigma_electrons(atom)
    if d == 0:
        return np.nan
    dv = get_valence_electrons(atom)
    return ((2.0 / PERIOD[atom.GetAtomicNum()]) ** 2 * dv + 1) / d


PROPERTY_FUNCS: dict[str, Callable[[Atom], float]] = {
    "atomic_number": get_atomic_number,
    "mass": get_mass,
    "van_der_Waals_volume": get_van_der_waals_volume,
    "Sanderson_electronegativity": get_sanderson_electronegativity,
    "Pauling_electronegativity": get_pauling_electronegativity,
    "Allred_Rochow_electronegativity": get_allred_rochow_electronegativity,
    "polarizability": get_polarizability,
    "ionization_potential": get_ionization_potential,
}
