from functools import cached_property

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, BondType, Mol
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    atoms_apply_func,
    bonds_apply_func,
)

from .periodic_table import (
    ALLRED_ROCHOW_ELECTRONEGATIVITY,
    ATOMIC_NUMBER,
    ELEMENT_PERIOD,
    IONIZATION_POTENTIAL,
    MASS,
    MC_GOWAN_VOLUME,
    PAULING_ELECTRONEGATIVITY,
    POLARIZABILITY_94,
    SANDERSON_ELECTRONEGATIVITY,
    VAN_DER_WAALS_RADII,
    VAN_DER_WAALS_VOLUME,
    PeriodicTable,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


_RDKIT_PERIODIC_TABLE = Chem.GetPeriodicTable()

# number of outer (valence) electrons per atomic number
_N_OUTER_ELECS = np.array(
    [0] + [_RDKIT_PERIODIC_TABLE.GetNOuterElecs(num) for num in range(1, 119)],
    dtype=np.intp,
)

# element properties are based only on atomic number and can be read from tables

# name -> table, in the order the descriptors expose these properties as features
ELEMENT_PROPERTY_TABLES: dict[str, PeriodicTable] = {
    "atomic_number": ATOMIC_NUMBER,
    "mass": MASS,
    "van_der_Waals_volume": VAN_DER_WAALS_VOLUME,
    "Sanderson_electronegativity": SANDERSON_ELECTRONEGATIVITY,
    "Pauling_electronegativity": PAULING_ELECTRONEGATIVITY,
    "Allred_Rochow_electronegativity": ALLRED_ROCHOW_ELECTRONEGATIVITY,
    "polarizability": POLARIZABILITY_94,
    "ionization_potential": IONIZATION_POTENTIAL,
}

# value of each element property for a carbon atom, used to normalize property values
CARBON_PROPERTY_VALUES: dict[str, float] = {
    name: table[6] for name, table in ELEMENT_PROPERTY_TABLES.items()
}


def get_element_symbol(atomic_num: int) -> str:
    return _RDKIT_PERIODIC_TABLE.GetElementSymbol(atomic_num)


def get_atomic_number_from_symbol(symbol: str) -> int:
    return _RDKIT_PERIODIC_TABLE.GetAtomicNumber(symbol)


def get_atomic_number(atom: Atom) -> int:
    return atom.GetAtomicNum()


def get_mass(atom: Atom) -> float:
    # element mass from Mordred data tables, different from RDKit ones
    return MASS[atom.GetAtomicNum()]


def get_van_der_waals_radius_rdkit(atom: Atom) -> float:
    # radius used by RDKit
    return _RDKIT_PERIODIC_TABLE.GetRvdw(atom.GetAtomicNum())


def get_van_der_waals_radius(atom: Atom) -> float:
    # radius used by Mordred & PaDEL-Descriptor
    return VAN_DER_WAALS_RADII[atom.GetAtomicNum()]


def get_van_der_waals_volume(atom: Atom) -> float:
    return VAN_DER_WAALS_VOLUME[atom.GetAtomicNum()]


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


# connectivity properties, depending on atom neighborhood

# name -> AtomicProperties attribute holding the corresponding array
CONNECTIVITY_PROPERTY_ATTRS: dict[str, str] = {
    "valence_electrons": "valence_electrons",
    "sigma_electrons": "sigma_electrons",
    "intrinsic_state": "intrinsic_state",
    "gasteiger_charge": "gasteiger_charges",
}

# every property usable as an atom weight, in the order descriptors expose them
WEIGHTING_PROPERTY_NAMES: list[str] = [
    *ELEMENT_PROPERTY_TABLES,
    *CONNECTIVITY_PROPERTY_ATTRS,
]


def get_sigma_electrons(atom: Atom) -> int:
    """
    Return the number of sigma (single-bond framework) electrons on an atom,
    approximated as the count of its non-hydrogen neighbors.

    See http://dx.doi.org/10.1002%2Fjps.2600721016.
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
    return ((2.0 / ELEMENT_PERIOD[atom.GetAtomicNum()]) ** 2 * dv + 1) / d


def get_gasteiger_charge(atom: Atom) -> float:
    """
    Gasteiger charge of an atom plus that of the hydrogens attached to it.

    Requires molecule after calling ComputeGasteigerCharges() on it.
    """
    return (
        atom.GetDoubleProp("_GasteigerCharge") + atom.GetDoubleProp("_GasteigerHCharge")
        if atom.HasProp("_GasteigerHCharge")
        else 0.0
    )


class AtomicProperties:
    """
    Per-molecule atom and bond property arrays.
    """

    def __init__(self, mol: Mol):
        self.mol = mol
        self.num_atoms: int = mol.GetNumAtoms()
        self.num_bonds: int = mol.GetNumBonds()

    @classmethod
    def with_hydrogens_added(
        cls, mol_hydrogens: Mol, props: "AtomicProperties"
    ) -> "AtomicProperties":
        """
        Properties of ``AddHs(mol)``, derived from the properties of ``mol``.

        ``AddHs`` leaves the heavy atoms and their bonds untouched and appends one
        hydrogen per implicit hydrogen, grouped by the atom they belong to, in
        ascending order of that atom. Every array below therefore follows from the
        hydrogen-suppressed molecule, which saves a second pass over twice as many
        atoms and bonds in Python.
        """
        num_hydrogens = props.total_num_hs  # the hydrogens AddHs makes explicit
        num_added = mol_hydrogens.GetNumAtoms() - props.num_atoms
        derived = cls(mol_hydrogens)
        if num_added != num_hydrogens.sum():
            # AddHs did not do what is assumed above, so nothing can be reused
            return derived

        # per-hydrogen values, in the order the hydrogens were appended
        parents = np.repeat(np.arange(props.num_atoms), num_hydrogens)
        hydrogen_idxs = props.num_atoms + np.arange(num_added)
        # hydrogen has atomic number 1, exactly one bond, no charge and no hydrogens
        ones = np.ones(num_added, dtype=np.intp)
        zeros = np.zeros(num_added, dtype=np.intp)
        not_aromatic = np.zeros(num_added, dtype=bool)
        single_bonds = np.full(num_added, int(BondType.SINGLE), dtype=np.intp)

        derived._prefill(
            atomic_nums=np.concatenate([props.atomic_nums, ones]),
            formal_charges=np.concatenate([props.formal_charges, zeros]),
            is_aromatic=np.concatenate([props.is_aromatic, not_aromatic]),
            # each hydrogen bond also raises the degree of the atom it belongs to
            degrees=np.concatenate([props.degrees + num_hydrogens, ones]),
            # the hydrogens are now the neighbors they used to be counted as
            num_hydrogens=np.concatenate([props.num_hydrogens, zeros]),
            # each new bond runs from an atom to one of its hydrogens
            bond_begin_idxs=np.concatenate([props.bond_begin_idxs, parents]),
            bond_end_idxs=np.concatenate([props.bond_end_idxs, hydrogen_idxs]),
            bond_types=np.concatenate([props.bond_types, single_bonds]),
            bond_orders=np.concatenate([props.bond_orders, np.ones(num_added)]),
            bond_is_aromatic=np.concatenate([props.bond_is_aromatic, not_aromatic]),
        )
        return derived

    @classmethod
    def kekulized(
        cls, mol_kekulized: Mol, props: "AtomicProperties"
    ) -> "AtomicProperties":
        """
        Properties of the kekulized ``mol``, derived from the properties of ``mol``.

        Kekulization only rewrites bond types, leaving the atoms, the bonds they
        connect and the aromaticity flags alone, so everything but the bond types
        and orders carries over.
        """
        derived = cls(mol_kekulized)
        derived._prefill(
            atomic_nums=props.atomic_nums,
            formal_charges=props.formal_charges,
            is_aromatic=props.is_aromatic,
            degrees=props.degrees,
            total_num_hs=props.total_num_hs,
            bond_begin_idxs=props.bond_begin_idxs,
            bond_end_idxs=props.bond_end_idxs,
        )
        return derived

    def _prefill(self, **arrays: np.ndarray) -> None:
        """
        Store property arrays that are already known, so that the matching
        :func:`cached_property` never reads them off the molecule.
        """
        self.__dict__.update(arrays)

    def get(self, name: str) -> np.ndarray:
        """
        Weighting property array by name, for any of
        :data:`WEIGHTING_PROPERTY_NAMES`.
        """
        table = ELEMENT_PROPERTY_TABLES.get(name)
        if table is not None:
            return table.lookup(self.atomic_nums)

        attr = CONNECTIVITY_PROPERTY_ATTRS.get(name)
        if attr is None:
            raise KeyError(f'"{name}" is not an atomic weighting property')
        return getattr(self, attr).astype(np.float64)

    # atomic properties

    @cached_property
    def atomic_nums(self) -> np.ndarray:
        return atoms_apply_func(Atom.GetAtomicNum, self.mol, np.intp)

    @cached_property
    def is_hydrogen(self) -> np.ndarray:
        return self.atomic_nums == 1

    @cached_property
    def is_aromatic(self) -> np.ndarray:
        return atoms_apply_func(Atom.GetIsAromatic, self.mol, bool)

    @cached_property
    def degrees(self) -> np.ndarray:
        return atoms_apply_func(Atom.GetDegree, self.mol, np.intp)

    @cached_property
    def formal_charges(self) -> np.ndarray:
        return atoms_apply_func(Atom.GetFormalCharge, self.mol, np.intp)

    @cached_property
    def rdkit_masses(self) -> np.ndarray:
        # atom masses as reported by RDKit, honoring isotopes
        return atoms_apply_func(Atom.GetMass, self.mol, np.float64)

    @cached_property
    def total_num_hs(self) -> np.ndarray:
        # RDKit ``GetTotalNumHs``
        # implicit + stored H count, excluding neighbor H atoms
        return atoms_apply_func(Atom.GetTotalNumHs, self.mol, np.intp)

    @cached_property
    def num_hydrogens(self) -> np.ndarray:
        # total hydrogens per atom, counting neighbor H atoms
        # same value with or without explicit Hs
        return self.total_num_hs + self._neighbor_counts(self.is_hydrogen)

    # bond properties

    @cached_property
    def bond_begin_idxs(self) -> np.ndarray:
        return self._bond_endpoints[0]

    @cached_property
    def bond_end_idxs(self) -> np.ndarray:
        return self._bond_endpoints[1]

    @cached_property
    def bond_types(self) -> np.ndarray:
        return bonds_apply_func(Bond.GetBondType, self.mol, np.intp)

    @cached_property
    def bond_is_aromatic(self) -> np.ndarray:
        return bonds_apply_func(Bond.GetIsAromatic, self.mol, bool)

    @cached_property
    def bond_orders(self) -> np.ndarray:
        return bonds_apply_func(Bond.GetBondTypeAsDouble, self.mol, np.float64)

    @cached_property
    def _bond_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            bonds_apply_func(Bond.GetBeginAtomIdx, self.mol, np.intp),
            bonds_apply_func(Bond.GetEndAtomIdx, self.mol, np.intp),
        )

    # connectivity properties

    @cached_property
    def sigma_electrons(self) -> np.ndarray:
        return self._neighbor_counts(~self.is_hydrogen)

    @cached_property
    def outer_electrons(self) -> np.ndarray:
        """Number of outer (valence) shell electrons of the neutral atom."""
        return _N_OUTER_ELECS[self.atomic_nums]

    @cached_property
    def valence_electrons(self) -> np.ndarray:
        atomic_nums = self.atomic_nums
        outer_elecs = self.outer_electrons

        # the formal charge cancels out of the denominator: (Z - q) - (Zv - q) - 1
        numerator = outer_elecs - self.formal_charges - self.num_hydrogens
        denominator = atomic_nums - outer_elecs - 1

        return np.where(self.is_hydrogen, 0.0, numerator / denominator)

    @cached_property
    def intrinsic_state(self) -> np.ndarray:
        sigma_electrons = self.sigma_electrons
        periods = ELEMENT_PERIOD.lookup(self.atomic_nums)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (
                (2.0 / periods) ** 2 * self.valence_electrons + 1
            ) / sigma_electrons
        return np.where(sigma_electrons == 0, np.nan, values)

    @cached_property
    def gasteiger_charges(self) -> np.ndarray:
        ComputeGasteigerCharges(self.mol)
        return atoms_apply_func(get_gasteiger_charge, self.mol)

    def _neighbor_counts(self, atom_mask: np.ndarray) -> np.ndarray:
        # for every atom, count neighbors satisfying atom_mask
        begins = self.bond_begin_idxs
        ends = self.bond_end_idxs
        counts = np.bincount(begins, weights=atom_mask[ends], minlength=self.num_atoms)
        counts += np.bincount(ends, weights=atom_mask[begins], minlength=self.num_atoms)
        return counts.astype(np.intp)
