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

# bond order per bond type, as RDKit's GetBondTypeAsDouble reports it; molecules
# only ever carry these four types, and any other one stays NaN so that it shows up
# in the descriptors that use bond orders rather than passing as some other order
_BOND_ORDERS = np.full(max(int(bond_type) for bond_type in BondType.values) + 1, np.nan)
_BOND_ORDERS[[int(BondType.SINGLE), int(BondType.DOUBLE)]] = [1.0, 2.0]
_BOND_ORDERS[[int(BondType.TRIPLE), int(BondType.AROMATIC)]] = [3.0, 1.5]

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

_NUM_ELEMENTS = 119

# every element property of every element, as one (n_element_props, n_elements) table,
# so that a molecule reads all of its property values with a single indexing operation
# instead of one table lookup per property
ELEMENT_PROPERTY_MATRIX = np.vstack(
    [
        table.lookup(np.arange(_NUM_ELEMENTS))
        for table in ELEMENT_PROPERTY_TABLES.values()
    ]
)

# the carbon values again, in the row order of ELEMENT_PROPERTY_MATRIX
CARBON_ELEMENT_PROPERTIES = np.array(list(CARBON_PROPERTY_VALUES.values()))


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

# name -> its row in AtomicProperties.weighting_properties
_WEIGHTING_PROPERTY_ROWS: dict[str, int] = {
    name: row for row, name in enumerate(WEIGHTING_PROPERTY_NAMES)
}


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


def gasteiger_charges(mol: Mol) -> np.ndarray:
    """
    Gasteiger partial charge of every atom, hydrogens folded into their atom.
    """
    ComputeGasteigerCharges(mol)
    return atoms_apply_func(get_gasteiger_charge, mol)


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

    Note that from_mol() or with_hydrogens_added() should be used to calculate this.
    """

    def __init__(
        self,
        mol: Mol,
        *,
        atomic_nums: np.ndarray,
        is_aromatic: np.ndarray,
        formal_charges: np.ndarray,
        total_num_hs: np.ndarray,
        bond_begin_idxs: np.ndarray,
        bond_end_idxs: np.ndarray,
        bond_types: np.ndarray,
        bond_is_aromatic: np.ndarray,
        gasteiger_charges: np.ndarray,
    ):
        # properties read off the molecule
        self.mol = mol
        self.atomic_nums = atomic_nums
        self.is_aromatic = is_aromatic
        self.formal_charges = formal_charges
        # implicit and stored hydrogens, excluding neighbor hydrogen atoms
        self.total_num_hs = total_num_hs
        self.bond_begin_idxs = bond_begin_idxs
        self.bond_end_idxs = bond_end_idxs
        self.bond_types = bond_types
        self.bond_is_aromatic = bond_is_aromatic
        self.gasteiger_charges = gasteiger_charges

        self.num_atoms: int = len(atomic_nums)
        self.num_bonds: int = len(bond_types)
        self.is_hydrogen = atomic_nums == 1
        # heteroatom: any non-carbon atom, hydrogens included
        self.is_hetero = atomic_nums != 6
        self.outer_electrons = _N_OUTER_ELECS[atomic_nums]
        self.bond_orders = _BOND_ORDERS[bond_types]
        # the degree of an atom is the number of bonds it takes part in
        self.degrees = self._count_neighbors(np.ones(self.num_atoms, dtype=bool))

        # hydrogens per atom, counting neighbor hydrogen atoms as well, so that the
        # count is the same with and without explicit hydrogens
        self.num_hydrogens = total_num_hs + self._count_neighbors(self.is_hydrogen)
        # sigma electrons: the non-hydrogen neighbors of an atom
        self.sigma_electrons = self._count_neighbors(~self.is_hydrogen)
        self.valence_electrons = self._valence_electrons()
        self.intrinsic_state = self._intrinsic_state()

        # every element property of every atom, shape (n_element_props, n_atoms).
        # gathering columns of the table yields a Fortran-ordered array, and every
        # descriptor here reduces along the atom axis, so keep the rows contiguous
        self.element_properties = np.ascontiguousarray(
            ELEMENT_PROPERTY_MATRIX[:, atomic_nums]
        )
        # the same, followed by the connectivity properties, so that the descriptors
        # weighting atoms by each property in turn get them all in one array
        self.weighting_properties = np.vstack(
            [
                self.element_properties,
                *(getattr(self, attr) for attr in CONNECTIVITY_PROPERTY_ATTRS.values()),
            ]
        )

    @classmethod
    def from_mol(cls, mol: Mol) -> "AtomicProperties":
        """
        Read every property off a molecule.
        """
        return cls(
            mol,
            atomic_nums=atoms_apply_func(Atom.GetAtomicNum, mol, np.intp),
            is_aromatic=atoms_apply_func(Atom.GetIsAromatic, mol, bool),
            formal_charges=atoms_apply_func(Atom.GetFormalCharge, mol, np.intp),
            total_num_hs=atoms_apply_func(Atom.GetTotalNumHs, mol, np.intp),
            bond_begin_idxs=bonds_apply_func(Bond.GetBeginAtomIdx, mol, np.intp),
            bond_end_idxs=bonds_apply_func(Bond.GetEndAtomIdx, mol, np.intp),
            bond_types=bonds_apply_func(Bond.GetBondType, mol, np.intp),
            bond_is_aromatic=bonds_apply_func(Bond.GetIsAromatic, mol, bool),
            gasteiger_charges=gasteiger_charges(mol),
        )

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
        atoms and bonds in Python. Gasteiger charges are the exception: they
        redistribute over the whole molecule and have to be computed again.
        """
        num_added = mol_hydrogens.GetNumAtoms() - props.num_atoms
        if num_added != props.total_num_hs.sum():
            # AddHs did not do what is assumed above, so nothing can be reused
            return cls.from_mol(mol_hydrogens)

        # per-hydrogen values, in the order the hydrogens were appended
        parents = np.repeat(np.arange(props.num_atoms), props.total_num_hs)
        hydrogen_idxs = props.num_atoms + np.arange(num_added)
        # hydrogen has atomic number 1, one single bond, no charge and no hydrogens
        ones = np.ones(num_added, dtype=np.intp)
        zeros = np.zeros(num_added, dtype=np.intp)
        not_aromatic = np.zeros(num_added, dtype=bool)
        single_bonds = np.full(num_added, int(BondType.SINGLE), dtype=np.intp)

        return cls(
            mol_hydrogens,
            atomic_nums=np.concatenate([props.atomic_nums, ones]),
            is_aromatic=np.concatenate([props.is_aromatic, not_aromatic]),
            formal_charges=np.concatenate([props.formal_charges, zeros]),
            # the hydrogens are now neighbors rather than counts on their atom
            total_num_hs=np.zeros(mol_hydrogens.GetNumAtoms(), dtype=np.intp),
            # each new bond runs from an atom to one of its hydrogens
            bond_begin_idxs=np.concatenate([props.bond_begin_idxs, parents]),
            bond_end_idxs=np.concatenate([props.bond_end_idxs, hydrogen_idxs]),
            bond_types=np.concatenate([props.bond_types, single_bonds]),
            bond_is_aromatic=np.concatenate([props.bond_is_aromatic, not_aromatic]),
            gasteiger_charges=gasteiger_charges(mol_hydrogens),
        )

    def get(self, name: str) -> np.ndarray:
        """
        Weighting property array by name, for any of
        :data:`WEIGHTING_PROPERTY_NAMES`. The array is a row of
        :attr:`weighting_properties` and must not be modified.
        """
        row = _WEIGHTING_PROPERTY_ROWS.get(name)
        if row is None:
            raise KeyError(f'"{name}" is not an atomic weighting property')
        return self.weighting_properties[row]

    def _count_neighbors(self, atom_mask: np.ndarray) -> np.ndarray:
        """
        For every atom, the number of its neighbors satisfying the mask.
        """
        begins, ends = self.bond_begin_idxs, self.bond_end_idxs
        counts = np.bincount(begins, weights=atom_mask[ends], minlength=self.num_atoms)
        counts += np.bincount(ends, weights=atom_mask[begins], minlength=self.num_atoms)
        return counts.astype(np.intp)

    def _valence_electrons(self) -> np.ndarray:
        """
        Valence delta-value of every atom, as in :func:`get_valence_electrons`.
        """
        # the formal charge cancels out of the denominator: (Z - q) - (Zv - q) - 1
        numerator = self.outer_electrons - self.formal_charges - self.num_hydrogens
        denominator = self.atomic_nums - self.outer_electrons - 1
        return np.where(self.is_hydrogen, 0.0, numerator / denominator)

    def _intrinsic_state(self) -> np.ndarray:
        """
        Intrinsic state of every atom, as in :func:`get_intrinsic_state`.
        """
        periods = ELEMENT_PERIOD.lookup(self.atomic_nums)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (
                (2.0 / periods) ** 2 * self.valence_electrons + 1
            ) / self.sigma_electrons
        return np.where(self.sigma_electrons == 0, np.nan, values)
