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
    NUM_ELEMENTS,
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
BOND_ORDERS = np.full(max(int(bond_type) for bond_type in BondType.values) + 1, np.nan)
BOND_ORDERS[[int(BondType.SINGLE), int(BondType.DOUBLE)]] = [1.0, 2.0]
BOND_ORDERS[[int(BondType.TRIPLE), int(BondType.AROMATIC)]] = [3.0, 1.5]

# number of outer (valence) electrons per atomic number
_N_OUTER_ELECS = np.array(
    [0] + [_RDKIT_PERIODIC_TABLE.GetNOuterElecs(num) for num in range(1, NUM_ELEMENTS)],
    dtype=np.intp,
)

# van der Waals radius used by RDKit, defined for every element
VAN_DER_WAALS_RADII_RDKIT = PeriodicTable(
    [_RDKIT_PERIODIC_TABLE.GetRvdw(num) for num in range(1, NUM_ELEMENTS)]
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

# every element property of every element, as one (n_element_props, n_elements) table,
# so that a molecule reads all of its property values with a single indexing operation
# instead of one table lookup per property
ELEMENT_PROPERTY_MATRIX = np.vstack(
    [
        table.lookup(np.arange(NUM_ELEMENTS))
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
    Per-molecule atom and bond property arrays, read off the molecule once and
    shared by the descriptors.
    """

    def __init__(self, mol: Mol):
        # properties read off the molecule
        self.mol = mol
        self.atomic_nums = atoms_apply_func(Atom.GetAtomicNum, mol, np.intp)
        self.is_aromatic = atoms_apply_func(Atom.GetIsAromatic, mol, bool)
        self.formal_charges = atoms_apply_func(Atom.GetFormalCharge, mol, np.intp)
        # implicit and stored hydrogens, excluding neighbor hydrogen atoms
        self.total_num_hs = atoms_apply_func(Atom.GetTotalNumHs, mol, np.intp)
        self.bond_begin_idxs = bonds_apply_func(Bond.GetBeginAtomIdx, mol, np.intp)
        self.bond_end_idxs = bonds_apply_func(Bond.GetEndAtomIdx, mol, np.intp)
        self.bond_types = bonds_apply_func(Bond.GetBondType, mol, np.intp)
        self.bond_is_aromatic = bonds_apply_func(Bond.GetIsAromatic, mol, bool)
        self.gasteiger_charges = gasteiger_charges(mol)

        atomic_nums = self.atomic_nums
        self.num_atoms: int = len(atomic_nums)
        self.num_bonds: int = len(self.bond_types)
        self.is_hydrogen = atomic_nums == 1
        self.is_carbon = atomic_nums == 6
        # heteroatom: any non-carbon atom, hydrogens included
        self.is_hetero = ~self.is_carbon
        self.outer_electrons = _N_OUTER_ELECS[atomic_nums]
        self.bond_orders = BOND_ORDERS[self.bond_types]
        # the degree of an atom is the number of bonds it takes part in
        self.degrees = self.count_neighbors(np.ones(self.num_atoms, dtype=bool))

        # hydrogens per atom, counting neighbor hydrogen atoms as well, so that the
        # count is the same with and without explicit hydrogens
        self.num_hydrogens = self.total_num_hs + self.count_neighbors(self.is_hydrogen)
        # sigma electrons: the non-hydrogen neighbors of an atom
        self.sigma_electrons = self.count_neighbors(~self.is_hydrogen)
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

    def sum_over_bonds(
        self, at_begin: np.ndarray, at_end: np.ndarray | None = None
    ) -> np.ndarray:
        """
        For every atom, the sum of per-bond values over the bonds it takes part in.

        ``at_begin[i]`` is added to the atom at the beginning of bond ``i``, and
        ``at_end[i]`` to the atom at its end. Without ``at_end``, both atoms of a
        bond get the same value, ``at_begin[i]``.
        """
        if at_end is None:
            at_end = at_begin
        totals = np.bincount(
            self.bond_begin_idxs, weights=at_begin, minlength=self.num_atoms
        )
        totals += np.bincount(
            self.bond_end_idxs, weights=at_end, minlength=self.num_atoms
        )
        return totals

    def count_neighbors(self, atom_mask: np.ndarray) -> np.ndarray:
        """
        For every atom, the number of its neighbors satisfying the mask.
        """
        begins, ends = self.bond_begin_idxs, self.bond_end_idxs
        return self.sum_over_bonds(atom_mask[ends], atom_mask[begins]).astype(np.intp)

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
