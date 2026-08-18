import numpy as np
from rdkit.Chem import GetMolFrags, Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    gasteiger_charges,
    get_allred_rochow_electronegativity,
    get_atomic_number,
    get_intrinsic_state,
    get_ionization_potential,
    get_mass,
    get_pauling_electronegativity,
    get_polarizability,
    get_sanderson_electronegativity,
    get_sigma_electrons,
    get_valence_electrons,
    get_van_der_waals_volume,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


# atomic properties used to weight distance matrices
_PROPS_NAMES = [
    "atomic_number",
    "mass",
    "van_der_Waals_volume",
    "Sanderson_electronegativity",
    "Pauling_electronegativity",
    "Allred_Rochow_electronegativity",
    "polarizability",
    "ionization_potential",
    "valence_electrons",
    "sigma_electrons",  # http://dx.doi.org/10.1002%2Fjps.2600721016
    "intrinsic_state",  # http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html, p.283
    "gasteiger_charge",
]
_PROPS_FUNCS = [
    get_atomic_number,
    get_mass,
    get_van_der_waals_volume,
    get_sanderson_electronegativity,
    get_pauling_electronegativity,
    get_allred_rochow_electronegativity,
    get_polarizability,
    get_ionization_potential,
    get_valence_electrons,
    get_sigma_electrons,
    get_intrinsic_state,
    None,  # Gasteiger charges are calculated separately
]

FEATURE_NAMES = [
    f"BCUT_{prop}_{kind}_eigval"
    for prop in _PROPS_NAMES
    for kind in ["smallest", "largest"]
]


def calc(mol: Mol) -> np.ndarray:
    """
    BCUT descriptors.
    """
    burden_matrix = _get_burden_matrix(mol)

    if len(GetMolFrags(mol)) > 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    values = []
    for name, func in zip(_PROPS_NAMES, _PROPS_FUNCS, strict=True):
        if name == "gasteiger_charge":
            props = gasteiger_charges(mol)
        else:
            props = atoms_apply_func(func, mol, np.float32)  # type: ignore

        # some properties are undefined for exotic elements, e.g. Gasteiger charges
        # for metals; the eigendecomposition cannot run then
        if np.any(np.isnan(props)):
            values.extend([np.nan, np.nan])
            continue

        np.fill_diagonal(burden_matrix, props)
        eigvals = np.linalg.eigvalsh(burden_matrix)  # ascending order

        smallest = eigvals[0]
        largest = eigvals[-1]

        values.extend([smallest, largest])

    return np.asarray(values, dtype=np.float32)


def _get_burden_matrix(mol: Mol) -> np.ndarray:
    num_atoms = mol.GetNumAtoms()

    mat = 0.001 * np.ones((num_atoms, num_atoms))

    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        i = a.GetIdx()
        j = b.GetIdx()

        w = bond.GetBondTypeAsDouble() / 10.0

        if a.GetDegree() == 1 or b.GetDegree() == 1:
            w += 0.01

        mat[i, j] = w
        mat[j, i] = w

    return mat
