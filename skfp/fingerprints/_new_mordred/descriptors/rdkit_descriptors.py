import numpy as np
from rdkit.Chem import GraphDescriptors, Mol, rdMolDescriptors
from rdkit.Chem.EState.EState_VSA import estateBins, vsaBins

from skfp.fingerprints._new_mordred.utils.descriptor_evaluation import safe_value
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.molecular_properties import (
    MolecularProperties,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES_2D = [
    "BalabanJ",
    "BertzCT",
    "nHBAcc",
    "nHBDon",
    "LabuteASA",
    *[f"PEOE_VSA{i}" for i in range(1, 14)],
    *[f"SMR_VSA{i}" for i in range(1, 10)],
    *[f"SlogP_VSA{i}" for i in range(1, 12)],
    *[f"EState_VSA{i}" for i in range(1, 11)],
    *[f"VSA_EState{i}" for i in range(1, 10)],
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "MW",
    "AMW",
]

FEATURE_NAMES_3D = ["MOMI-Z", "MOMI-Y", "MOMI-X", "PBF"]


def _calc_moe_type_descriptors(mol: Mol, estate_indices: np.ndarray) -> list[float]:
    """
    Compute RDKit MOE-type VSA descriptors.

    Each VSA group splits approximate molecular surface area into bins based on
    atom-level properties such as partial charge, molar refractivity, logP, and
    E-State values.

    The charge, refractivity and logP groups are read from RDKit's C++ functions,
    which return all bins of a group at once; the per-bin functions in
    ``rdkit.Chem.MolSurf`` are Python wrappers around the very same values. The
    E-state groups are binned here instead, because RDKit does that in Python and
    would recompute the E-state indices it is given here to do it.
    """
    # per-atom surface areas; the second element is the hydrogen contribution
    surface_areas = np.asarray(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])

    # RDKit's own bin edges for the EState_VSA and VSA_EState descriptors; a bin
    # holds the values from its lower edge up to, but excluding, the next one
    estate_bins = np.searchsorted(estateBins, estate_indices, side="right")
    surface_area_bins = np.searchsorted(vsaBins, surface_areas, side="right")

    # surface area of the atoms in each E-state bin, and the other way round
    estate_vsa = np.bincount(
        estate_bins, weights=surface_areas, minlength=len(estateBins) + 1
    )
    vsa_estate = np.bincount(
        surface_area_bins, weights=estate_indices, minlength=len(vsaBins) + 1
    )
    return [
        *rdMolDescriptors.PEOE_VSA_(mol)[:13],
        *rdMolDescriptors.SMR_VSA_(mol)[:9],
        *rdMolDescriptors.SlogP_VSA_(mol)[:11],
        *estate_vsa[:10],
        *vsa_estate[:9],
    ]


def _average_exact_mol_wt(mol_properties: MolecularProperties) -> float:
    """
    Compute average exact molecular weight.

    The AMW descriptor is exact molecular weight divided by total atom count,
    including implicit hydrogens in the atom denominator.
    """
    return mol_properties.exact_mol_wt / mol_properties.num_atoms


def calc_rdkit_2d(
    mol_regular: Mol,
    distance_matrix_regular: DistanceMatrix,
    estate_indices: np.ndarray,
    mol_properties: MolecularProperties,
) -> np.ndarray:
    """
    Compute 2D descriptors that map directly to RDKit descriptor functions.
    """
    values = [
        safe_value(
            GraphDescriptors.BalabanJ,
            mol_regular,
            dMat=distance_matrix_regular.matrix,
        ),
        safe_value(
            GraphDescriptors.BertzCT,
            mol_regular,
            dMat=distance_matrix_regular.matrix,
        ),
        mol_properties.num_h_bond_acceptors,
        mol_properties.num_h_bond_donors,
        rdMolDescriptors.CalcLabuteASA(mol_regular),
        *_calc_moe_type_descriptors(mol_regular, estate_indices),
        mol_properties.log_p,
        mol_properties.molar_refractivity,
        rdMolDescriptors.CalcTPSA(mol_regular),
        rdMolDescriptors.CalcTPSA(mol_regular, includeSandP=True),
        mol_properties.exact_mol_wt,
        safe_value(_average_exact_mol_wt, mol_properties),
    ]

    return np.asarray(values, dtype=np.float32)


def calc_rdkit_3d(mol_with_3d_conformer: Mol) -> np.ndarray:
    """
    Compute 3D descriptors that map directly to RDKit descriptor functions.
    """
    values = [
        safe_value(rdMolDescriptors.CalcPMI1, mol_with_3d_conformer),
        safe_value(rdMolDescriptors.CalcPMI2, mol_with_3d_conformer),
        safe_value(rdMolDescriptors.CalcPMI3, mol_with_3d_conformer),
        safe_value(rdMolDescriptors.CalcPBF, mol_with_3d_conformer),
    ]

    return np.asarray(values, dtype=np.float32)
