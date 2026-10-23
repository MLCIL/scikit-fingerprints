import numpy as np
from rdkit.Chem import GraphDescriptors, Mol, MolSurf, rdMolDescriptors
from rdkit.Chem.EState import EState_VSA

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
    *[f"PEOE VSA {i}" for i in range(1, 14)],
    *[f"SMR VSA {i}" for i in range(1, 10)],
    *[f"SlogP VSA {i}" for i in range(1, 12)],
    *[f"EState VSA {i}" for i in range(1, 11)],
    *[f"VSA EState {i}" for i in range(1, 10)],
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "MW",
    "AMW",
]

FEATURE_NAMES_3D = [
    "principal moment of inertia Z",
    "principal moment of inertia Y",
    "principal moment of inertia X",
    "PBF",
]


def _calc_moe_type_descriptors(mol: Mol) -> list[float]:
    """
    Compute RDKit MOE-type VSA descriptors.

    Each VSA group splits approximate molecular surface area into bins based on
    atom-level properties such as partial charge, molar refractivity, logP, and
    E-State values.
    """
    return [
        *rdMolDescriptors.PEOE_VSA_(mol)[:13],
        *rdMolDescriptors.SMR_VSA_(mol)[:9],
        *rdMolDescriptors.SlogP_VSA_(mol)[:11],
        # force=False reuses the E-state indices that the EState descriptors
        # already computed and RDKit cached on the molecule
        *EState_VSA.EState_VSA_(mol, force=False)[:10],
        *EState_VSA.VSA_EState_(mol, force=False)[:9],
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
    mol_hydrogens: Mol,
    distance_matrix_regular: DistanceMatrix,
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
        MolSurf.LabuteASA(mol_regular),
        *_calc_moe_type_descriptors(mol_regular),
        mol_properties.log_p,
        mol_properties.molar_refractivity,
        rdMolDescriptors.CalcTPSA(mol_hydrogens),
        rdMolDescriptors.CalcTPSA(mol_hydrogens, includeSandP=True),
        mol_properties.exact_mol_wt,
        safe_value(_average_exact_mol_wt, mol_properties),
    ]

    return np.asarray(values, dtype=np.float32)


def calc_rdkit_3d(mol_hydrogens: Mol, conf_id: int) -> np.ndarray:
    """
    Compute 3D descriptors that map directly to RDKit descriptor functions, on
    the conformer ``conf_id``.
    """
    values = [
        safe_value(rdMolDescriptors.CalcPMI1, mol_hydrogens, confId=conf_id),
        safe_value(rdMolDescriptors.CalcPMI2, mol_hydrogens, confId=conf_id),
        safe_value(rdMolDescriptors.CalcPMI3, mol_hydrogens, confId=conf_id),
        safe_value(rdMolDescriptors.CalcPBF, mol_hydrogens, confId=conf_id),
    ]

    return np.asarray(values, dtype=np.float32)
