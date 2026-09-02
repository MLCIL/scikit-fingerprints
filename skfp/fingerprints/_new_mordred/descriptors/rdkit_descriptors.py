import numpy as np
from rdkit.Chem import (
    Crippen,
    Descriptors,
    GraphDescriptors,
    Mol,
    MolSurf,
    rdMolDescriptors,
)
from rdkit.Chem.EState import EState_VSA

from skfp.fingerprints._new_mordred.utils.descriptor_evaluation import safe_value
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

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


def _calc_moe_type_descriptors(mol: Mol) -> list[float]:
    """
    Compute RDKit MOE-type VSA descriptors.

    Each VSA group splits approximate molecular surface area into bins based on
    atom-level properties such as partial charge, molar refractivity, logP, and
    E-State values.
    """
    return [
        *[getattr(MolSurf, f"PEOE_VSA{idx}")(mol) for idx in range(1, 14)],
        *[getattr(MolSurf, f"SMR_VSA{idx}")(mol) for idx in range(1, 10)],
        *[getattr(MolSurf, f"SlogP_VSA{idx}")(mol) for idx in range(1, 12)],
        *[getattr(EState_VSA, f"EState_VSA{idx}")(mol) for idx in range(1, 11)],
        *[getattr(EState_VSA, f"VSA_EState{idx}")(mol) for idx in range(1, 10)],
    ]


def _average_exact_mol_wt(mol: Mol) -> float:
    """
    Compute average exact molecular weight.

    The AMW descriptor is exact molecular weight divided by total atom count,
    including implicit hydrogens in the atom denominator.
    """
    return Descriptors.ExactMolWt(mol) / rdMolDescriptors.CalcNumAtoms(mol)


def calc_rdkit_2d(
    mol_regular: Mol,
    distance_matrix_regular: DistanceMatrix,
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
        rdMolDescriptors.CalcNumHBA(mol_regular),
        rdMolDescriptors.CalcNumHBD(mol_regular),
        MolSurf.LabuteASA(mol_regular),
        *_calc_moe_type_descriptors(mol_regular),
        Crippen.MolLogP(mol_regular),
        Crippen.MolMR(mol_regular),
        rdMolDescriptors.CalcTPSA(mol_regular),
        rdMolDescriptors.CalcTPSA(mol_regular, includeSandP=True),
        Descriptors.ExactMolWt(mol_regular),
        safe_value(_average_exact_mol_wt, mol_regular),
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
