"""Mordred descriptors implemented as direct RDKit wrappers.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

from collections.abc import Callable
from typing import Any

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

from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

FEATURE_NAMES_2D = [
    "nAtom",
    "nHeavyAtom",
    "nSpiro",
    "nBridgehead",
    "nHetero",
    "FCSP3",
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
    "nRing",
    "nHRing",
    "naRing",
    "naHRing",
    "nARing",
    "nAHRing",
    "nRot",
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "MW",
    "AMW",
]

FEATURE_NAMES_3D = ["MOMI-X", "MOMI-Y", "MOMI-Z", "PBF"]


def _safe_value(func: Callable[..., float | int], *args: Any, **kwargs: Any) -> float:
    """
    Execute a direct RDKit descriptor function and normalize its result.

    Mordred reports missing descriptor values as NaN when a calculation cannot
    be performed for a molecule. This helper mirrors that behavior for RDKit
    calls that fail with known numerical or chemistry-related exceptions.
    """
    try:
        return float(func(*args, **kwargs))
    except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
        return np.nan


def _append_moe_type_descriptors(values: list[float], mol: Mol) -> None:
    """
    Append RDKit MOE-type VSA descriptors to an existing value list.

    This covers the Mordred MoeType descriptor family implemented directly by
    RDKit: PEOE_VSA, SMR_VSA, SlogP_VSA, EState_VSA, and VSA_EState.
    """
    for idx in range(1, 14):
        descriptor_func = getattr(MolSurf, f"PEOE_VSA{idx}")
        values.append(_safe_value(descriptor_func, mol))

    for idx in range(1, 10):
        descriptor_func = getattr(MolSurf, f"SMR_VSA{idx}")
        values.append(_safe_value(descriptor_func, mol))

    for idx in range(1, 12):
        descriptor_func = getattr(MolSurf, f"SlogP_VSA{idx}")
        values.append(_safe_value(descriptor_func, mol))

    for idx in range(1, 11):
        descriptor_func = getattr(EState_VSA, f"EState_VSA{idx}")
        values.append(_safe_value(descriptor_func, mol))

    for idx in range(1, 10):
        descriptor_func = getattr(EState_VSA, f"VSA_EState{idx}")
        values.append(_safe_value(descriptor_func, mol))


def calc_2d(
    mol_regular: Mol,
    mol_with_hydrogens: Mol,
    distance_matrix_regular: DistanceMatrix,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute 2D Mordred descriptors available as direct RDKit calls.

    The returned descriptors include atom counts, graph descriptors, hydrogen
    bond counts, MOE-type VSA descriptors, ring counts, Crippen descriptors,
    topological polar surface area, and molecular weights.
    """
    values = [
        _safe_value(lambda: rdMolDescriptors.CalcNumAtoms(mol_with_hydrogens)),
        _safe_value(lambda: rdMolDescriptors.CalcNumHeavyAtoms(mol_regular)),
        _safe_value(lambda: rdMolDescriptors.CalcNumSpiroAtoms(mol_regular)),
        _safe_value(lambda: rdMolDescriptors.CalcNumBridgeheadAtoms(mol_regular)),
        _safe_value(lambda: rdMolDescriptors.CalcNumHeteroatoms(mol_regular)),
        _safe_value(lambda: rdMolDescriptors.CalcFractionCSP3(mol_regular)),
        _safe_value(
            lambda: GraphDescriptors.BalabanJ(
                mol_regular, dMat=distance_matrix_regular.matrix
            )
        ),
        _safe_value(
            lambda: GraphDescriptors.BertzCT(
                mol_regular, dMat=distance_matrix_regular.matrix
            )
        ),
        _safe_value(lambda: rdMolDescriptors.CalcNumHBA(mol_regular)),
        _safe_value(lambda: rdMolDescriptors.CalcNumHBD(mol_regular)),
        _safe_value(lambda: MolSurf.LabuteASA(mol_regular)),
    ]

    _append_moe_type_descriptors(values, mol_regular)

    values.extend(
        [
            _safe_value(lambda: rdMolDescriptors.CalcNumRings(mol_regular)),
            _safe_value(lambda: rdMolDescriptors.CalcNumHeterocycles(mol_regular)),
            _safe_value(lambda: rdMolDescriptors.CalcNumAromaticRings(mol_regular)),
            _safe_value(
                lambda: rdMolDescriptors.CalcNumAromaticHeterocycles(mol_regular)
            ),
            _safe_value(lambda: rdMolDescriptors.CalcNumAliphaticRings(mol_regular)),
            _safe_value(
                lambda: rdMolDescriptors.CalcNumAliphaticHeterocycles(mol_regular)
            ),
            _safe_value(lambda: rdMolDescriptors.CalcNumRotatableBonds(mol_regular)),
            _safe_value(lambda: Crippen.MolLogP(mol_regular)),
            _safe_value(lambda: Crippen.MolMR(mol_regular)),
            _safe_value(lambda: rdMolDescriptors.CalcTPSA(mol_regular)),
            _safe_value(
                lambda: rdMolDescriptors.CalcTPSA(mol_regular, includeSandP=True)
            ),
            _safe_value(lambda: Descriptors.ExactMolWt(mol_with_hydrogens)),
            _safe_value(
                lambda: Descriptors.ExactMolWt(mol_with_hydrogens)
                / mol_with_hydrogens.GetNumAtoms()
            ),
        ]
    )

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES_2D


def calc_3d(mol_with_3d_conformer: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute 3D Mordred descriptors available as direct RDKit calls.

    The moment of inertia descriptors are returned in Mordred axis order
    (MOMI-X, MOMI-Y, MOMI-Z), which corresponds to RDKit PMI3, PMI2, and PMI1.
    """
    values = [
        _safe_value(lambda: rdMolDescriptors.CalcPMI3(mol_with_3d_conformer)),
        _safe_value(lambda: rdMolDescriptors.CalcPMI2(mol_with_3d_conformer)),
        _safe_value(lambda: rdMolDescriptors.CalcPMI1(mol_with_3d_conformer)),
        _safe_value(lambda: rdMolDescriptors.CalcPBF(mol_with_3d_conformer)),
    ]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES_3D
