import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.sasa import solvent_accessible_surface_area

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_VERSIONS = range(1, 6)

_SINGLE_2D = ("RNCG", "RPCG")

_VERSIONED_3D = ("PNSA", "PPSA", "DPSA", "FNSA", "FPSA", "WNSA", "WPSA")
_SINGLE_3D = ("RNCS", "RPCS", "TASA", "TPSA", "RASA", "RPSA")


FEATURE_NAMES_2D = [*_SINGLE_2D]

FEATURE_NAMES_3D = [
    *[f"{desc}{v}" for desc in _VERSIONED_3D for v in _VERSIONS],
    *_SINGLE_3D,
]


def calc_2d(gasteiger_charges_hydrogens: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Relative negative (RNCG) and relative positive (RPCG) charge descriptors.

    Each is the most extreme partial charge of a given sign divided by the
    total charge of that sign; ``0.0`` when no atom of that sign is present.
    Charge-only, so no 3D conformer is required.
    """
    masks = [
        gasteiger_charges_hydrogens < 0.0,  # RNCG
        gasteiger_charges_hydrogens > 0.0,  # RPCG
    ]

    values = []

    for mask in masks:
        charges = gasteiger_charges_hydrogens[mask]
        if charges.size == 0:
            values.append(np.nan)
        else:
            q_max = charges[np.argmax(np.abs(charges))]
            values.append(q_max / np.sum(charges))

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES_2D


def calc_3d(
    mol_hydrogens_conformer: Mol,
    cpsa_2d: tuple[np.ndarray, list[str]],
    gasteiger_charges_hydrogens: np.ndarray,
):
    num_atoms = mol_hydrogens_conformer.GetNumAtoms()
    if num_atoms == 0:
        return np.full(
            len(FEATURE_NAMES_3D), np.nan, dtype=np.float32
        ), FEATURE_NAMES_3D

    rncg, rpcg = cpsa_2d[0]
    surface_area = solvent_accessible_surface_area(mol_hydrogens_conformer)
    surface_area_sum = surface_area.sum()
    masks = [
        gasteiger_charges_hydrogens < 0.0,  # negative
        gasteiger_charges_hydrogens > 0.0,  # positive
    ]

    pnsa, ppsa = _pnsa_ppsa(gasteiger_charges_hydrogens, masks, surface_area, num_atoms)

    values = [
        pnsa,
        ppsa,
        ppsa - pnsa,  # DPSA
        pnsa / surface_area_sum,  # FNSA
        ppsa / surface_area_sum,  # FPSA
        pnsa * surface_area_sum / 1000.0,  # WNSA
        ppsa * surface_area_sum / 1000.0,  # WPSA
        _rncs_rpcs(gasteiger_charges_hydrogens, masks, surface_area, rncg, rpcg),
    ]

    return np.concatenate(values, dtype=np.float32), FEATURE_NAMES_3D


def _pnsa_ppsa(
    gasteiger_charges_hydrogens: np.ndarray,
    masks: list[np.ndarray],
    surface_area: np.ndarray,
    num_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    pnsa: list[np.floating] = []
    ppsa: list[np.floating] = []

    for mask, desc in zip(masks, [pnsa, ppsa], strict=True):
        charges = gasteiger_charges_hydrogens[mask]
        if charges.size == 0:
            desc.extend([np.nan] * 5)
        else:
            sa = surface_area[mask]
            charges_sum = np.sum(charges)

            factors = [
                1.0,
                charges_sum,
                charges,
                charges_sum / num_atoms,
                charges_sum / charges.size,
            ]
            desc.extend(np.sum(f * sa) for f in factors)

    return np.array(pnsa), np.array(ppsa)


def _rncs_rpcs(
    gasteiger_charges_hydrogens: np.ndarray,
    masks: list[np.ndarray],
    surface_area: np.ndarray,
    rncg: np.ndarray,
    rpcg: np.ndarray,
) -> np.ndarray:
    values = []

    for mask, desc in zip(masks, [rncg, rpcg], strict=True):
        charges = gasteiger_charges_hydrogens[mask]
        if charges.size == 0:
            values.append(np.nan)
        else:
            sa_max = surface_area[mask][np.argmax(np.abs(charges))]
            values.append(sa_max / desc)

    return np.asarray(values, dtype=np.float32)
