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

    Based on Stanton, D. T., & Jurs, P. C. (1990). Development and use of
    charged partial surface area structural descriptors in computer-assisted
    quantitative structure-property relationship studies. Analytical Chemistry,
    62(21), 2323-2329. https://doi.org/10.1021/ac00220a013
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
    """
    Charged partial surface area (CPSA) descriptors.

    Combine per-atom solvent-accessible surface areas with Gasteiger partial
    charges to describe polar interactions on the molecular surface. All values
    are ``nan`` for an empty molecule.

    PNSA{v}/PPSA{v} (partial negative/positive surface area) sum the surface
    area of the negatively/positively charged atoms, weighted by a
    version-dependent charge factor ``f``:

        * v1: f = 1 (plain surface area)
        * v2: f = total charge of that sign
        * v3: f = the per-atom charge
        * v4: f = total charge of that sign / number of atoms
        * v5: f = total charge of that sign / number of atoms of that sign

    Descriptors derived from these:

        * DPSA{v} (difference in charged partial surface area): PPSA - PNSA
        * FNSA{v}/FPSA{v} (fractional charged partial negative/positive surface
          area): PNSA/PPSA divided by the total surface area
        * WNSA{v}/WPSA{v} (surface weighted charged partial negative/positive
          surface area): PNSA/PPSA times the total surface area / 1000
        * RNCS/RPCS (relative negative/positive charge surface area): surface
          area of the most extreme negatively/positively charged atom divided
          by the relative charge (RNCG/RPCG)
        * TASA/TPSA (total hydrophobic/polar surface area): sum of the surface
          area of atoms with |charge| below / at or above 0.2
        * RASA/RPSA (relative hydrophobic/polar surface area): TASA/TPSA divided
          by the total surface area

    Based on Stanton, D. T., & Jurs, P. C. (1990). Development and use of
    charged partial surface area structural descriptors in computer-assisted
    quantitative structure-property relationship studies. Analytical Chemistry,
    62(21), 2323-2329. https://doi.org/10.1021/ac00220a013
    """
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
    tasa, tpsa = _tasa_tpsa(gasteiger_charges_hydrogens, surface_area)

    values = [
        pnsa,
        ppsa,
        ppsa - pnsa,  # DPSA
        pnsa / surface_area_sum,  # FNSA
        ppsa / surface_area_sum,  # FPSA
        pnsa * surface_area_sum / 1000.0,  # WNSA
        ppsa * surface_area_sum / 1000.0,  # WPSA
        _rncs_rpcs(gasteiger_charges_hydrogens, masks, surface_area, rncg, rpcg),
        [tasa, tpsa],
        [tasa / surface_area_sum],  # RASA
        [tpsa / surface_area_sum],  # RPSA
    ]

    return np.concatenate(values, dtype=np.float32), FEATURE_NAMES_3D


def _pnsa_ppsa(
    gasteiger_charges_hydrogens: np.ndarray,
    masks: list[np.ndarray],
    surface_area: np.ndarray,
    num_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partial negative and positive surface area descriptors (PNSA{v}, PPSA{v}).

    For each sign, sums the surface area of the charged atoms weighted by a
    version-dependent factor ``f`` (see :func:`calc_3d`); ``nan`` for all five
    versions when no atom of that sign is present.
    """
    pnsa: list[float] = []
    ppsa: list[float] = []

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
    """
    Relative negative and positive charge surface area descriptors (RNCS, RPCS).

    Surface area of the atom carrying the most extreme charge of a given sign,
    divided by the relative charge of that sign (RNCG/RPCG); ``nan`` when no
    atom of that sign is present.
    """
    values = []

    for mask, desc in zip(masks, [rncg, rpcg], strict=True):
        charges = gasteiger_charges_hydrogens[mask]
        if charges.size == 0:
            values.append(np.nan)
        else:
            sa_max = surface_area[mask][np.argmax(np.abs(charges))]
            values.append(sa_max / desc)

    return np.asarray(values, dtype=np.float32)


def _tasa_tpsa(
    gasteiger_charges_hydrogens: np.ndarray, surface_area: np.ndarray
) -> tuple[float, float]:
    """
    Total hydrophobic and polar surface area descriptors (TASA, TPSA).

    TASA sums the surface area of hydrophobic atoms (|charge| < 0.2) and TPSA
    the surface area of polar atoms (|charge| >= 0.2); each is ``nan`` when no
    atom meets its condition.
    """
    abs_charges = np.abs(gasteiger_charges_hydrogens)
    tasa_mask = abs_charges < 0.2
    tpsa_mask = abs_charges >= 0.2

    tasa = np.sum(surface_area[tasa_mask]) if tasa_mask.any() else np.nan
    tpsa = np.sum(surface_area[tpsa_mask]) if tpsa_mask.any() else np.nan

    return tasa, tpsa
