import numpy as np

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

# mol_hydrogens everywhere!


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
