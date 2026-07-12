from math import pi

import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.descriptors.ring_count import (
    _ring_atom_sets,
    _ring_properties,
)

"""
van der Waals volume (ABC) descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Vabc"]

# Bondi van der Waals radii (in angstroms) per atomic element
_BONDI_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "P": 1.80,
    "S": 1.80,
    "As": 1.85,
    "B": 2.13,
    "Si": 2.10,
    "Se": 1.90,
}

# Per-atom van der Waals volume contribution, i.e. sphere volume 4/3 * pi * r^3.
_ATOM_CONTRIB = {symbol: 4.0 / 3.0 * pi * r**3 for symbol, r in _BONDI_RADII.items()}


def calc(mol_regular: Mol, mol_hydrogens: Mol) -> tuple[np.ndarray, list[str]]:
    r"""
    Compute the Mordred ABC van der Waals volume descriptor.

    `Vabc` follows Zhao et al. (:doi:`10.1021/jo034808o`):

    .. math::
        V_{abc} = \sum_i V_i - 5.92 N_b - 14.7 R_a - 3.8 R_A

    where :math:`V_i` are per-atom Bondi sphere volumes (including hydrogens),
    :math:`N_b` is the total number of bonds, :math:`R_a` the number of aromatic
    rings, and :math:`R_A` the number of non-aromatic rings. Returns NaN when the
    molecule contains an atom without a defined Bondi radius.

    Atom volumes and bonds use the hydrogen-explicit molecule, while rings are
    counted on ``mol_regular`` whose aromaticity is reliably perceived (unlike
    ``AddHs``, ``RemoveHs`` re-sanitizes the molecule).
    """
    try:
        atom_volume = sum(
            _ATOM_CONTRIB[atom.GetSymbol()] for atom in mol_hydrogens.GetAtoms()
        )
    except KeyError:
        return np.asarray([np.nan], dtype=np.float32), FEATURE_NAMES

    n_bonds = mol_hydrogens.GetNumBonds()

    # reuse ring detection from the ring count descriptor: simple (non-fused)
    # aromatic and non-aromatic rings
    rings = _ring_properties(mol_regular, _ring_atom_sets(mol_regular))
    n_aromatic_rings = sum(1 for ring in rings if ring.is_aromatic)
    n_aliphatic_rings = sum(1 for ring in rings if not ring.is_aromatic)

    vabc = (
        atom_volume - 5.92 * n_bonds - 14.7 * n_aromatic_rings - 3.8 * n_aliphatic_rings
    )

    return np.asarray([vabc], dtype=np.float32), FEATURE_NAMES
