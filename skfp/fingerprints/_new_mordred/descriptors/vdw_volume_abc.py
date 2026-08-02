from math import pi

import numpy as np

from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    AtomicProperties,
    get_atomic_number_from_symbol,
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
# Bondi sphere volume per atomic number, NaN where the element has no Bondi radius
_ATOM_VOLUMES = np.full(119, np.nan)
for _symbol, _radius in _BONDI_RADII.items():
    _ATOM_VOLUMES[get_atomic_number_from_symbol(_symbol)] = 4.0 / 3.0 * pi * _radius**3


def calc(rings_regular: RingSets, props_hydrogens: AtomicProperties) -> np.ndarray:
    r"""
    Compute the Mordred ABC van der Waals volume descriptor.

    `Vabc` follows Zhao et al. (:doi:`10.1021/jo034808o`):

    .. math::
        V_{abc} = \sum_i V_i - 5.92 N_b - 14.7 R_a - 3.8 R_A

    where :math:`V_i` are per-atom Bondi sphere volumes (including hydrogens),
    :math:`N_b` is the total number of bonds, :math:`R_a` the number of aromatic
    rings, and :math:`R_A` the number of non-aromatic rings. Returns NaN when the
    molecule contains an atom without a defined Bondi radius.

    Atom volumes and bonds use the hydrogen-explicit molecule, while rings come
    from ``rings_regular``, built on the hydrogen-suppressed molecule whose
    aromaticity is reliably perceived (unlike ``AddHs``, ``RemoveHs``
    re-sanitizes the molecule).
    """
    # an element without a Bondi radius makes the whole descriptor undefined
    atom_volume = _ATOM_VOLUMES[props_hydrogens.atomic_nums].sum()
    if not np.isfinite(atom_volume):
        return np.asarray([np.nan], dtype=np.float32)

    n_bonds = props_hydrogens.num_bonds

    # reuse ring detection from the ring count descriptor: simple (non-fused)
    # aromatic and non-aromatic rings
    rings = rings_regular.simple_rings
    n_aromatic_rings = sum(1 for ring in rings if ring.is_aromatic)
    n_aliphatic_rings = len(rings) - n_aromatic_rings

    vabc = (
        atom_volume - 5.92 * n_bonds - 14.7 * n_aromatic_rings - 3.8 * n_aliphatic_rings
    )

    return np.asarray([vabc], dtype=np.float32)
