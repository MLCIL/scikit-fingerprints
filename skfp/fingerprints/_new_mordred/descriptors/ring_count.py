"""
Ring count descriptors implemented with RDKit SSSR rings.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

_GENERAL_RING_SPECS = [
    ("nRing", None, False, False, None, None),
    ("nHRing", None, False, False, None, True),
    ("naRing", None, False, False, True, None),
    ("naHRing", None, False, False, True, True),
    ("nARing", None, False, False, False, None),
    ("nAHRing", None, False, False, False, True),
]
_GENERAL_RING_NAMES = {spec[0] for spec in _GENERAL_RING_SPECS}


def _feature_name(
    order: int | None,
    greater: bool,
    fused: bool,
    aromatic: bool | None,
    hetero: bool | None,
) -> str:
    """
    Build the Mordred feature name for a ring count parameter combination.
    """
    attrs = []
    if greater:
        attrs.append("G")
    if order is not None:
        attrs.append(str(order))
    if fused:
        attrs.append("F")
    if aromatic is True:
        attrs.append("a")
    elif aromatic is False:
        attrs.append("A")
    if hetero is True:
        attrs.append("H")
    elif hetero is False:
        attrs.append("C")
    return f"n{''.join(attrs)}Ring"


def _feature_specs() -> list[
    tuple[str, int | None, bool, bool, bool | None, bool | None]
]:
    """
    Generate Mordred RingCount feature specs in Mordred preset order.
    """
    specs: list[tuple[str, int | None, bool, bool, bool | None, bool | None]] = [
        *_GENERAL_RING_SPECS
    ]
    for fused in [False, True]:
        for aromatic in [None, True, False]:
            for hetero in [None, True]:
                name = _feature_name(None, False, fused, aromatic, hetero)
                if name not in _GENERAL_RING_NAMES:
                    specs.append((name, None, False, fused, aromatic, hetero))

                start = 4 if fused else 3
                for order in range(start, 13):
                    name = _feature_name(order, False, fused, aromatic, hetero)
                    specs.append((name, order, False, fused, aromatic, hetero))

                name = _feature_name(12, True, fused, aromatic, hetero)
                specs.append((name, 12, True, fused, aromatic, hetero))

    return specs


FEATURE_SPECS = _feature_specs()
FEATURE_NAMES = [spec[0] for spec in FEATURE_SPECS]


def _rings(mol: Mol) -> list[frozenset[int]]:
    """
    Return RDKit SSSR rings as atom-index sets, matching Mordred `Rings`.
    """
    return [frozenset(ring) for ring in Chem.GetSymmSSSR(mol)]


def _fused_rings(rings: list[frozenset[int]]) -> list[frozenset[int]]:
    """
    Return fused ring components, matching Mordred `FusedRings`.
    """
    if len(rings) < 2:
        return []

    parent = list(range(len(rings)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    fused_ring_ids = set()
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2:
                fused_ring_ids.update([i, j])
                union(i, j)

    components: dict[int, set[int]] = {}
    for idx in fused_ring_ids:
        ring = rings[idx]
        root = find(idx)
        components.setdefault(root, set()).update(ring)

    return [frozenset(component) for component in components.values()]


def _matches_order(ring: frozenset[int], order: int | None, greater: bool) -> bool:
    """
    Check the Mordred ring-size predicate.
    """
    if order is None:
        return True
    return len(ring) >= order if greater else len(ring) == order


def _matches_aromaticity(mol: Mol, ring: frozenset[int], aromatic: bool | None) -> bool:
    """
    Check the Mordred aromatic/aliphatic ring predicate.
    """
    if aromatic is None:
        return True
    is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
    return is_aromatic if aromatic else not is_aromatic


def _matches_hetero(mol: Mol, ring: frozenset[int], hetero: bool | None) -> bool:
    """
    Check the Mordred hetero ring predicate.
    """
    if hetero is None:
        return True
    has_hetero = any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in ring)
    return has_hetero if hetero else not has_hetero


def calc(mol_regular: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred RingCount descriptors from RDKit SSSR rings.
    """
    simple_rings = _rings(mol_regular)
    fused_rings = _fused_rings(simple_rings)
    ring_sets = {False: simple_rings, True: fused_rings}

    values = []
    for _, order, greater, fused, aromatic, hetero in FEATURE_SPECS:
        values.append(
            sum(
                1
                for ring in ring_sets[fused]
                if _matches_order(ring, order, greater)
                and _matches_aromaticity(mol_regular, ring, aromatic)
                and _matches_hetero(mol_regular, ring, hetero)
            )
        )

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
