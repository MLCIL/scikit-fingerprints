"""
Ring count descriptors implemented with RDKit SSSR rings.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol


@dataclass(frozen=True, slots=True)
class RingCountFeature:
    name: str
    size: int | None
    match_size_or_larger: bool
    use_fused_rings: bool
    required_aromatic: bool | None
    required_hetero: bool | None


@dataclass(slots=True)
class RingProperties:
    atoms: set[int]
    is_aromatic: bool
    has_hetero: bool

    @property
    def size(self) -> int:
        return len(self.atoms)


_GENERAL_RING_FEATURES = [
    RingCountFeature("nRing", None, False, False, None, None),
    RingCountFeature("nHRing", None, False, False, None, True),
    RingCountFeature("naRing", None, False, False, True, None),
    RingCountFeature("naHRing", None, False, False, True, True),
    RingCountFeature("nARing", None, False, False, False, None),
    RingCountFeature("nAHRing", None, False, False, False, True),
]
_GENERAL_RING_NAMES = {feature.name for feature in _GENERAL_RING_FEATURES}


def _feature_name(
    size: int | None,
    match_size_or_larger: bool,
    fused: bool,
    aromatic: bool | None,
    hetero: bool | None,
) -> str:
    """
    Build the descriptor name from the ring filters encoded in the feature.

    `G` means "size greater than or equal to", `F` means fused ring system,
    lowercase `a` means aromatic, uppercase `A` means non-aromatic, and `H`
    means at least one heteroatom.
    """
    attrs = []
    if match_size_or_larger:
        attrs.append("G")
    if size is not None:
        attrs.append(str(size))
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


def _ring_count_features() -> list[RingCountFeature]:
    """
    Generate descriptor definitions in the order exposed by the public feature list.

    Each definition stores the descriptor name and the ring filters used to count
    matching rings.
    """
    features = [*_GENERAL_RING_FEATURES]
    for fused in [False, True]:
        for aromatic in [None, True, False]:
            for hetero in [None, True]:
                # Start each fused/aromatic/hetero block with the all-size count,
                # unless it is one of the general features already listed above.
                name = _feature_name(None, False, fused, aromatic, hetero)
                if name not in _GENERAL_RING_NAMES:
                    features.append(
                        RingCountFeature(name, None, False, fused, aromatic, hetero)
                    )

                start = 4 if fused else 3
                for size in range(start, 13):
                    # Then add exact-size counts: simple rings start at 3 atoms,
                    # fused ring systems start at 4 atoms.
                    name = _feature_name(size, False, fused, aromatic, hetero)
                    features.append(
                        RingCountFeature(name, size, False, fused, aromatic, hetero)
                    )

                name = _feature_name(12, True, fused, aromatic, hetero)
                features.append(
                    RingCountFeature(name, 12, True, fused, aromatic, hetero)
                )

    return features


RING_COUNT_FEATURES = _ring_count_features()
FEATURE_NAMES = [feature.name for feature in RING_COUNT_FEATURES]


def calc(mol_regular: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Count simple and fused rings across size, aromaticity, and heteroatom filters.
    """
    simple_ring_atom_sets = _ring_atom_sets(mol_regular)
    fused_ring_atom_sets = _fused_ring_atom_sets(simple_ring_atom_sets)
    simple_rings = _ring_properties(mol_regular, simple_ring_atom_sets)
    fused_rings = _ring_properties(mol_regular, fused_ring_atom_sets)
    ring_sets = {False: simple_rings, True: fused_rings}

    values = [
        sum(
            1
            for ring in ring_sets[feature.use_fused_rings]
            if _matches_size(ring, feature)
            and _matches_aromaticity(ring, feature.required_aromatic)
            and _matches_hetero(ring, feature.required_hetero)
        )
        for feature in RING_COUNT_FEATURES
    ]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES


def _ring_atom_sets(mol: Mol) -> list[set[int]]:
    """
    Return RDKit SSSR rings as atom-index sets.
    """
    return [set(ring) for ring in Chem.GetSymmSSSR(mol)]


def _fused_ring_atom_sets(rings: list[set[int]]) -> list[set[int]]:
    """
    Return fused ring components.
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

    return list(components.values())


def _ring_properties(mol: Mol, ring_atom_sets: list[set[int]]) -> list[RingProperties]:
    """
    Cache ring size, aromaticity, and heteroatom presence once per ring.
    """
    return [
        RingProperties(
            ring,
            all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring),
            any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in ring),
        )
        for ring in ring_atom_sets
    ]


def _matches_size(ring: RingProperties, feature: RingCountFeature) -> bool:
    """
    Check whether a ring has any size, an exact size, or a minimum size.
    """
    if feature.size is None:
        return True
    if feature.match_size_or_larger:
        return ring.size >= feature.size
    return ring.size == feature.size


def _matches_aromaticity(ring: RingProperties, required: bool | None) -> bool:
    """
    Check aromaticity only when the feature requires aromatic or aliphatic rings.
    """
    if required is None:
        return True
    return ring.is_aromatic if required else not ring.is_aromatic


def _matches_hetero(ring: RingProperties, required: bool | None) -> bool:
    """
    Check heteroatom content only when the feature requires hetero or carbocyclic rings.
    """
    if required is None:
        return True
    return ring.has_hetero if required else not ring.has_hetero
