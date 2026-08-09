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

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties


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


class RingSets:
    """
    SSSR rings of a molecule and their per-ring properties.

    Shared by the ring count and van der Waals volume descriptors, which would
    otherwise each re-run ``GetSymmSSSR`` and re-inspect every ring atom.
    """

    def __init__(self, mol: Mol, props: AtomicProperties):
        self.mol = mol
        self._props = props

        self.simple_ring_atom_sets = [set(ring) for ring in Chem.GetSymmSSSR(mol)]
        self.num_rings = len(self.simple_ring_atom_sets)
        self.simple_rings = self._ring_properties(self.simple_ring_atom_sets)
        self.fused_rings = self._ring_properties(
            _fused_ring_atom_sets(self.simple_ring_atom_sets)
        )

    def _ring_properties(self, ring_atom_sets: list[set[int]]) -> list[RingProperties]:
        """
        Cache ring size, aromaticity, and heteroatom presence once per ring.
        """
        is_aromatic = self._props.is_aromatic
        is_hetero = self._props.atomic_nums != 6
        return [
            RingProperties(
                ring,
                bool(is_aromatic[list(ring)].all()),
                bool(is_hetero[list(ring)].any()),
            )
            for ring in ring_atom_sets
        ]


_GENERAL_RING_FEATURES = [
    # name, size, match size or larger, use fused rings, required aromatic, required hetero
    RingCountFeature("nRing", None, False, False, None, None),
    RingCountFeature("nHRing", None, False, False, None, True),
    RingCountFeature("naRing", None, False, False, True, None),
    RingCountFeature("naHRing", None, False, False, True, True),
    RingCountFeature("nARing", None, False, False, False, None),
    RingCountFeature("nAHRing", None, False, False, False, True),
]
_GENERAL_RING_NAMES = {feature.name for feature in _GENERAL_RING_FEATURES}

_RING_FILTER_BLOCKS = [
    # use fused rings, required aromatic, required hetero
    (False, None, None),
    (False, None, True),
    (False, True, None),
    (False, True, True),
    (False, False, None),
    (False, False, True),
    (True, None, None),
    (True, None, True),
    (True, True, None),
    (True, True, True),
    (True, False, None),
    (True, False, True),
]


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
    for fused, aromatic, hetero in _RING_FILTER_BLOCKS:
        min_ring_size = 4 if fused else 3

        # Each block contributes an all-size descriptor, exact-size descriptors,
        # and finally a G12 descriptor for rings with size 12 or greater.
        name = _feature_name(None, False, fused, aromatic, hetero)
        if name not in _GENERAL_RING_NAMES:
            features.append(
                RingCountFeature(name, None, False, fused, aromatic, hetero)
            )

        # Simple rings start at 3 atoms, fused ring systems start at 4 atoms.
        for size in range(min_ring_size, 13):
            name = _feature_name(size, False, fused, aromatic, hetero)
            features.append(
                RingCountFeature(name, size, False, fused, aromatic, hetero)
            )

        name = _feature_name(12, True, fused, aromatic, hetero)
        features.append(RingCountFeature(name, 12, True, fused, aromatic, hetero))

    return features


RING_COUNT_FEATURES = _ring_count_features()
FEATURE_NAMES = [feature.name for feature in RING_COUNT_FEATURES]

# rings buckets: size (max_size+2), aromaticity yes/no (2), heteroatom presence (2)
# last bucket gathers everything above the largest size
_MAX_RING_SIZE = 12
_HISTOGRAM_SHAPE = (_MAX_RING_SIZE + 2, 2, 2)


def _histogram(rings: list[RingProperties]) -> np.ndarray:
    """
    Count rings per (size, is aromatic, has heteroatom) bucket.
    """
    histogram = np.zeros(_HISTOGRAM_SHAPE)
    for ring in rings:
        # int(), because NumPy reads a bool index as a mask rather than as 0 or 1
        histogram[
            min(ring.size, _MAX_RING_SIZE + 1),
            int(ring.is_aromatic),
            int(ring.has_hetero),
        ] += 1
    return histogram


def _selector(feature: RingCountFeature) -> np.ndarray:
    """
    Create a binary mask to select histogram buckets used
    by a particular descriptor.
    """
    sizes = np.zeros(_HISTOGRAM_SHAPE[0], dtype=bool)
    if feature.size is None:
        sizes[:] = True
    elif feature.match_size_or_larger:
        sizes[feature.size :] = True
    else:
        sizes[feature.size] = True

    # a required flag selects one of the two buckets, no requirement selects both
    def flag_selector(required: bool | None) -> np.ndarray:
        if required is None:
            return np.ones(2, dtype=bool)
        return np.arange(2) == required

    return (
        sizes[:, None, None]
        & flag_selector(feature.required_aromatic)[None, :, None]
        & flag_selector(feature.required_hetero)[None, None, :]
    )


def _selector_matrix(use_fused_rings: bool) -> np.ndarray:
    """
    Bucket masks of every descriptor reading a given ring set, stacked into a
    matrix. Descriptors reading the other ring set contribute an all-zero row.
    """
    return np.array(
        [
            _selector(feature).ravel()
            if feature.use_fused_rings == use_fused_rings
            else np.zeros(np.prod(_HISTOGRAM_SHAPE), dtype=bool)
            for feature in RING_COUNT_FEATURES
        ],
        dtype=np.float64,
    )


_SIMPLE_RING_SELECTORS = _selector_matrix(use_fused_rings=False)
_FUSED_RING_SELECTORS = _selector_matrix(use_fused_rings=True)


def calc(rings: RingSets) -> np.ndarray:
    """
    Count simple and fused rings across size, aromaticity, and heteroatom filters.

    Every descriptor is the number of rings falling into some set of
    (size, aromaticity, heteroatom) buckets. Thus, we can read them all from
    a single histogram per ring set.
    """
    values = _SIMPLE_RING_SELECTORS @ _histogram(rings.simple_rings).ravel()
    values += _FUSED_RING_SELECTORS @ _histogram(rings.fused_rings).ravel()

    return values.astype(np.float32)


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
