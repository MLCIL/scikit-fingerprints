import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import floyd_warshall

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    CARBON_PROPERTY_VALUES,
    ELEMENT_PROPERTY_TABLES,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.matrix_attributes import spectral_attributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_ATTR_NAMES = [
    "SpAbs",
    "SpMax",
    "SpDiam",
    "SpAD",
    "SpMAD",
    "LogEE",
    "SM1",
    "VE1",
    "VE2",
    "VE3",
    "VR1",
    "VR2",
    "VR3",
]

FEATURE_NAMES = [
    f"{attr}_Dz{prop}" for prop in ELEMENT_PROPERTY_TABLES for attr in _ATTR_NAMES
]


def calc(atomic_props_regular: AtomicProperties, n_frags: int) -> np.ndarray:
    """
    Barysz matrix spectral descriptors.

    Constructs a weighted distance matrix where bond weights are inversely
    proportional to atomic properties and bond order, normalized by the
    corresponding carbon-carbon value. Spectral attributes of the resulting
    matrix are computed for each atomic property.

    Requires a connected molecule (single fragment).
    """
    if n_frags != 1:
        values_nan = np.full(
            len(ELEMENT_PROPERTY_TABLES) * len(_ATTR_NAMES), np.nan, dtype=np.float32
        )

        return values_nan

    # bond graph is the same for all properties, only the shortest paths weighted
    # by properties and the spectral attributes differ
    bond_graph = _BondGraph(atomic_props_regular)
    weights, diagonals = _bond_weights_and_diagonals(atomic_props_regular)
    is_defined = np.isfinite(weights).all(axis=1) & np.isfinite(diagonals).all(axis=1)

    matrices = []
    for weight_row, diagonal in zip(
        weights[is_defined], diagonals[is_defined], strict=True
    ):
        matrix = floyd_warshall(
            bond_graph.get_prop_weighted_matrix(weight_row.astype(np.float32)),
            directed=False,
        )
        np.fill_diagonal(matrix, diagonal)
        matrices.append(matrix)

    # a property whose matrix does not exist keeps its attributes at NaN
    values = np.full((len(ELEMENT_PROPERTY_TABLES), len(_ATTR_NAMES)), np.nan)
    if matrices:
        values[is_defined] = spectral_attributes(
            np.stack(matrices), atomic_props_regular, hermitian=True, n_frags=n_frags
        )

    return values.ravel().astype(np.float32)


@np.errstate(divide="ignore", invalid="ignore")
def _bond_weights_and_diagonals(
    props: AtomicProperties,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bond weights and matrix diagonals of every atomic property, of shapes
    ``(n_props, n_bonds)`` and ``(n_props, n_atoms)``.

    A bond weighs the inverse of the properties of the atoms it joins and of its own
    order, normalized by the value a carbon-carbon bond of that kind would have.
    """
    carbon_values = np.array(list(CARBON_PROPERTY_VALUES.values()))
    prop_vals = np.stack([props.get(name) for name in ELEMENT_PROPERTY_TABLES]).astype(
        np.float32
    )

    begins, ends = props.bond_begin_idxs, props.bond_end_idxs
    weights = (carbon_values**2)[:, np.newaxis] / (
        prop_vals[:, begins] * prop_vals[:, ends] * props.bond_orders
    )
    diagonals = 1.0 - carbon_values.astype(np.float32)[:, np.newaxis] / prop_vals
    return weights, diagonals


class _BondGraph:
    """
    COO sparse matrix graph representation. Easy to add weighting by properties
    later.
    """

    def __init__(self, props: AtomicProperties):
        self.shape = (props.num_atoms, props.num_atoms)
        # compressing sorts the bonds by atom, which reorders their weights as well
        pattern = coo_matrix(
            (np.arange(props.num_bonds), (props.bond_begin_idxs, props.bond_end_idxs)),
            shape=self.shape,
            dtype=np.intp,
        ).tocsr()
        self._bond_order = pattern.data
        self._indices = pattern.indices
        self._indptr = pattern.indptr

    def get_prop_weighted_matrix(self, weights: np.ndarray) -> csr_matrix:
        """
        Return the same bonds, carrying the given weights.
        """
        return csr_matrix(
            (weights[self._bond_order], self._indices, self._indptr), shape=self.shape
        )
