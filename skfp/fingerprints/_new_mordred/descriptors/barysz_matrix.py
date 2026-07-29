import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import floyd_warshall

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    CARBON_PROPERTY_VALUES,
    ELEMENT_PROPERTY_TABLES,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

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

    values: list = []
    for prop_name in ELEMENT_PROPERTY_TABLES:
        matrix = _barysz_matrix(atomic_props_regular, prop_name)
        if matrix is None:
            values.extend([np.nan] * len(_ATTR_NAMES))
        else:
            values.extend(
                _barysz_matrix_attribute_values(atomic_props_regular, n_frags, matrix)
            )

    return np.asarray(values, dtype=np.float32)


@np.errstate(divide="ignore", invalid="ignore")
def _barysz_matrix(props: AtomicProperties, prop_name: str) -> np.ndarray | None:
    carbon_value = CARBON_PROPERTY_VALUES[prop_name]

    props_vals = props.get(prop_name).astype(np.float32)
    if not np.isfinite(props_vals).all():
        return None

    n_atoms = props.num_atoms
    i_arr = props.bond_begin_idxs
    j_arr = props.bond_end_idxs
    weights = carbon_value**2 / (
        props_vals[i_arr] * props_vals[j_arr] * props.bond_orders
    )
    if not np.isfinite(weights).all():
        return None

    # Floyd-Warshall is the fastest on sparse COO adjacency matrix
    graph = coo_matrix(
        (weights.astype(np.float32), (i_arr, j_arr)),
        shape=(n_atoms, n_atoms),
        dtype=np.float32,
    ).tocsr()
    matrix = floyd_warshall(graph, directed=False)

    diagonal = 1.0 - carbon_value / props_vals
    if not np.isfinite(diagonal).all():
        return None

    np.fill_diagonal(matrix, diagonal)
    return matrix


def _barysz_matrix_attribute_values(
    props: AtomicProperties, n_frags: int, matrix: np.ndarray
) -> list[float | np.floating]:
    attrs = MatrixAttributes(matrix, props, hermitian=True, n_frags=n_frags)
    return [
        attrs.graph_energy,
        attrs.leading_eigenvalue,
        attrs.spectral_diameter,
        attrs.sp_ad,
        attrs.sp_mad,
        attrs.log_ee,
        attrs.sm1,
        attrs.ve1,
        attrs.ve2,
        attrs.ve3,
        attrs.vr1,
        attrs.vr2,
        attrs.vr3,
    ]
