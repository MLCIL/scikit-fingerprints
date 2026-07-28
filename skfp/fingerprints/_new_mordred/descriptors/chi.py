import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "Xch-3d",
    "Xch-4d",
    "Xch-5d",
    "Xch-6d",
    "Xch-7d",
    "Xch-3dv",
    "Xch-4dv",
    "Xch-5dv",
    "Xch-6dv",
    "Xch-7dv",
    "Xc-3d",
    "Xc-4d",
    "Xc-5d",
    "Xc-6d",
    "Xc-3dv",
    "Xc-4dv",
    "Xc-5dv",
    "Xc-6dv",
    "Xpc-4d",
    "Xpc-5d",
    "Xpc-6d",
    "Xpc-4dv",
    "Xpc-5dv",
    "Xpc-6dv",
    "Xp-0d",
    "Xp-1d",
    "Xp-2d",
    "Xp-3d",
    "Xp-4d",
    "Xp-5d",
    "Xp-6d",
    "Xp-7d",
    "AXp-0d",
    "AXp-1d",
    "AXp-2d",
    "AXp-3d",
    "AXp-4d",
    "AXp-5d",
    "AXp-6d",
    "AXp-7d",
    "Xp-0dv",
    "Xp-1dv",
    "Xp-2dv",
    "Xp-3dv",
    "Xp-4dv",
    "Xp-5dv",
    "Xp-6dv",
    "Xp-7dv",
    "AXp-0dv",
    "AXp-1dv",
    "AXp-2dv",
    "AXp-3dv",
    "AXp-4dv",
    "AXp-5dv",
    "AXp-6dv",
    "AXp-7dv",
]
_CHI_PREFIX_TO_TYPE = {
    "Xch": "chain",
    "Xp": "path",
    "AXp": "path",
    "Xpc": "path_cluster",
    "Xc": "cluster",
}


def _parse_chi_feature_name(name: str) -> tuple[str, int, str, bool]:
    prefix, order_and_prop = name.split("-", maxsplit=1)
    averaged = prefix.startswith("A")
    subgraph_type = _CHI_PREFIX_TO_TYPE[prefix]
    order = int(order_and_prop[0])
    prop = order_and_prop[1:]
    return subgraph_type, order, prop, averaged


_PARSED_FEATURE_NAMES = [_parse_chi_feature_name(name) for name in FEATURE_NAMES]


def calc(props: AtomicProperties, subgraphs: Subgraphs) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Chi descriptors without adding explicit hydrogens.

    Each descriptor sums ``prod(property over the subgraph atoms) ** -0.5`` over
    the subgraphs of one order and type.
    """
    prop_vals = {
        "d": props.sigma_electrons.astype(np.float64),
        "dv": props.valence_electrons.astype(np.float64),
    }
    values = [
        _chi_value(subgraphs.node_sets(order, subgraph_type), prop_vals[prop], averaged)
        for subgraph_type, order, prop, averaged in _PARSED_FEATURE_NAMES
    ]
    return np.asarray(values, dtype=np.float32), FEATURE_NAMES


def _chi_value(
    node_sets: list[np.ndarray], prop_vals: np.ndarray, averaged: bool
) -> float:
    """
    Sum of ``prod(prop_vals over the subgraph atoms) ** -0.5`` over subgraphs.

    NaN when any subgraph has a non-positive product, or when an averaged
    descriptor has no subgraph to average over.
    """
    total = 0.0
    num_subgraphs = 0

    for nodes in node_sets:
        products = prop_vals[nodes].prod(axis=1)
        if np.any(products <= 0):
            return np.nan
        total += (products**-0.5).sum()
        num_subgraphs += len(nodes)

    if averaged:
        return np.nan if num_subgraphs == 0 else total / num_subgraphs

    return total
