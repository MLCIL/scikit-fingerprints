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


def calc(props: AtomicProperties, subgraphs: Subgraphs) -> np.ndarray:
    """
    Chi descriptors.

    Each descriptor sums ``prod(property over the subgraph atoms) ** -0.5`` over
    the subgraphs of one order and type. Both properties are summed in the same
    pass over each set of subgraphs, since the two differ only in what they weigh
    the same atoms by.
    """
    prop_vals = np.stack(
        [props.sigma_electrons.astype(np.float64), props.valence_electrons]
    )
    prop_idxs = {"d": 0, "dv": 1}

    sums: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    values = []
    for subgraph_type, order, prop, averaged in _PARSED_FEATURE_NAMES:
        key = (subgraph_type, order)
        if key not in sums:
            node_set = subgraphs.node_sets(order, subgraph_type)
            sums[key] = _chi_sums(node_set, prop_vals)

        totals, num_subgraphs = sums[key]
        total = totals[prop_idxs[prop]]
        if averaged:
            total = total / num_subgraphs if num_subgraphs else np.nan
        values.append(total)

    return np.asarray(values, dtype=np.float32)


def _chi_sums(
    node_sets: list[np.ndarray], prop_vals: np.ndarray
) -> tuple[np.ndarray, int]:
    """
    Sum of ``prod(prop_vals over the subgraph atoms) ** -0.5`` over subgraphs, for
    every property at once, together with the number of subgraphs summed over.

    A property whose subgraph product is non-positive anywhere sums to NaN.
    """
    totals = np.zeros(len(prop_vals))
    num_subgraphs = 0

    for nodes in node_sets:
        # (n_props, n_subgraphs), the property product over each subgraph's atoms
        products = prop_vals[:, nodes].prod(axis=2)
        totals += np.where(
            (products <= 0).any(axis=1), np.nan, (products**-0.5).sum(axis=1)
        )
        num_subgraphs += len(nodes)

    return totals, num_subgraphs
