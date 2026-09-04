import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.subgraphs import (
    Subgraphs,
    SubgraphsTopology,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# subgraph classes
CHAIN = "chain"  # closes a cycle, branched or not
PATH = "path"  # acyclic, and no atom joins more than two of its bonds
PATH_CLUSTER = "path_cluster"  # acyclic and branched, some atom joining exactly two
CLUSTER = "cluster"  # acyclic and branched, no atom joining exactly two
SUBGRAPH_TYPES = (CHAIN, PATH, PATH_CLUSTER, CLUSTER)

# The chi families: each covers one subgraph class over a range of orders, and is
# named by the prefix its features carry. An "A" prefix averages over the subgraphs
# instead of only summing, so Xp and AXp read the same sums.
_FAMILIES = (
    (CHAIN, range(3, 8), ("Xch",)),
    (CLUSTER, range(3, 7), ("Xc",)),
    (PATH_CLUSTER, range(4, 7), ("Xpc",)),
    (PATH, range(8), ("Xp", "AXp")),
)

# the atom weightings the chi indices use, in the order calc() stacks them:
# "d" weights an atom by its sigma electrons, "dv" by its valence electrons
_PROPS = ("d", "dv")


FEATURE_NAMES = [
    f"{prefix}-{order}{prop}"
    for _, orders, prefixes in _FAMILIES
    for order in orders
    for prefix in prefixes
    for prop in _PROPS
]


def calc(props: AtomicProperties, subgraphs: Subgraphs) -> np.ndarray:
    """
    Chi descriptors, shape ``(n_features,)``.

    Each descriptor sums ``prod(property over the subgraph atoms) ** -0.5`` over the
    subgraphs of one order and class. Both properties are summed in the same pass
    over each set of subgraphs, since the two differ only in what they weigh the
    same atoms by, and the averaged families reuse the sums of their plain ones.
    """
    # one row per weighting, in _PROPS order, so that a row of the sums below lines
    # up with the feature name that reads it
    prop_vals = np.stack([props.sigma_electrons.astype(float), props.valence_electrons])

    values = []
    # properties with negative values end up as NaNs
    with np.errstate(divide="ignore", invalid="ignore"):
        for subgraph_type, orders, prefixes in _FAMILIES:
            for order in orders:
                products = _subgraph_prop_products(
                    subgraphs, order, subgraph_type, prop_vals
                )
                totals, num_subgraphs = _chi_sums(products)

                for prefix in prefixes:
                    averaged = prefix.startswith("A")
                    for total in totals:
                        if averaged:
                            total = total / num_subgraphs if num_subgraphs else np.nan
                        values.append(total)

    return np.asarray(values, dtype=np.float32)


def _subgraph_prop_products(
    subgraphs: Subgraphs,
    order: int,
    subgraph_type: str,
    prop_vals: np.ndarray,
) -> np.ndarray:
    """
    Product of every property over the atoms of each subgraph of one order and chi
    class, shape ``(n_props, n_subgraphs)``.

    Order 0 and the paths are read off directly. Other classes need this
    order's subgraphs classified.
    """
    if order == 0:
        # order 0 subgraphs are the individual atoms, which belong to every class,
        # so each product is over a single atom
        return prop_vals

    if subgraph_type == PATH:
        # the paths are already held as atoms, for the path count descriptors
        return prop_vals[:, subgraphs.paths(order).atom_idxs].prod(axis=2)

    topology = subgraphs.topology(order)
    class_mask = _class_mask(topology, subgraph_type)
    return topology.atom_products(prop_vals)[:, class_mask]


def _class_mask(topology: SubgraphsTopology, subgraph_type: str) -> np.ndarray:
    """
    Which subgraphs of a given order belong to a given subgraph type (Chi class).
    Which of one order's subgraphs belong to a chi class, shape ``(n_subgraphs,)``.

    The four classes partition the subgraphs, so exactly one of these masks holds
    for any given subgraph.

    Returns a mask over subgraphs, array of shape (n_subgraphs,).
    """
    if subgraph_type == CHAIN:
        return topology.is_cyclic
    if subgraph_type == PATH:
        return topology.is_path

    is_branched = ~topology.is_cyclic & (topology.max_degree > 2)
    if subgraph_type == PATH_CLUSTER:
        return is_branched & topology.has_degree_2
    if subgraph_type == CLUSTER:
        return is_branched & ~topology.has_degree_2
    raise ValueError(f"Unknown chi subgraph class {subgraph_type!r}")


def _chi_sums(products: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Sum of ``product over the subgraph atoms ** -0.5`` over subgraphs, for every
    property at once, given the products of shape ``(n_props, n_subgraphs)``.

    A property whose subgraph product is non-positive anywhere sums to NaN.
    Assumes this function is wrapped in np.errstate().
    """
    totals = (1 / np.sqrt(products)).sum(axis=1)
    totals = np.where(np.isfinite(totals), totals, np.nan)
    return totals, products.shape[1]
