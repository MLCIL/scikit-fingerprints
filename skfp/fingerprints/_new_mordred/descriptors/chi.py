from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_sigma_electrons,
    get_valence_electrons,
)

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
_CHI_TYPES = ("chain", "path", "path_cluster", "cluster")
_CHI_PREFIX_TO_TYPE = {
    "Xch": "chain",
    "Xp": "path",
    "AXp": "path",
    "Xpc": "path_cluster",
    "Xc": "cluster",
}


def calc(mol: Mol) -> np.ndarray:
    """
    Compute Mordred Chi descriptors without adding explicit hydrogens.
    """
    properties = {
        "d": np.asarray(
            [get_sigma_electrons(atom) for atom in mol.GetAtoms()],
            dtype=np.float32,
        ),
        "dv": np.asarray(
            [get_valence_electrons(atom) for atom in mol.GetAtoms()],
            dtype=np.float32,
        ),
    }
    subgraphs_by_order = {order: _chi_subgraphs(mol, order) for order in range(1, 8)}

    values = []
    for name in FEATURE_NAMES:
        chi_type, order, prop, averaged = _parse_chi_feature_name(name)
        if order == 0:
            node_sets = [[atom.GetIdx()] for atom in mol.GetAtoms()]
        else:
            node_sets = subgraphs_by_order[order][chi_type]
        values.append(_chi_value(node_sets, properties[prop], averaged))

    return np.asarray(values, dtype=np.float32)


def _chi_subgraphs(mol: Mol, order: int) -> dict[str, list[list[int]]]:
    classified: dict[str, list[list[int]]] = {chi_type: [] for chi_type in _CHI_TYPES}
    bond_endpoints = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    ]

    for bond_idxs in Chem.FindAllSubgraphsOfLengthN(mol, order):
        # count how many subgraph edges are incident to each atom (its degree
        # within this subgraph)
        deg: defaultdict[int, int] = defaultdict(int)
        for a, b in (bond_endpoints[i] for i in bond_idxs):
            deg[a] += 1
            deg[b] += 1

        # A subgraph contains a cycle iff n_edges >= n_nodes (for connected subgraphs).
        if len(bond_idxs) >= len(deg):
            chi_type = "chain"
        else:
            d = set(deg.values())
            if d <= {1, 2}:
                chi_type = "path"
            elif 2 in d:
                chi_type = "path_cluster"
            else:
                chi_type = "cluster"

        classified[chi_type].append(list(deg.keys()))

    return classified


def _parse_chi_feature_name(name: str) -> tuple[str, int, str, bool]:
    prefix, order_and_prop = name.split("-", maxsplit=1)
    averaged = prefix.startswith("A")
    chi_type = _CHI_PREFIX_TO_TYPE[prefix]
    order = int(order_and_prop[0])
    prop = order_and_prop[1:]
    return chi_type, order, prop, averaged


def _chi_value(
    node_sets: list[list[int]],
    prop_values: np.ndarray,
    averaged: bool,
) -> np.float32:
    if averaged and len(node_sets) == 0:
        return np.float32(np.nan)

    value = 0.0
    for nodes in node_sets:
        product = 1.0
        for node in nodes:
            product *= prop_values[node]

        if product <= 0:
            return np.float32(np.nan)

        value += product**-0.5

    if averaged:
        value /= len(node_sets)

    return np.float32(value)
