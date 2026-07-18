import numpy as np
from rdkit.Chem import Mol
from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_sigma_electrons,
    get_valence_electrons,
)
from rdkit import Chem
from collections import defaultdict
from typing import Sequence

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

def calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Chi descriptors without adding explicit hydrogens.
    """
    properties = {
        "d": np.asarray(
            [get_sigma_electrons(atom) for atom in mol.GetAtoms()],
            dtype=float,
        ),
        "dv": np.asarray(
            [get_valence_electrons(atom) for atom in mol.GetAtoms()],
            dtype=float,
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

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES



def _chi_subgraphs(mol: Mol, order: int) -> dict[str, list[list[int]]]:
    classified: dict[str, list[list[int]]] = {chi_type: [] for chi_type in _CHI_TYPES}
    bond_endpoints = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    ]

    for bond_idxs in Chem.FindAllSubgraphsOfLengthN(mol, order):
        neighbors = _subgraph_neighbors(bond_idxs, bond_endpoints)
        chi_type = _classify_neighbors(neighbors)
        classified[chi_type].append(list(neighbors.keys()))

    return classified


def _subgraph_neighbors(
    bond_idxs: Sequence[int], bond_endpoints: Sequence[tuple[int, int]]
) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for bond_idx in bond_idxs:
        begin_idx, end_idx = bond_endpoints[bond_idx]
        neighbors[begin_idx].add(end_idx)
        neighbors[end_idx].add(begin_idx)
    return neighbors


def _classify_neighbors(neighbors: dict[int, set[int]]) -> str:
    visited: set[int] = set()
    visited_edges: set[tuple[int, int]] = set()
    degrees: set[int] = set()
    is_chain = _depth_first_search(
        next(iter(neighbors.keys())),
        neighbors,
        visited,
        visited_edges,
        degrees,
    )

    if is_chain:
        return "chain"
    if not degrees - {1, 2}:
        return "path"
    if 2 in degrees:
        return "path_cluster"
    return "cluster"


def _depth_first_search(
    node: int,
    neighbors: dict[int, set[int]],
    visited: set[int],
    visited_edges: set[tuple[int, int]],
    degrees: set[int],
) -> bool:
    visited.add(node)
    degrees.add(len(neighbors[node]))
    saw_cycle = False

    for neighbor in neighbors[node]:
        edge = (neighbor, node) if node > neighbor else (node, neighbor)

        if neighbor not in visited:
            visited_edges.add(edge)
            if _depth_first_search(
                neighbor, neighbors, visited, visited_edges, degrees
            ):
                saw_cycle = True
        elif edge not in visited_edges:
            visited_edges.add(edge)
            saw_cycle = True

    return saw_cycle


def _parse_chi_feature_name(name: str) -> tuple[str, int, str, bool]:
    prefix, order_and_prop = name.split("-", maxsplit=1)
    averaged = prefix.startswith("A")
    chi_type = _CHI_PREFIX_TO_TYPE[prefix]
    order = int(order_and_prop[0])
    prop = order_and_prop[1:]
    return chi_type, order, prop, averaged


def _chi_value(
    node_sets: Sequence[Sequence[int] | set[int]],
    prop_values: np.ndarray,
    averaged: bool,
) -> float:
    if averaged and len(node_sets) == 0:
        return np.nan

    value = 0.0
    for nodes in node_sets:
        product = 1.0
        for node in nodes:
            product *= prop_values[node]

        if product <= 0:
            return np.nan

        value += product**-0.5

    if averaged:
        value /= len(node_sets)

    return value

