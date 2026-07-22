import networkx as nx
import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "SpAbs_Dt",
    "SpMax_Dt",
    "SpDiam_Dt",
    "SpAD_Dt",
    "SpMAD_Dt",
    "LogEE_Dt",
    "SM1_Dt",
    "VE1_Dt",
    "VE2_Dt",
    "VE3_Dt",
    "VR1_Dt",
    "VR2_Dt",
    "VR3_Dt",
    "DetourIndex",
]


def calc(mol_regular: Mol, n_frags: int) -> tuple[np.ndarray, list[str]]:
    """
    Detour matrix descriptor.

    Computes matrix-aggregating features (spectral and Randic-like) of the
    detour matrix, together with the detour index. The detour matrix is
    undefined for disconnected molecules, so NaN is returned for those.
    """
    # avoids unnecessary eigendecomposition for disconnected molecules
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32), FEATURE_NAMES

    detour_matrix = _get_detour_matrix(mol_regular)
    attrs = MatrixAttributes(
        detour_matrix,
        mol_regular,
        hermitian=True,  # as in Mordred's reference implementation
        n_frags=n_frags,
    )

    values = np.asarray(
        [
            attrs.graph_energy,  # SpAbs_Dt
            attrs.leading_eigenvalue,  # SpMax_Dt
            attrs.spectral_diameter,  # SpDiam_Dt
            attrs.sp_ad,  # SpAD_Dt
            attrs.sp_mad,  # SpMAD_Dt
            attrs.log_ee,  # LogEE_Dt
            attrs.sm1,  # SM1_Dt
            attrs.ve1,  # VE1_Dt
            attrs.ve2,  # VE2_Dt
            attrs.ve3,  # VE3_Dt
            attrs.vr1,  # VR1_Dt
            attrs.vr2,  # VR2_Dt
            attrs.vr3,  # VR3_Dt
            int(0.5 * detour_matrix.sum()),  # DetourIndex
        ],
        dtype=np.float32,
    )

    return values, FEATURE_NAMES


def _get_detour_matrix(mol: Mol) -> np.ndarray:
    """
    Build the detour (maximum topological distance) matrix of a molecule.

    Entry ``(i, j)`` is the length of the longest simple path between atoms
    ``i`` and ``j``. The graph is split into biconnected blocks, the longest
    simple paths are solved per block, and the blocks are stitched back
    together through their articulation nodes (see :func:`_merge`).
    """
    n = mol.GetNumAtoms()

    if n == 1:
        return np.array([[0]], dtype=np.float32)

    G = nx.from_edgelist(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    )
    Q = []

    for bcc in (G.subgraph(components) for components in nx.biconnected_components(G)):
        lsp = _longest_simple_paths(bcc)
        nodes = set(bcc.nodes())

        Q.append((nodes, lsp))

    merged = _merge(Q, n)

    return merged


def _longest_simple_paths(graph: nx.Graph) -> dict[tuple[int, int], int]:
    """
    Longest simple path length between every pair of nodes in a block.

    Returns a mapping ``(i, j) -> distance`` with ``i < j``. Solved by
    brute-force DFS, which is affordable only because the biconnected blocks
    of molecular graphs are small.
    """
    # it's faster not to use NetworkX in brute-force DFS
    G = {u: list(graph[u]) for u in graph}
    longest_distances: dict[tuple[int, int], int] = {}

    for s in G:
        for e, dist in _longest_paths_from(s, G).items():
            if s < e:
                longest_distances[(s, e)] = dist

    return longest_distances


def _longest_paths_from(
    s: int,
    G: dict[int, list[int]],
) -> dict[int, int]:
    """
    Longest simple path length from a single source ``s`` to every node.

    Returns a mapping ``node -> distance``, with 0 for the source itself.
    """
    result = dict.fromkeys(G, 0)
    visited = {s}

    _dfs(s, 0, G, visited, result)

    return result


def _dfs(
    u: int,
    dist: int,
    G: dict[int, list[int]],
    visited: set[int],
    result: dict[int, int],
) -> None:
    """
    Recursive DFS with backtracking for :func:`_longest_paths_from`.

    Explores every simple path leaving ``u`` and records in ``result`` the
    maximum distance reached at each visited node.
    """
    dist += 1

    for v in G[u]:
        if v in visited:
            continue

        result[v] = max(result[v], dist)

        visited.add(v)
        _dfs(v, dist, G, visited, result)
        visited.remove(v)


def _merge(Q: list[tuple[set[int], dict[tuple[int, int], int]]], n: int) -> np.ndarray:
    """
    Assemble the full detour matrix from the per-block longest-path maps.

    Blocks are merged one at a time following the block-cut tree: each new
    block touches the already-merged part at exactly one articulation node
    ``common``, so a cross-block detour distance is
    ``D[i, common] + D[common, j]``.
    """
    D = np.zeros((n, n), dtype=np.float32)
    merged: set[int] = set()

    while Q:
        # pick a block adjacent to the already-merged set, in a block-cut tree
        # such a block shares exactly one node with it
        idx = next(
            i
            for i, (bnodes, _) in enumerate(Q)
            if not merged or any(u in merged for u in bnodes)
        )

        nodes, lsp = Q.pop(idx)

        # within-block distances
        if lsp:
            ij = np.array(list(lsp), dtype=np.intp)  # (k, 2) node-pair keys
            vals = np.fromiter(lsp.values(), dtype=np.float32, count=len(lsp))
            D[ij[:, 0], ij[:, 1]] = vals
            D[ij[:, 1], ij[:, 0]] = vals

        # cross-block pairs, joined through the articulation node;
        # common stays in both index arrays: D[common, common] == 0 acts as
        # an identity
        if merged:
            common = (nodes & merged).pop()
            old = np.fromiter(merged, dtype=np.intp)
            new = np.fromiter(nodes, dtype=np.intp)
            block = np.add.outer(D[old, common], D[common, new])
            D[np.ix_(old, new)] = block
            D[np.ix_(new, old)] = block.T

        merged |= nodes

    return D
