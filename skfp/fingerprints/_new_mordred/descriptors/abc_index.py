r"""ABC Index descriptor.

References
----------
    * http://match.pmf.kg.ac.rs/electronic_versions/Match75/n1/match75n1_233-242.pdf

"""

import numpy as np
from rdkit.Chem import Bond, Mol

from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix


def _calc_abc_index(mol: Mol) -> float:
    r"""atom-bond connectivity index descriptor.

    References
    ----------
        * :doi:`10.2298/FIL1204733D`

    """
    total = 0.0
    bond: Bond
    for bond in mol.GetBonds():
        du = bond.GetBeginAtom().GetDegree()
        dv = bond.GetEndAtom().GetDegree()
        total += np.sqrt((du + dv - 2.0) / (du * dv))
    return total


def _calc_abcgg_index(mol: Mol, distance_matrix_regular: DistanceMatrix) -> float:
    r"""Graovac-Ghorbani atom-bond connectivity index descriptor.

    References
    ----------
        * Furtula, B. Atom-bond connectivity index versus Graovac-Ghorbani analog. MATCH Commun. Math. Comput. Chem 75, 233-242 (2016).

    """
    D = distance_matrix_regular.matrix

    total = 0.0
    bond: Bond
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        nu = np.sum(D[u, :] < D[v, :])
        nv = np.sum(D[v, :] < D[u, :])

        total += np.sqrt((nu + nv - 2.0) / (nu * nv))
    return total


def calc(mol_regular: Mol, distance_matrix_regular: DistanceMatrix) -> np.ndarray:
    return np.array(
        [
            _calc_abc_index(mol_regular),
            _calc_abcgg_index(mol_regular, distance_matrix_regular),
        ],
        dtype=np.float32,
    )
