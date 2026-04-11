import numpy as np
from rdkit.Chem import Bond, Mol

from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _calc_abc_index(mol: Mol) -> float:
    """
    Atom-bond connectivity (ABC) index descriptor.

    Based on Das, K. C., Gutman, I., & Furtula, B. (2012). On atom-bond
    connectivity index. Filomat, 26(4), 733-738.
    https://doi.org/10.2298/FIL1204733D
    """
    total = 0.0
    bond: Bond
    for bond in mol.GetBonds():
        du = bond.GetBeginAtom().GetDegree()
        dv = bond.GetEndAtom().GetDegree()
        total += np.sqrt((du + dv - 2.0) / (du * dv))
    return total


def _calc_abcgg_index(mol: Mol, distance_matrix_regular: DistanceMatrix) -> float:
    """
    Graovac-Ghorbani atom-bond connectivity index descriptor.

    Based on Furtula, B. (2016). Atom-bond connectivity index versus
    Graovac-Ghorbani analog. MATCH Communications in Mathematical and in
    Computer Chemistry, 75(1), 233-242.
    http://match.pmf.kg.ac.rs/electronic_versions/Match75/n1/match75n1_233-242.pdf
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
    """
    ABC index descriptor, combining the classical ABC index and its
    Graovac-Ghorbani analog.

    Based on Furtula, B. (2016). Atom-bond connectivity index versus
    Graovac-Ghorbani analog. MATCH Communications in Mathematical and in
    Computer Chemistry, 75(1), 233-242.
    http://match.pmf.kg.ac.rs/electronic_versions/Match75/n1/match75n1_233-242.pdf
    """
    return np.array(
        [
            _calc_abc_index(mol_regular),
            _calc_abcgg_index(mol_regular, distance_matrix_regular),
        ],
        dtype=np.float32,
    )
