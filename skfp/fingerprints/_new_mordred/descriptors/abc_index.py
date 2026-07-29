import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["ABC", "ABCGG"]


def calc(
    atomic_props_regular: AtomicProperties, distance_matrix_regular: DistanceMatrix
) -> np.ndarray:
    """
    ABC index descriptor, combining the classical ABC index and its
    Graovac-Ghorbani analog.

    Based on Furtula, B. (2016). Atom-bond connectivity index versus
    Graovac-Ghorbani analog. MATCH Communications in Mathematical and in
    Computer Chemistry, 75(1), 233-242.
    http://match.pmf.kg.ac.rs/electronic_versions/Match75/n1/match75n1_233-242.pdf
    """
    values = np.array(
        [
            _calc_abc_index(atomic_props_regular),
            _calc_abcgg_index(atomic_props_regular, distance_matrix_regular),
        ],
        dtype=np.float32,
    )
    return values


def _calc_abc_index(props: AtomicProperties) -> float:
    """
    Atom-bond connectivity (ABC) index descriptor.

    Based on Das, K. C., Gutman, I., & Furtula, B. (2012). On atom-bond
    connectivity index. Filomat, 26(4), 733-738.
    https://doi.org/10.2298/FIL1204733D
    """
    degrees = props.degrees
    du = degrees[props.bond_begin_idxs]
    dv = degrees[props.bond_end_idxs]
    return float(np.sqrt((du + dv - 2.0) / (du * dv)).sum())


def _calc_abcgg_index(
    props: AtomicProperties, distance_matrix_regular: DistanceMatrix
) -> float:
    """
    Graovac-Ghorbani atom-bond connectivity index descriptor.

    Based on Furtula, B. (2016). Atom-bond connectivity index versus
    Graovac-Ghorbani analog. MATCH Communications in Mathematical and in
    Computer Chemistry, 75(1), 233-242.
    http://match.pmf.kg.ac.rs/electronic_versions/Match75/n1/match75n1_233-242.pdf
    """
    D = distance_matrix_regular.matrix

    # rows of the distance matrix for the two ends of every bond, at once
    dist_u = D[props.bond_begin_idxs]
    dist_v = D[props.bond_end_idxs]

    # nu/nv: atoms closer to one end of the bond than to the other
    nu = (dist_u < dist_v).sum(axis=1)
    nv = (dist_v < dist_u).sum(axis=1)

    return float(np.sqrt((nu + nv - 2.0) / (nu * nv)).sum())
