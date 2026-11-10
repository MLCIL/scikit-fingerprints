import numpy as np

from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Wiener index", "Wiener polarity number"]


def calc(distance_matrix_regular: DistanceMatrix) -> np.ndarray:
    r"""
    Wiener index and Wiener polarity number [1]_ [2]_ [3]_.

    Given the topological distance :math:`d_{ij}` between atoms :math:`i` and
    :math:`j` of the hydrogen-depleted molecular graph:

    - ``Wiener index`` :math:`W = \frac{1}{2} \sum_i \sum_j d_{ij}`, the sum of
      the distances over all unordered atom pairs
    - ``Wiener polarity number`` :math:`W_p`, the number of unordered atom pairs
      with :math:`d_{ij} = 3`

    References
    ----------
    .. [1] `Wiener, H.
        "Structural Determination of Paraffin Boiling Points"
        Journal of the American Chemical Society 69.1 (1947): 17-20.
        <https://doi.org/10.1021/ja01193a005>`_

    .. [2] `Rouvray, D. H.
        "The Rich Legacy of Half a Century of the Wiener Index"
        Topology in Chemistry (2002): 16-37.
        <https://www.sciencedirect.com/science/article/abs/pii/B9781898563761500068>`_

    .. [3] `Liu, M., & Liu, B.
        "On the Wiener Polarity Index"
        MATCH Commun. Math. Comput. Chem. 66.1 (2011): 293-304.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match66/n1/match66n1_293-304.pdf>`_
    """
    distances = distance_matrix_regular.matrix
    wiener = int(distances.sum() // 2)
    polarity = int(np.count_nonzero(distances == 3) // 2)
    return np.array([wiener, polarity], dtype=np.float32)
