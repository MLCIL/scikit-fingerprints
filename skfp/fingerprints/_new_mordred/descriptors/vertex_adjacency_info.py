import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
Vertex adjacency information descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["VAdjMat"]


def calc(props: AtomicProperties) -> np.ndarray:
    r"""
    Compute the Mordred vertex adjacency information descriptor.

    `VAdjMat` is defined as :math:`1 + \log_2(m)`, where :math:`m` is the number
    of heavy-heavy bonds. Returns NaN when :math:`m = 0`.
    """
    is_heavy = ~props.is_hydrogen
    m = int(
        np.count_nonzero(
            is_heavy[props.bond_begin_idxs] & is_heavy[props.bond_end_idxs]
        )
    )

    vadj_mat = np.nan if m == 0 else 1 + np.log2(m)

    return np.asarray([vadj_mat], dtype=np.float32)
