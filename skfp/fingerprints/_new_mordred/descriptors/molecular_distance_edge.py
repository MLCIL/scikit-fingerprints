import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "MDE primary carbons",
    "MDE primary and secondary carbons",
    "MDE primary and tertiary carbons",
    "MDE primary and quaternary carbons",
    "MDE secondary carbons",
    "MDE secondary and tertiary carbons",
    "MDE secondary and quaternary carbons",
    "MDE tertiary carbons",
    "MDE tertiary and quaternary carbons",
    "MDE quaternary carbons",
    "MDE primary nitrogens",
    "MDE primary and secondary nitrogens",
    "MDE primary and tertiary nitrogens",
    "MDE secondary nitrogens",
    "MDE secondary and tertiary nitrogens",
    "MDE tertiary nitrogens",
    "MDE primary oxygens",
    "MDE primary and secondary oxygens",
    "MDE secondary oxygens",
]

# elements (C, N, O) and heavy-atom degree pairs, in feature order
_ELEMENTS = [6, 7, 8]
_BUCKETS = [
    (atomic_num, valence_1, valence_2)
    for atomic_num in _ELEMENTS
    for valence_1 in range(1, 11 - atomic_num)
    for valence_2 in range(valence_1, 11 - atomic_num)
]

# array mapping (atomic_num, valence_1, valence_2) -> bucket index
# requires: valence_1 <= valence_2, max possible valence is 11, other pairs map to -1
_BUCKET_LOOKUP = np.full((max(_ELEMENTS) + 1, 11, 11), -1, int)
for _idx, (_z, _v1, _v2) in enumerate(_BUCKETS):
    _BUCKET_LOOKUP[_z, _v1, _v2] = _idx


@np.errstate(divide="ignore", invalid="ignore")
def calc(
    atomic_props_regular: AtomicProperties,
    distance_matrix_regular: DistanceMatrix,
) -> np.ndarray:
    num_atoms = atomic_props_regular.num_atoms
    dists = distance_matrix_regular.matrix

    atomic_nums = atomic_props_regular.atomic_nums
    valences = atomic_props_regular.degrees

    # enumerate atom pairs (unordered) and bucket by valence
    i, j = np.triu_indices(num_atoms, k=1)
    valence_low = np.minimum(valences[i], valences[j])
    valence_high = np.maximum(valences[i], valences[j])
    same_element = (atomic_nums[i] == atomic_nums[j]) & np.isin(
        atomic_nums[i], _ELEMENTS
    )
    in_range = (valence_low >= 0) & (valence_high < 11)
    mask = same_element & in_range

    bucket = _BUCKET_LOOKUP[atomic_nums[i[mask]], valence_low[mask], valence_high[mask]]
    log_dists = np.log(dists[i[mask], j[mask]])
    valid = bucket >= 0
    bucket, log_dists = bucket[valid], log_dists[valid]

    counts = np.bincount(bucket, minlength=len(_BUCKETS)).astype(float)
    sum_log = np.bincount(bucket, weights=log_dists, minlength=len(_BUCKETS))

    values = counts * np.exp(-sum_log / counts)

    return values.astype(np.float32)
