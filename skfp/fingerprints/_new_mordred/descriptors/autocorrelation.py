"""
Autocorrelation descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    AUTOCORRELATION_ALL_PROPERTIES,
    AUTOCORRELATION_ATS_PROPERTIES,
    AUTOCORRELATION_MAX_DISTANCE,
    MordredMolCache,
)

MAX_DISTANCE = AUTOCORRELATION_MAX_DISTANCE
ATS_PROPERTIES = AUTOCORRELATION_ATS_PROPERTIES
ALL_PROPERTIES = AUTOCORRELATION_ALL_PROPERTIES

FEATURE_NAMES = [
    *[
        f"ATS{distance}{prop}"
        for prop in ATS_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"AATS{distance}{prop}"
        for prop in ATS_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"ATSC{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"AATSC{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(MAX_DISTANCE + 1)
    ],
    *[
        f"MATS{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(1, MAX_DISTANCE + 1)
    ],
    *[
        f"GATS{distance}{prop}"
        for prop in ALL_PROPERTIES
        for distance in range(1, MAX_DISTANCE + 1)
    ],
]


def _nan_if_not_finite(value: float) -> float:
    return value if np.isfinite(value) else np.nan


def _safe_divide(numerator: float, denominator: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _nan_if_not_finite(float(np.divide(numerator, denominator)))


def _ats(w: np.ndarray, gmat: np.ndarray, order: int) -> float:
    if order == 0:
        return float((w**2).sum())
    return float(0.5 * w.dot(gmat).dot(w))


def _aats(ats: float, gsum: float) -> float:
    return _safe_divide(ats, gsum)


def _atsc(c: np.ndarray, gmat: np.ndarray, order: int) -> float:
    if order == 0:
        return float((c**2).sum())
    return float(0.5 * c.dot(gmat).dot(c))


def _mats(w: np.ndarray, aatsc: float, c: np.ndarray) -> float:
    return _safe_divide(len(w) * aatsc, float((c**2).sum()))


def _gats(w: np.ndarray, c: np.ndarray, gmat: np.ndarray, gsum: float) -> float:
    if len(w) <= 1:
        return np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (gmat * (w[:, np.newaxis] - w) ** 2).sum() / (4 * gsum)
        denominator = (c**2).sum() / (len(w) - 1)
        return _nan_if_not_finite(float(numerator / denominator))


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred autocorrelation descriptors.

    The 2D autocorrelation descriptors are calculated without adding explicit
    hydrogens, matching the rest of the new Mordred 2D implementation.
    """
    gmats = cache.autocorrelation_gmats
    gsums = cache.autocorrelation_gsums
    weights = cache.autocorrelation_weights
    centered_weights = cache.autocorrelation_centered_weights

    ats_values: dict[str, list[float]] = {}
    aats_values: dict[str, list[float]] = {}
    for prop in ATS_PROPERTIES:
        w = weights[prop]
        ats_values[prop] = [_ats(w, gmat, order) for order, gmat in enumerate(gmats)]
        aats_values[prop] = [
            _aats(ats, gsum) for ats, gsum in zip(ats_values[prop], gsums, strict=True)
        ]

    atsc_values: dict[str, list[float]] = {}
    aatsc_values: dict[str, list[float]] = {}
    for prop in ALL_PROPERTIES:
        c = centered_weights[prop]
        atsc_values[prop] = [_atsc(c, gmat, order) for order, gmat in enumerate(gmats)]
        aatsc_values[prop] = [
            _aats(atsc, gsum)
            for atsc, gsum in zip(atsc_values[prop], gsums, strict=True)
        ]

    values: list[float] = []
    for prop in ATS_PROPERTIES:
        values.extend(ats_values[prop])
    for prop in ATS_PROPERTIES:
        values.extend(aats_values[prop])
    for prop in ALL_PROPERTIES:
        values.extend(atsc_values[prop])
    for prop in ALL_PROPERTIES:
        values.extend(aatsc_values[prop])
    for prop in ALL_PROPERTIES:
        w = weights[prop]
        c = centered_weights[prop]
        values.extend(_mats(w, aatsc_values[prop][order], c) for order in range(1, 9))
    for prop in ALL_PROPERTIES:
        w = weights[prop]
        c = centered_weights[prop]
        values.extend(_gats(w, c, gmats[order], gsums[order]) for order in range(1, 9))

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
