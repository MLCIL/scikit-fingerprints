import json
import warnings
from collections.abc import Callable, Sequence
from functools import lru_cache
from numbers import Real
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol
from scipy import stats
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols, no_rdkit_logs

_UNBRANCHED_ALKANE_PATTERN = Chem.MolFromSmarts("[R0;D2][R0;D2][R0;D2][R0;D2]")


def _count_unbranched_alkanes(mol: Mol) -> int:
    return len(mol.GetSubstructMatches(_UNBRANCHED_ALKANE_PATTERN, uniquify=True))


@lru_cache(maxsize=1)
def _get_rdkit_2d_descriptors() -> list[tuple[str, Callable]]:
    path = Path(__file__).parent / "data" / "rdkit_2d_descriptor_names.json"

    with path.open(encoding="utf-8") as file:
        names = json.load(file)

    functions = dict(Descriptors.descList)
    functions["fr_unbrch_alkane"] = _count_unbranched_alkanes

    missing = set(names) - set(functions)
    if missing:
        raise ValueError(f"Missing RDKit descriptors: {sorted(missing)}")

    return [(name, functions[name]) for name in names]


@lru_cache(maxsize=1)
def _get_rdkit_2d_distributions() -> dict:
    path = Path(__file__).parent / "data" / "rdkit_2d_cdf_novartis.json"

    with path.open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def _get_rdkit_2d_molpile_cdf() -> dict:
    path = Path(__file__).parent / "data" / "rdkit_2d_cdf_molpile.json"

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    return {
        name: (
            np.asarray(thresholds, dtype=np.float64),
            np.asarray(probabilities, dtype=np.float64),
        )
        for name, (thresholds, probabilities) in data.items()
    }


def _normalize_rdkit_2d(
    values: Sequence[float | None], feature_names: Sequence[str]
) -> np.ndarray:
    """Normalize RDKit 2D descriptors using the original CDF distributions."""
    distributions = _get_rdkit_2d_distributions()
    result = np.zeros(len(values), dtype=np.float64)

    for i, (name, value) in enumerate(zip(feature_names, values, strict=True)):
        if name not in distributions:
            continue

        dist_name, params, min_val, max_val = distributions[name]

        if dist_name in ("gilbrat", "gibrat"):
            dist_name = "gilbrat" if hasattr(stats, "gilbrat") else "gibrat"

        try:
            dist = getattr(stats, dist_name)
            clipped = np.clip(value, min_val, max_val)

            normalized = dist.cdf(
                clipped, *params[:-2], loc=params[-2], scale=params[-1]
            )

            result[i] = np.clip(normalized, 0.0, 1.0)
        except Exception:  # noqa: BLE001
            result[i] = 0.0

    return result


def _normalize_rdkit_2d_molpile(
    values: Sequence[float | None],
    feature_names: Sequence[str],
) -> np.ndarray:
    distributions = _get_rdkit_2d_molpile_cdf()
    result = np.zeros(len(values), dtype=np.float64)

    for i, (name, value) in enumerate(zip(feature_names, values, strict=True)):
        if name not in distributions or value is None or not np.isfinite(value):
            continue

        thresholds, probabilities = distributions[name]
        index = np.searchsorted(thresholds, value, side="right") - 1

        if index >= 0:
            result[i] = probabilities[index]

    return result


class RDKit2DDescriptorsFingerprint(BaseFingerprintTransformer):
    """
    RDKit 2D descriptors fingerprint.

    The implementation uses descriptastorus [1]_ and RDKit. This fingerprint consists
    of 200 2D descriptors available in RDKit (almost all). List of all features is
    available in descriptastorus code and in the supplementary material of the original
    paper [2]_.

    Normalized variant uses cumulative distribution function (CDF) normalization, as
    proposed in [2]_. Distributions for normalization have been determined using a
    large collection of molecules from ChEMBL [3]_.

    Typical correct values should be small, but it often results in NaN or infinity
    for some descriptors. Value clipping with ``clip_val`` parameter, feature selection,
    and/or imputation should be used.

    Parameters
    ----------
    normalized : bool, default=False
        Whether to return CDF-normalized descriptor values.

    clip_val : float or None, default=2147483647
        Value to clip results at, both positive and negative ones.The default value is
        the maximal value of 32-bit integer, but should often be set lower, depending
        on the application. ``None`` means that no clipping is applied.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 200
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] Descriptastorus
        https://github.com/bp-kelley/descriptastorus

    .. [2] `Kevin Yang et al.
        "Analyzing Learned Molecular Representations for Property Prediction"
        Journal of Chemical Information and Modeling 59.8 (2019): 3370-3388
        <https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237>`_

    .. [3] Descriptastorus normalized descriptors discussion
        https://github.com/bp-kelley/descriptastorus/issues/3

    Examples
    --------
    >>> from skfp.fingerprints import RDKit2DDescriptorsFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = RDKit2DDescriptorsFingerprint()
    >>> fp
    RDKit2DDescriptorsFingerprint()

    >>> fp.transform(smiles)  # doctest: +SKIP
    array([[ 1.00000000e+00,  0.00000000e+00,  ...  0.00000000e+00,  3.27747673e-01]
           [ 1.00000000e+00,  1.00000000e+00,  ...  0.00000000e+00,  3.72785568e-01]
           [ 1.00000000e+00,  3.00000000e+00,  ...  0.00000000e+00,  3.44374359e-01]
           [ 1.00000000e+00,  2.18749619e+00,  ...  0.00000000e+00,  3.55007619e-01]], dtype=float32)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "normalized": ["boolean", None],
        "normalization": [StrOptions({"none", "descriptastorus", "molpile"})],
        "clip_val": [None, Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        normalized: bool | None = None,
        clip_val: float = 2147483647,  # max int32 value
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        normalization: str = "none",
    ):
        super().__init__(
            n_features_out=200,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.normalized = normalized
        self.clip_val = clip_val

        self.normalization = normalization

        if normalized is not None:
            if normalization != "none":
                raise ValueError(
                    "Use either 'normalized' or 'normalization', not both."
                )

            warnings.warn(
                "The 'normalized' parameter is deprecated. Use 'normalization' instead.",
                FutureWarning,
                stacklevel=2,
            )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to RDKit function
        names for computing descriptors.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            RDKit 2D descriptor names.
        """
        feature_names = [name for name, _ in _get_rdkit_2d_descriptors()]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute fingerprints consisting of all RDKit 2D descriptors.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 200)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        mols = ensure_mols(X)
        descriptors = _get_rdkit_2d_descriptors()
        feature_names = [name for name, _ in descriptors]

        if self.normalized is None:
            use_normalization = self.normalization == "descriptastorus"
        else:
            use_normalization = self.normalized

        # turn off RDKit logs, since descriptastorus does not use MorganGenerator
        # and generates a lot of warnings
        with no_rdkit_logs():
            results = []
            for mol in mols:
                raw_values = []

                for _, func in descriptors:
                    try:
                        raw_values.append(func(mol))
                    except Exception:  # noqa: BLE001
                        raw_values.append(None)

                if self.normalization == "molpile":
                    values = _normalize_rdkit_2d_molpile(raw_values, feature_names)
                elif use_normalization:
                    values = _normalize_rdkit_2d(raw_values, feature_names)
                else:
                    values = np.asarray(raw_values, dtype=np.float64)

                results.append(values)

        # clip values to float32 range
        results = [np.clip(x, -self.clip_val, self.clip_val) for x in results]

        if self.sparse:
            return csr_array(results, dtype=np.float32)
        else:
            return np.array(results, dtype=np.float32)
