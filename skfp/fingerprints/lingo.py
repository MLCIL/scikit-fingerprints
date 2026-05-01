import functools
import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_smiles


class LingoFingerprint(BaseFingerprintTransformer):
    """
    Lingo fingerprint.

    The Lingo fingerprint [1]_ is a hashed fingerprint that checks the occurrences of
    substrings of a given length in a SMILES string. These strings are overlapping.
    Original paper uses raw dictionaries os output, but here constant-length hashed
    fingerprints are returned. SHA-1 hash function is used here.

    You can use method :meth:`smiles_to_dicts` to convert SMILES strings to
    dictionaries of substring, like original Lingo fingerprint.

    Parameters
    ----------
    fp_size : int, default=1024
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    substring_length : int, default=4
        Length of the substrings to count in the SMILES strings. Must be positive.

    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

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
    n_features_out : int
        Number of output features, size of fingerprints. Equal to ``fp_size``.

    References
    ----------
    .. [1] `David Vidal, Michael Thormann, and Miquel Pons
        "LINGO, an Efficient Holographic Text Based Method To Calculate
        Biophysical Properties and Intermolecular Similarities"
        J. Chem. Inf. Model. 2005, 45, 2, 386-393
        <https://pubs.acs.org/doi/abs/10.1021/ci0496797>`_

    Examples
    --------
    >>> from skfp.fingerprints import LingoFingerprint
    >>> smiles = ["[C-]#N", "CC(=O)NC1=CC=C(C=C1)O"]
    >>> fp = LingoFingerprint()
    >>> fp
    LingoFingerprint()
    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0]], shape=(2, 1024), dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "substring_length": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        fp_size: int = 1024,
        substring_length: int = 4,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.substring_length = substring_length

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute Lingo fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy X or modify it in place.

        Returns
        -------
        X: {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            2D array containing Lingo fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = self._smiles_to_dicts(X)
        X = self._dicts_to_array(X)
        return csr_array(X) if self.sparse else X

    def _smiles_to_dicts(self, X: Sequence[str | Mol]) -> list[dict[str, int]]:
        # SMILES -> dictionary of substring counts
        X = ensure_smiles(X)

        # based on the original paper, we reduce the number of possible substrings
        # to improve statistical sampling in the QSPR models

        # replace numbers for rings, but not charges in []
        # we use negative lookback/lookahead in regex
        X = [re.sub(r"(?<!\[)%?\d+(?!\])", "0", smi) for smi in X]
        X = [
            smi.replace("Cl", "L").replace("Br", "R")
            if "Cl" in smi or "Br" in smi
            else smi
            for smi in X
        ]

        k = self.substring_length
        return [Counter(smi[i : i + k] for i in range(len(smi) - k + 1)) for smi in X]

    def _dicts_to_array(self, X: list[dict[str, int]]) -> np.ndarray:
        dtype = np.uint32 if self.count else np.uint8
        result = np.zeros((len(X), self.fp_size), dtype=dtype)

        for i, dictionary in enumerate(X):
            for key, value in dictionary.items():
                hash_index = self._get_hash_index(key, self.fp_size)
                if self.count:
                    result[i, hash_index] += value
                else:
                    result[i, hash_index] = 1

        return result

    @staticmethod
    @functools.cache
    def _get_hash_index(key: str, fp_size: int) -> int:
        hash_bytes = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(hash_bytes, byteorder="big") % fp_size
