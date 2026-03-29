from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol, MolFromSmarts
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.bases.base_fp_transformer import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class BaseSubstructureFingerprint(BaseFingerprintTransformer):
    """
    Base class for substructure fingerprints.

    The implementation uses RDKit. Substructure fingerprint checks for presence
    (or number, depending on ``count`` parameter) of provided molecular patterns.
    Number of features in fingerprint is equal to the number of patterns it was
    constructed from.

    This class is not meant to be used directly. If you want to use custom SMARTS
    patterns, inherit from this class and pass the ``patterns`` parameter to the
    parent constructor.

    Parameters
    ----------
    patterns : Sequence[str]
        Sequence of molecular patterns in SMARTS format.

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
        Number of output features, size of patterns. Equal to length of ``patterns``.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "patterns": [list],
    }

    def __init__(
        self,
        patterns: Sequence[str],
        count: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=len(patterns),
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.patterns = self._compile_smarts_patterns(patterns)

    def _compile_smarts_patterns(self, patterns: Sequence[str]) -> list[Mol]:
        # we have to perform validation manually, since we want to compile SMARTS
        # patterns in constructor, and this happens before scikit-learn calls
        # parameter validation
        if not all(isinstance(pattern, str) for pattern in patterns):
            raise InvalidParameterError(
                "The 'patterns' parameter must be a sequence of SMARTS patterns."
            )
        if not patterns:
            raise InvalidParameterError(
                "The 'patterns' parameter must be a non-empty list of SMARTS patterns."
            )

        compiled_patterns = []
        for pattern in patterns:
            pattern_mol = MolFromSmarts(pattern)
            if not pattern_mol:
                raise InvalidParameterError(f"Got invalid SMARTS pattern: '{pattern}'")
            else:
                compiled_patterns.append(pattern_mol)

        return compiled_patterns

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = ensure_mols(X)

        X = [
            [len(mol.GetSubstructMatches(pattern)) for pattern in self.patterns]
            for mol in X
        ]
        X = np.array(X, dtype=np.uint32)

        if not self.count:
            X = (X > 0).astype(np.uint8)

        return csr_array(X) if self.sparse else X
