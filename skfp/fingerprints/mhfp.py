from collections.abc import Sequence
from numbers import Integral

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils.validators import ensure_smiles


class MHFPFingerprint(BaseFingerprintTransformer):
    """
    MinHashed FingerPrint (MHFP).

    The implementation uses RDKit. This is a hashed fingerprint [1]_, where
    fragments are computed based on circular substructures around each atom.

    Subgraphs are created around each atom with increasing radius, starting
    with just an atom itself. It is then transformed into a canonical SMILES
    and hashed into an integer. In each iteration, it is increased by another
    atom (one "hop" on the graph). The resulting hashes are MinHashed. Depending
    on ``variant`` argument, either those values are returned, or they are further
    folded (with modulo) into a vector of size ``fp_size``.

    Additionally, the SMILES strings of the symmetrized smallest set of smallest
    rings (SSSR) are included by default, to incorporate ring information for
    small radii.

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    radius : int, default=3
        Number of iterations performed, i.e. maximum radius of resulting subgraphs.
        Another common notation uses diameter, therefore ECFP4 has radius 2.

    min_radius : int, default=1
        Initial radius of subgraphs.

    sssr_rings : bool, default=True
        Whether to include the symmetrized smallest set of smallest rings (SSSR)
        in addition to circular subgraphs.

    isomeric_smiles : bool, default=False
        Whether to use isomeric SMILES, instead of just the canonical SMILES.

    kekulize : bool, default=True
        Whether to kekulize the subgraphs before SMILES generation.

    variant : {"bit", "count", "raw_hashes"}, default="bit"
        Fingerprint variant. "raw_hashes" follows the original paper and results in
        raw integer values of hashes. "bit" and "count" result in integer vectors
        with hashes folded with modulo operation into a ``fp_size`` length.

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

    random_state : int or None, default=0
        Controls the seed used when initializing the MinHash encoder. Pass an
        integer for reproducible results across multiple function calls.
        ``None`` means that no fixed seed is used.

    Attributes
    ----------
    n_features_out : int
        Number of output features. Equal to ``fp_size``.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    See Also
    --------
    :class:`ECFPFingerprint` : Related fingerprint, which uses atom invariants instead
        of raw SMILES strings.

    :class:`SECFPFingerprint` : Related fingerprint, which hashes subgraph SMILES and
        folds the resulting vector.

    References
    ----------
    .. [1] `Daniel Probst and Jean-Louis Reymond
        "A probabilistic molecular fingerprint for big data settings"
        Journal of Cheminformatics 10, 1-12 (2018)
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8>`_

    Examples
    --------
    >>> from skfp.fingerprints import MHFPFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MHFPFingerprint()
    >>> fp
    MHFPFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 1],
           [0, 1, 0, ..., 0, 0, 1],
           [1, 1, 0, ..., 1, 1, 0],
           [0, 0, 1, ..., 1, 0, 1]], shape=(4, 2048), dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "radius": [Interval(Integral, 0, None, closed="left")],
        "min_radius": [Interval(Integral, 0, None, closed="left")],
        "rings": ["boolean"],
        "isomeric": ["boolean"],
        "kekulize": ["boolean"],
        "variant": [StrOptions({"bit", "count", "raw_hashes"})],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        min_radius: int = 1,
        sssr_rings: bool = True,
        isomeric_smiles: bool = False,
        kekulize: bool = True,
        variant: str = "bit",
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        random_state: int | None = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.min_radius = min_radius
        self.sssr_rings = sssr_rings
        self.isomeric_smiles = isomeric_smiles
        self.kekulize = kekulize
        self.variant = variant

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.radius < self.min_radius:
            raise InvalidParameterError(
                f"The radius parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_radius, got: "
                f"min_radius={self.min_radius}, radius={self.radius}"
            )

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute MHFP fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

        X = ensure_smiles(X)

        # outputs raw hash values, not feature vectors!
        encoder = MHFPEncoder(self.fp_size, self.random_state)
        X = encoder.EncodeSmilesBulk(
            X,
            radius=self.radius,
            min_radius=self.min_radius,
            rings=self.sssr_rings,
            isomeric=self.isomeric_smiles,
            kekulize=self.kekulize,
        )
        X = np.array(X, dtype=np.uint32)

        if self.variant in {"bit", "count"}:
            X = np.mod(X, self.fp_size)
            X = np.stack([np.bincount(fp, minlength=self.fp_size) for fp in X])
            if self.variant == "bit":
                X = (X > 0).astype(np.uint8)
            else:
                X = X.astype(np.uint32)

        return csr_array(X) if self.sparse else np.array(X)
