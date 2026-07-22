from collections.abc import Sequence
from numbers import Integral

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols, get_conf_id, require_mols_with_conformations


class PharmacophoreFingerprint(BaseFingerprintTransformer):
    """
    Pharmacophore fingerprint.

    The implementation uses RDKit. This is a hashed fingerprint, where fragments are
    computed based on N-point tuples, using pharmacophoric points.

    An N-point pharmacophoric structure encodes N pharmacophoric points and pairwise
    distances between them, e.g. 3-point pharmacophore uses 6-element tuples
    (P1 D12 P2 D23 P3 D13). P is a pharmacophoric point, atom or subgraph, of a particular
    type (see below), and Dij is a topological distance (shortest path) between points
    i and j. Distance values are limited to 8 (higher values are capped at 8).

    Pharmacophoric point types are (based on SMARTS patterns definitions from [1]_):
    - hydrophobic atom
    - hydrogen bond donor
    - hydrogen bond acceptor
    - aromatic attachment
    - aliphatic attachment
    - "unusual" atom (not H, C, N, O, F, S, Cl, Br, I)
    - basic group
    - acidic group

    By default, 2-point and 3-point pharmacophores are used, resulting in 39972-element
    vector. Alternatively, folding can be applied. Note that due to RDKit limitations,
    only 2-point and 3-point variants are available.

    Note that this is by far the slowest fingerprint, particularly for larger molecules.
    This is due to the 3-point pharmacophore calculation. Consider filtering out large
    (heavy) molecules or setting ``max_points=2`` if it takes too long.

    Parameters
    ----------
    variant: {"raw_bits", "folded"} = "raw_bits"
        Whether to return raw bit values, or to fold them. Length of raw bits variant
        depends on used N-points, see ``n_features_out`` attribute.

    min_points: int, default=2
        Lower bound of N-point pharmacophore. Must be 2 or 3, and less or equal to
        ``max_points``.

    max_points: int, default=3
        Upper bound of N-point pharmacophore. Must be 2 or 3, and greater or equal to
        ``min_points``.

    fp_size : int, default=2048
        Size of output vectors, i.e. number of bits for each fingerprint. Only used
        for `"folded"` variant. Must be positive.

    use_3D : bool, default=False
        Whether to use 3D Euclidean distance matrix, instead of topological distance.
        Binning is used to discretize values into values 0-8.

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
    n_features_out : int, default=39972
        Number of output features, size of fingerprints. For ``"folded"`` variant,
        it is equal to ``fp_size``. For ``"raw_bits"`` variant, it depends on
        ``min_points`` and ``max_points``: 252 for (2,2), 39720 for (3,3), and
        39972 for (2,3).

    requires_conformers : bool
        Whether the fingerprint is 3D-based and requires molecules with conformers as
        inputs. This depends on the ``use_3D`` parameter, and has the same value. If the
        ``conf_id`` property is set, it will be used to determine the conformation.
        You can use :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

    References
    ----------
    .. [1] `A Gobbi, D Poppinger
        "Genetic optimization of combinatorial libraries"
        Biotechnology and Bioengineering: Volume 61, Issue 1, Winter 1998, Pages 47-54
        <https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z>`_

    Examples
    --------
    >>> from skfp.fingerprints import PharmacophoreFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = PharmacophoreFingerprint()
    >>> fp
    PharmacophoreFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], shape=(4, 39972), dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "variant": [StrOptions({"raw_bits", "folded"})],
        "min_points": [Interval(Integral, 2, 3, closed="both")],
        "max_points": [Interval(Integral, 2, 3, closed="both")],
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        variant: str = "raw_bits",
        min_points: int = 2,
        max_points: int = 3,
        fp_size: int = 2048,
        use_3D: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        n_features_out = self._get_n_features_out(
            variant, min_points, max_points, fp_size
        )
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.variant = variant
        self.min_points = min_points
        self.max_points = max_points
        self.fp_size = fp_size
        self.use_3D = use_3D

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_points < self.min_points:
            raise InvalidParameterError(
                f"The max_points parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_points, got: "
                f"min_points={self.min_points}, max_points={self.max_points}"
            )

    def _get_n_features_out(
        self, variant: str, min_points: int, max_points: int, fp_size: int
    ) -> int:
        if variant == "folded":
            return fp_size

        # "raw_bits"
        if min_points == max_points == 2:
            return 252
        elif min_points == max_points == 3:
            return 39720
        elif min_points == 2 and max_points == 3:
            return 39972
        else:
            raise ValueError(
                "min_points and max_points must be 2 or 3, "
                "and min_points <= max_points, got:"
                f"min_points={min_points}, max_points={max_points}"
            )

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute MACCS fingerprints. Output shape depends on ``min_points``
        and ``max_points`` parameters.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.n_features_out)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem import Get3DDistanceMatrix
        from rdkit.Chem.ChemicalFeatures import BuildFeatureFactoryFromString
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
        from rdkit.Chem.Pharm2D.SigFactory import SigFactory

        atom_features = BuildFeatureFactoryFromString(Gobbi_Pharm2D.fdef)
        factory = SigFactory(
            atom_features,
            minPointCount=self.min_points,
            maxPointCount=self.max_points,
            useCounts=self.count,
        )
        factory.SetBins(Gobbi_Pharm2D.defaultBins)
        factory.Init()

        if not self.use_3D:
            X = ensure_mols(X)
            X = [Gen2DFingerprint(mol, factory) for mol in X]
        else:
            X = require_mols_with_conformations(X)
            X = [
                Gen2DFingerprint(
                    mol,
                    factory,
                    dMat=Get3DDistanceMatrix(mol, confId=get_conf_id(mol)),
                )
                for mol in X
            ]

        if self.variant == "folded":
            # X at this point is a list of RDKit fingerprints, but MyPy doesn't get it
            return self._hash_fingerprint_bits(
                X,  # type: ignore
                fp_size=self.fp_size,
                count=self.count,
                sparse=self.sparse,
            )

        dtype = np.uint32 if self.count else np.uint8

        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
