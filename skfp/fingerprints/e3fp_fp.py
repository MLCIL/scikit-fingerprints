import logging
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import scipy.sparse
from rdkit import RDLogger
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import require_mols_with_conf_ids

"""
Note: this file cannot have the "e3fp.py" name due to conflict with E3FP library.
"""


class E3FPFingerprint(BaseFingerprintTransformer):
    """
    E3FP (Extended 3-Dimensional FingerPrint) fingerprint.

    The implementation uses ``e3fp`` library. This is a hashed fingerprint [1]_, where
    fragments are computed based on "shells", i.e. spherical areas around each
    atom in the 3D conformation of a molecule. The initial vector is quite large,
    and is then folded to the ``fp_size`` length.

    Shells are created around each atom with increasing radius, multiplied each
    time by ``radius_multiplier``, until ``level`` iterations are reached or when there
    is no change. Shell of each radius is hashed.

    Each shells get an identifier based on atom types in their radius, which is then
    hashed. Atom types (invariants) by default are based on Daylight invariants:

    - number of heavy neighbors
    - valence (excluding hydrogen neighbors)
    - atomic number
    - atomic mass
    - formal charge
    - number of bound hydrogens
    - whether it is a part of a ring

    This is a 3D fingerprint, and requires molecules with ``conf_id`` integer property
    set. They can be generated with :class:`~skfp.preprocessing.ConformerGenerator`.

    Parameters
    ----------
    fp_size : int, default=1024
        Size of output vectors, i.e. number of bits for each fingerprint. Must be
        positive.

    n_bits_before_folding : int, default=4096
        Size of fingerprint vector after initial hashing. It is then folded to
        ``fp_size``. Must be positive, and larger or equal to ``fp_size``.

    level : int, default=None
        Maximal number of iterations with increasing shell radius. None means that
        it stops only when the is no change from the last iteration.

    radius_multiplier : float, default=1.718
        How much to multiply the radius by to get the next, larger shell. Must be
        greater than 1.

    rdkit_invariants : bool, default=False
        Whether to use RDKit ECFP invariants instead of Daylight ones.

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

    random_state : int, RandomState instance or None, default=0
        Controls the randomness of conformer generation.

    Attributes
    ----------
    n_features_out : int
        Number of output features, size of fingerprints. Equal to ``fp_size``.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D based. It always requires
        molecules with conformers as inputs, with ``conf_id`` integer property set.

    See Also
    --------
    :class:`ECFPFingerprint` : Related 2D fingerprint. E3FP was designed to extend it
        to 3D features.

    References
    ----------
    .. [1] `Axen, Seth D., et al.
        "A simple representation of three-dimensional molecular structure"
        J. Med. Chem. 2017, 60, 17, 7393-7409
        <https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.7b00696>`_

    Examples
    --------
    >>> from skfp.fingerprints import E3FPFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = E3FPFingerprint()
    >>> fp
    E3FPFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], shape=(4, 1024), dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "n_bits_before_folding": [Interval(Integral, 1, None, closed="left")],
        "level": [None, Interval(Integral, 1, None, closed="left")],
        "radius_multiplier": [Interval(Real, 1.0, None, closed="neither")],
        "rdkit_invariants": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 1024,
        n_bits_before_folding: int = 4096,
        level: int | None = None,
        radius_multiplier: float = 1.718,
        rdkit_invariants: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            requires_conformers=True,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.n_bits_before_folding = n_bits_before_folding
        self.level = level
        self.radius_multiplier = radius_multiplier
        self.rdkit_invariants = rdkit_invariants

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.n_bits_before_folding < self.fp_size:
            raise InvalidParameterError(
                f"n_bits_before_folding must be greater of equal to fp_size, got:"
                f"n_bits_before_folding={self.n_bits_before_folding}, "
                f"fp_size={self.fp_size}"
            )

    def transform(self, X: Sequence[Mol], copy: bool = False) -> np.ndarray | csr_array:
        """
        Compute E3FP fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> np.ndarray | csr_array:
        X = require_mols_with_conf_ids(X)
        X = [self._calculate_single_mol_fingerprint(mol) for mol in X]
        return scipy.sparse.vstack(X) if self.sparse else np.array(X)

    def _calculate_single_mol_fingerprint(self, mol: Mol) -> np.ndarray | csr_array:
        from e3fp.pipeline import fprints_from_mol

        # suppress flood of logs
        try:
            if not self.verbose:
                logging.disable(logging.INFO)
                RDLogger.DisableLog("rdApp.*")

            fps = fprints_from_mol(
                mol,
                fprint_params={
                    "bits": self.n_bits_before_folding,
                    "level": self.level,
                    "radius_multiplier": self.radius_multiplier,
                    "rdkit_invariants": self.rdkit_invariants,
                    "counts": self.count,
                },
            )
        finally:
            RDLogger.EnableLog("rdApp.*")
            logging.disable(logging.NOTSET)

        fp = fps[0]
        fp = fp.fold(self.fp_size)
        dtype = np.uint32 if self.count else np.uint8
        return fp.to_vector(sparse=self.sparse, dtype=dtype)
