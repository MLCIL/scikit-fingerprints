from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from rdkit.Chem import Mol, RemoveAllHs
from rdkit.Chem.rdMolDescriptors import _CalcCrippenContribs
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.descriptors import burden_matrix
from skfp.descriptors.charge import atomic_partial_charges
from skfp.utils import ensure_mols


class BCUT2DFingerprint(BaseFingerprintTransformer):
    """
    Burden-CAS-University of Texas (BCUT2D) fingerprint.

    This is a descriptor-based fingerprint, based on Burden descriptors, which are
    largest and smallest eigenvalues of the Burden matrix [1]_ [2]_. It is a modified
    connectivity matrix, aimed to combine topological structure with atomic properties.
    Diagonal elements are atom descriptors, e.g. atomic number, charge. Off-diagonal
    elements for bonded atoms are 1/sqrt(bond order), with minimum of 0.001 in case of
    no bond between given pair of atoms.

    BCUT2D descriptors use the largest and smallest eigenvalue for 4 atomic properties:
    mass, charge, logP and molar refractivity (MR). This results in 8 features.

    The implementation differs slightly from RDKit [3]_, because they use Gasteiger model
    by default, and here formal charge model is used. It is simpler and more robust, since
    Gasteiger fails for metal atoms. Like RDKit, we use Wildman-Crippen atomic contributions
    model [4]_ for logP and MR.

    Parameters
    ----------
    partial_charge_model : {"Gasteiger", "MMFF94", "formal", "precomputed"}, default="formal"
        Which model to use to compute atomic partial charges. Default ``"formal"``
        computes formal charges, and is the simplest and most error-resistant one.
        ``"precomputed"`` assumes that every atom of the input molecules has a
        "charge" float property set, e.g. with ``atom.SetDoubleProp("charge", value)``.

    charge_errors : {"raise", "ignore", "zero"}, default="raise"
        How to handle errors during calculation of atomic partial charges. ``"raise"``
        immediately raises any errors. ``"NaN"`` ignores any atoms that failed the
        computation; note that if all atoms fail, the error will be raised (use
        ``errors`` parameter to control this). ``"zero"`` uses default value of 0 to
        fill all problematic charges.

    errors : {"raise", "NaN", "ignore"}, default="raise"
        How to handle errors during fingerprint calculation. ``"raise"`` immediately
        raises any errors. ``"NaN"`` returns NaN values for molecules which resulted in
        errors. ``"ignore"`` suppresses errors and does not return anything for
        molecules with errors. This potentially results in fewer output vectors than
        input molecules, and should be used with caution.

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
    n_features_out : int = 8
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Frank R. Burden
        "Molecular identification number for substructure searches"
        J. Chem. Inf. Comput. Sci. 1989, 29, 3, 225–227
        <https://doi.org/10.1021/ci00063a011>`_

    .. [2] `R. Todeschini, V. Consonni
        "Molecular Descriptors for Chemoinformatics"
        Wiley‐VCH Verlag GmbH & Co. KGaA
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    .. [3] `RDKit BCUT2D descriptors
        <https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.BCUT2D>`_

    .. [4] `Scott A. Wildman and Gordon M. Crippen
        "Prediction of Physicochemical Parameters by Atomic Contributions"
        J. Chem. Inf. Comput. Sci. 1999, 39, 5, 868-873
        <https://pubs.acs.org/doi/10.1021/ci990307l>`_

    Examples
    --------
    >>> from skfp.fingerprints import BCUT2DFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = BCUT2DFingerprint()
    >>> fp
    BCUT2DFingerprint()

    >>> fp.transform(smiles)
    array([[15.999     , 15.999     ,  0.        ,  0.        , -0.2893    ,
            -0.2893    ,  0.8238    ,  0.8238    ],
           [13.011     , 11.011     ,  1.        , -1.        ,  1.1441    ,
            -0.8559    ,  3.503     ,  1.503     ],
           [14.16196892, 11.85603108,  0.26376262, -1.26376262,  0.6264836 ,
            -0.5301136 ,  3.43763218,  1.53036782],
           [16.12814585, 10.96029404,  1.22521641, -1.2242736 ,  1.12869643,
            -1.35803037,  5.43954434, -0.10561046]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "partial_charge_model": [
            StrOptions({"Gasteiger", "MMFF94", "formal", "precomputed"})
        ],
        "charge_errors": [StrOptions({"raise", "ignore", "zero"})],
        "errors": [StrOptions({"raise", "NaN", "ignore"})],
    }

    def __init__(
        self,
        partial_charge_model: str = "formal",
        charge_errors: str = "raise",
        errors: str = "raise",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=8,
            requires_conformers=False,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.partial_charge_model = partial_charge_model
        self.charge_errors = charge_errors
        self.errors = errors

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They are largest and smallest
        eigenvalues of Burden matrix for 4 atomic properties.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            BCUT2D feature names.
        """
        feature_names = [
            "max Burden eigenvalue mass",
            "min Burden eigenvalue mass",
            "max Burden eigenvalue charge",
            "min Burden eigenvalue charge",
            "max Burden eigenvalue logP",
            "min Burden eigenvalue logP",
            "max Burden eigenvalue MR",
            "min Burden eigenvalue MR",
        ]
        return np.asarray(feature_names, dtype=object)

    def transform(self, X: Sequence[Mol], copy: bool = False) -> np.ndarray | csr_array:
        """
        Compute BCUT2D fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 15)
            Array with fingerprints.
        """
        y = np.empty(len(X))
        X, _ = self.transform_x_y(X, y, copy=copy)
        return X

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[np.ndarray | csr_array, np.ndarray]:
        """
        Compute BCUT2D fingerprints. The returned values for X and y are
        properly synchronized.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=False
            Copy the input labels y or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 15)
            Array with fingerprints.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.
        """
        if copy:
            y = deepcopy(y)

        X = super().transform(X)

        if self.errors == "ignore":
            # errors are marked as NaN rows
            idxs_to_keep = [idx for idx, x in enumerate(X) if not np.any(np.isnan(x))]
            X = X[idxs_to_keep]
            y = y[idxs_to_keep]

        return X, y

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> np.ndarray | csr_array:
        mols = ensure_mols(X)

        if self.errors == "raise":
            fps = [self._get_fp(mol) for mol in mols]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = self._get_fp(mol)
                except ValueError:
                    fp = np.full(self.n_features_out, np.nan)
                fps.append(fp)

        return np.array(fps)

    def _get_fp(self, mol: Mol) -> np.ndarray:
        # Burden descriptors are defined for hydrogen-depleted molecule
        mol = RemoveAllHs(mol)

        # atomic descriptors: mass, partial charge, logP, molar refractivity
        masses = [atom.GetMass() for atom in mol.GetAtoms()]
        charges = atomic_partial_charges(
            mol, self.partial_charge_model, self.charge_errors
        )
        if self.charge_errors == "ignore":
            charges = charges[~np.isnan(charges)]
        elif self.charge_errors == "zero":
            charges = np.nan_to_num(charges, nan=0)

        atomic_logp_mr_contribs = _CalcCrippenContribs(mol)
        logp_vals = [logp for logp, mr in atomic_logp_mr_contribs]
        mr_vals = [mr for logp, mr in atomic_logp_mr_contribs]

        matrix = burden_matrix(mol)
        fp = np.empty(8, dtype=float)

        np.fill_diagonal(matrix, masses)
        fp[0:2] = self._get_max_and_min_eigenvals(matrix)

        np.fill_diagonal(matrix, charges)
        fp[2:4] = self._get_max_and_min_eigenvals(matrix)

        np.fill_diagonal(matrix, logp_vals)
        fp[4:6] = self._get_max_and_min_eigenvals(matrix)

        np.fill_diagonal(matrix, mr_vals)
        fp[6:8] = self._get_max_and_min_eigenvals(matrix)

        return fp

    @staticmethod
    def _get_max_and_min_eigenvals(arr: np.ndarray) -> tuple[float, float]:
        eigvals = np.linalg.eigvals(arr)
        # since Burden matrix is symmetric, those eigenvalues should always be real
        # NumPy sometimes prints a warning about complex number, no idea why, so we
        # extract real part manually
        max_eigval = np.real(np.max(eigvals))
        min_eigval = np.real(np.min(eigvals))
        return max_eigval, min_eigval
