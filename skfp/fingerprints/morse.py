from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import get_conf_id, require_mols_with_conformations


class MORSEFingerprint(BaseFingerprintTransformer):
    r"""
    MoRSE (Molecule Representation of Structures based on Electron diffraction)
    fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, where
    features encode basic 3D atomic coordinates (molecule structure) with a
    constant-size vector, based on principles similar to electron diffraction.

    MoRSE features are scattered electron intensity values for different values
    of scattering ``s``. It uses general molecular transform which, for all atoms i and j
    with distance :math:`r_{ij}` in the molecule with N atoms, is defined as:

    .. math::

        RDF(r) = \sum_{i}^{N-1} \sum_{j > i}^N w_i * w_j * \frac{\sin(s * r_{ij})}{s * r_{ij}}

    7 weighting variants are used, unweighted and 6 based on atomic features:
    unweighted, atomic mass, van der Waals volume, electronegativity, polarizability,
    ion polarity, and IState [1]_ [2]_. They are relative to the carbon, e.g. molecular
    weight is: MW(atom_type) / MW(carbon).

    Scattering values between 0 and 31 (inclusive) are used, in angstroms (Å).
    This results in 224 features. See [3]_ [4]_ [5]_ [6]_ [7]_ for details.

    Parameters
    ----------
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
    n_features_out : int = 224
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D-based. It always requires
        molecules with conformers as input. If the ``conf_id`` property is set, it
        will be used to determine the conformation. You can use
        :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

    See Also
    --------
    :class:`RDFFingerprint` : Related fingerprint, which uses radial distribution
        function (RDF) instead of scattered electron intensity.

    References
    ----------
    .. [1] RDKit IState discussion
        https://github.com/rdkit/rdkit/discussions/6122

    .. [2] RDKit IState implementation
        https://github.com/rdkit/rdkit/blob/df2b6c8a8c775748c1dcac83af0f92af434bab81/Code/GraphMol/Descriptors/MolData3Ddescriptors.cpp#L127

    .. [3] `DRAGON documentation
        "MoRSE descriptors"
        <http://chemgps.bmc.uu.se/help/dragonx/MoRSEdescriptors.html>`_

    .. [4] `Roberto Todeschini and Viviana Consonni
        "Molecular Descriptors for Chemoinformatics"
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    .. [5] `Guillaume Godin
        "3D	descriptors in RDKit"
        UGM 2017
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf>`_

    .. [6] `Jan H. Schuur, Paul Selzer, and Johann Gasteiger
        "The Coding of the Three-Dimensional Structure of Molecules by Molecular
        Transforms and Its Application to Structure-Spectra Correlations and Studies
        of Biological Activity"
        J. Chem. Inf. Comput. Sci. 1996, 36, 2, 334-344
        <https://pubs.acs.org/doi/abs/10.1021/ci950164c>`_

    .. [7] `Devinyak, Oleg, Dmytro Havrylyuk, and Roman Lesyk
        "3D-MoRSE descriptors explained"
        Journal of Molecular Graphics and Modelling 54 (2014): 194-203
        <https://www.sciencedirect.com/science/article/pii/S109332631400165X>`_

    Examples
    --------
    >>> from skfp.fingerprints import MORSEFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MORSEFingerprint()
    >>> fp
    MORSEFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[ 3.0000e+00,  2.3090e+00,  9.0900e-01, ..., -1.3000e-02,
            -8.3000e-02, -5.3000e-02],
           [ 2.8000e+01,  1.2408e+01, -8.1900e-01, ...,  3.5400e-01,
             4.8200e-01, -1.2000e-01],
           [ 1.0000e+00,  7.9500e-01,  3.2700e-01, ...,  8.6700e-01,
             1.2400e-01, -7.1200e-01],
           [ 2.1000e+01,  9.2430e+00, -4.3800e-01, ..., -5.8700e-01,
             2.7400e-01, -3.9000e-02]])
    """

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=224,
            requires_conformers=True,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to 7 weighting
        variants and 32 scattering values.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            MoRSE feature names.
        """
        feature_names = [
            f"{weighting_variant} {scattering_value}"
            for weighting_variant in [
                "unweighted",
                "atomic mass",
                "van der Waals volume",
                "electronegativity",
                "polarizability",
                "ion polarity",
                "IState",
            ]
            for scattering_value in range(32)
        ]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute MORSE fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 224)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem.rdMolDescriptors import CalcMORSE

        X = require_mols_with_conformations(X)
        X = [CalcMORSE(mol, confId=get_conf_id(mol)) for mol in X]
        return csr_array(X) if self.sparse else np.array(X)
