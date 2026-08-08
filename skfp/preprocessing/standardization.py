from collections.abc import Sequence
from contextlib import nullcontext

from rdkit.Chem import Mol, SanitizeMol
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    CleanupInPlace,
    FragmentParentInPlace,
)

from skfp.bases import BasePreprocessor
from skfp.utils import ensure_mols, no_rdkit_logs


class MolStandardizer(BasePreprocessor):
    """
    Performs common molecule standardization operations.

    Applies the following cleanup transformations to the inputs:
    - create RDKit ``Mol`` objects, if SMILES strings are passed
    - sanitize [1]_ (performs basic validity checks)
    - if ``largest_fragment_only``, select the largest fragment for further processing
    - remove hydrogens
    - disconnect metal atoms
    - normalize (transform functional groups to normal form)
    - reionize

    See [1]_ [2]_ [3]_ [4]_ [5]_ for details, rationale and alternatives. Note that
    there is no one-size-fits-all solution, and here we use pretty minimalistic, most
    common steps. This class by design does not allow much parametrization, and for
    custom purposes you should build the pipeline yourself.

    New molecules are always returned, and any set properties are not kept, as this
    should normally be the first step in a pipeline.

    Parameters
    ----------
    largest_fragment_only : bool, default=False
        Whether to select only the largest fragment from each molecule.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized over
        the input molecules. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See scikit-learn documentation on
        ``n_jobs`` for more details.

    verbose : int or dict, default=0
        Controls the verbosity when standardizing molecules. By default, all warnings
        are turned off. If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `Gregory Landrum
        "The RDKit Book: Molecular Sanitization"
        <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_

    .. [2] `RDKit mailing list thread on molecular sanitization
        <https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CANjYGkS%3Dqz3NbmdhGAEvfOec0o%3DCL%3D43aex%3DT1RGmUDgSb63og%40mail.gmail.com/>`_

    .. [3] `Matt Swain
        "MolVS Standardization"
        <https://molvs.readthedocs.io/en/latest/guide/standardize.html>`_

    .. [4] `Gregory Landrum
        "RSC OpenScience workshop: Chemical structure validation and standardization with the RDKit"
        <https://github.com/greglandrum/RSC_OpenScience_Standardization_202104>`_

    .. [5] `Jean-Paul Ebejer
        "Standardizing a molecule using RDKit"
        <https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolStandardizer
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> standardizer = MolStandardizer()
    >>> standardizer
    MolStandardizer()

    >>> standardizer.transform(smiles)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        **BasePreprocessor._parameter_constraints,
        "largest_fragment_only": ["boolean"],
    }

    def __init__(
        self,
        largest_fragment_only: bool = False,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__()
        self.largest_fragment_only = largest_fragment_only
        self.n_jobs = n_jobs
        self.verbose = verbose

    def transform(self, X: Sequence[str | Mol], copy: bool = True) -> list[Mol]:
        """
        Standardize molecule structures.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=True
            Copy the input X or not. In contrast to most classes, input molecules
            are copied by default, since RDKit standardizes their structure in place.

        Returns
        -------
        X : list of shape (n_samples,)
            List with RDKit ``Mol`` objects.
        """
        return super().transform(X, copy)

    def _transform_batch(self, X: Sequence[str | Mol]) -> list[Mol]:
        # if SMILES is provided, MolFromSmiles also includes sanitization
        mols = ensure_mols(X)

        with nullcontext() if self.verbose else no_rdkit_logs():
            # we can safely modify in place here, thanks to default copy=True
            for mol in mols:
                # basic sanitization, if Mol objects were passed
                SanitizeMol(mol)

                # select the largest ("parent") fragment if needed
                if self.largest_fragment_only:
                    FragmentParentInPlace(mol)

                # remove Hs, disconnect metals, normalize functional groups, reionize
                CleanupInPlace(mol)

        return mols
