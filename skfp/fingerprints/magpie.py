import itertools
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols

"""
Elemental properties used by MAGPIE, in the order defined by the original
implementation. Renamed from the terse names used in the original lookup tables,
e.g. "GSvolume_pa" or "NsUnfilled".
"""
_ELEMENTAL_PROPERTIES = [
    "atomic number",
    "Mendeleev number",
    "atomic weight",
    "melting temperature",
    "periodic table column",
    "periodic table row",
    "covalent radius",
    "electronegativity",
    "s valence electrons",
    "p valence electrons",
    "d valence electrons",
    "f valence electrons",
    "valence electrons",
    "unfilled s orbitals",
    "unfilled p orbitals",
    "unfilled d orbitals",
    "unfilled f orbitals",
    "unfilled orbitals",
    "ground state volume per atom",
    "ground state band gap",
    "ground state magnetic moment",
    "space group number",
]

_STATISTICS = ["mean", "range", "mean abs deviation", "max", "min", "mode"]

_P_NORMS = [2, 3, 5, 7, 10]


@lru_cache(maxsize=1)
def _load_elemental_data() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    # load static MAGPIE elemental property lookup tables
    # indexed by atomic number - 1 (zero-aligned), and per-element tuples
    # of common oxidation states
    filepath = Path(__file__).parent / "data" / "magpie_elemental_properties.csv"
    df = pd.read_csv(filepath, dtype={"oxidation states": str})

    properties = df[_ELEMENTAL_PROPERTIES].to_numpy(dtype=float)
    oxidation_states = tuple(
        tuple(int(state) for state in states.split())
        for states in df["oxidation states"].fillna("")
    )
    return properties, oxidation_states


class MAGPIEFingerprint(BaseFingerprintTransformer):
    """
    Materials Agnostic Platform for Informatics and Exploration (MAGPIE) fingerprint.

    This is a descriptor-based fingerprint, computed from the elemental composition
    (chemical formula) of a molecule [1]_. It was originally designed for inorganic
    materials, but is applicable to any compound with a well-defined stoichiometry.
    Here, the composition is the count of each element in the molecule, including
    hydrogens, normalized to fractions.

    Note that this fingerprint uses only the molecular formula and is therefore
    identical for all isomers, ignoring any structural information.

    It computes 145 features, in 4 groups:

    - stoichiometric attributes (6 features), depending only on element fractions,
      and not on element identity:

        - number of distinct elements
        - :math:`L^p` norms of the element fraction vector, for p = 2, 3, 5, 7, 10

    - elemental property statistics (132 features), i.e. 6 weighted statistics
      (mean, range, mean absolute deviation, max, min, mode) of 22 elemental
      properties: atomic number, Mendeleev number, atomic weight, melting
      temperature, periodic table column and row, covalent radius,
      electronegativity, number of valence electrons in the s, p, d and f orbitals
      and in total, number of unfilled s, p, d and f orbitals and unfilled orbitals
      in total, ground state volume per atom, ground state band gap energy, ground
      state magnetic moment, and space group number

    - valence orbital occupation attributes (4 features), i.e. the fraction of
      valence electrons in the s, p, d, and f orbitals

    - ionic compound attributes (3 features):

        - whether a charge-neutral ionic compound can be formed
        - maximal ionic character of any pair of constituent elements
        - mean ionic character, weighted by element fractions

    Elements are supported up to copernicium (atomic numbers 1-112). Some properties
    are undefined for a few elements, like electronegativity of noble gases, resulting
    in NaN values for some properties in those cases.

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
    n_features_out : int = 145
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only the molecular formula and does not require
        conformers.

    References
    ----------
    .. [1] `Logan Ward, Ankit Agrawal, Alok Choudhary, Christopher Wolverton
        "A general-purpose machine learning framework for predicting properties of
        inorganic materials"
        npj Computational Materials 2, 16028 (2016)
        <https://www.nature.com/articles/npjcompumats201628>`_

    Examples
    --------
    >>> from skfp.fingerprints import MAGPIEFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MAGPIEFingerprint()
    >>> fp
    MAGPIEFingerprint()

    >>> fp.transform(smiles).shape
    (4, 145)

    >>> fp.transform(smiles)[:, :3]  # number of elements and L2, L3 norms
    array([[2.        , 0.74535599, 0.69336127],
           [2.        , 0.79056942, 0.75914724],
           [2.        , 0.70710678, 0.62996052],
           [3.        , 0.65465367, 0.59704846]])
    """

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=145,
            requires_conformers=False,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            MAGPIE feature names.
        """
        feature_names = ["number of elements"]
        feature_names += [f"L{p} norm of element fractions" for p in _P_NORMS]
        feature_names += [
            f"{stat} {prop}" for prop in _ELEMENTAL_PROPERTIES for stat in _STATISTICS
        ]
        feature_names += [
            f"fraction of {orbital} valence electrons" for orbital in "spdf"
        ]
        feature_names += [
            "can form ionic compound",
            "max ionic character",
            "mean ionic character",
        ]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute MAGPIE fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 145)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = ensure_mols(X)
        X = np.array([self._get_magpie_features(mol) for mol in X], dtype=float)
        return csr_array(X) if self.sparse else X

    def _get_magpie_features(self, mol: Mol) -> np.ndarray:
        atomic_nums, fractions = self._get_composition(mol)
        properties, oxidation_states = _load_elemental_data()

        if np.any(atomic_nums > len(properties)):
            unsupported = sorted(set(atomic_nums[atomic_nums > len(properties)]))
            raise ValueError(
                f"MAGPIEFingerprint supports elements with atomic numbers 1-"
                f"{len(properties)}, got: {unsupported}"
            )

        # (n_elements, 22) matrix of elemental property values
        values = properties[atomic_nums - 1]

        return np.concatenate(
            [
                self._stoichiometric_features(fractions),
                self._elemental_property_features(values, fractions),
                self._valence_shell_features(values, fractions),
                self._ionicity_features(
                    values,
                    fractions,
                    [oxidation_states[num - 1] for num in atomic_nums],
                ),
            ]
        )

    def _get_composition(self, mol: Mol) -> tuple[np.ndarray, np.ndarray]:
        # elemental composition of molecules: atomic numbers and their fractions
        # includes implicit hydrogens
        counts: dict[int, int] = defaultdict(int)
        for atom in mol.GetAtoms():
            counts[atom.GetAtomicNum()] += 1
            counts[1] += atom.GetTotalNumHs()

        # atoms with atomic number 0 (dummy atoms) have no elemental properties
        counts = {num: count for num, count in counts.items() if num > 0 and count > 0}
        if not counts:
            raise ValueError("Cannot compute MAGPIE fingerprint for an empty molecule")

        atomic_nums = np.array(sorted(counts), dtype=int)
        fractions = np.array([counts[num] for num in atomic_nums], dtype=float)
        fractions /= fractions.sum()
        return atomic_nums, fractions

    def _stoichiometric_features(self, fractions: np.ndarray) -> np.ndarray:
        norms = [np.sum(fractions**p) ** (1 / p) for p in _P_NORMS]
        return np.array([len(fractions), *norms], dtype=float)

    def _elemental_property_features(
        self, values: np.ndarray, fractions: np.ndarray
    ) -> np.ndarray:
        mean = fractions @ values
        max_ = values.max(axis=0)
        min_ = values.min(axis=0)
        mean_abs_dev = fractions @ np.abs(values - mean)

        # mode is the average property value over the most prevalent elements
        most_prevalent = np.isclose(fractions, fractions.max())
        mode = values[most_prevalent].mean(axis=0)

        stats = np.stack([mean, max_ - min_, mean_abs_dev, max_, min_, mode], axis=1)
        return stats.ravel()

    def _valence_shell_features(
        self, values: np.ndarray, fractions: np.ndarray
    ) -> np.ndarray:
        valence_idx = [
            _ELEMENTAL_PROPERTIES.index(f"{orbital} valence electrons")
            for orbital in "spdf"
        ]
        valence_values = values[:, valence_idx]

        mean_valence = fractions @ valence_values
        total = mean_valence.sum()

        return mean_valence / total if total > 0 else np.full(4, np.nan)

    def _ionicity_features(
        self,
        values: np.ndarray,
        fractions: np.ndarray,
        oxidation_states: list[tuple[int, ...]],
    ) -> np.ndarray:
        can_form_ionic = float(self._can_form_ionic(fractions, oxidation_states))

        electronegativity = values[:, _ELEMENTAL_PROPERTIES.index("electronegativity")]
        if np.isnan(electronegativity).any():
            return np.array([can_form_ionic, np.nan, np.nan])

        # bond ionicity between all pairs of constituent elements
        diffs = electronegativity[:, None] - electronegativity[None, :]
        ionic_char = 1 - np.exp(-0.25 * diffs**2)

        max_ionic_char = ionic_char.max()
        mean_ionic_char = fractions @ ionic_char @ fractions
        return np.array([can_form_ionic, max_ionic_char, mean_ionic_char])

    def _can_form_ionic(
        self, fractions: np.ndarray, oxidation_states: list[tuple[int, ...]]
    ) -> bool:
        # check if charge-neutral ionic compound can be formed from constituent elements
        # i.e. is there a combination of common oxidation states with a fraction-weighted sum of zero?

        # single element, or one with no known oxidation states, cannot form an ionic compound
        if len(fractions) < 2 or not all(oxidation_states):
            return False

        combinations = np.array(list(itertools.product(*oxidation_states)))
        return bool(np.any(np.abs(combinations @ fractions) < 1e-6))
