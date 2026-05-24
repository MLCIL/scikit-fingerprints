import inspect
import pickle
from collections.abc import Sequence
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import Mol
from sklearn import clone
from sklearn.utils._testing import create_memmap_backed_data, set_random_state
from sklearn.utils.estimator_checks import (
    check_dont_overwrite_parameters,
    check_estimators_fit_returns_self,
    check_estimators_overwrite_params,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

import skfp.fingerprints
import skfp.preprocessing
from skfp.bases import BaseFilter
from skfp.bases.base_fp_transformer import BaseFingerprintTransformer

"""
Note: several check functions are adapted from Scikit-learn
to work with our inputs (SMILES strings or RDKit molecules).
"""

# global test data
n_samples = 10
X: Sequence[str | Mol] = []
y = None


def get_all_fingerprints() -> list[tuple[str, type]]:
    return [
        (name, obj)
        for name, obj in inspect.getmembers(skfp.fingerprints)
        if inspect.isclass(obj)
    ]


def get_all_preprocessors() -> list[tuple[str, type]]:
    return [
        (name, obj)
        for name, obj in inspect.getmembers(skfp.preprocessing)
        if inspect.isclass(obj)
    ]


@pytest.mark.parametrize("fp_name, fp_cls", get_all_fingerprints())
def test_basic_sklearn_checks_for_fingerprints(fp_name, fp_cls, mols_conformers_list):
    global X, y
    X = mols_conformers_list[:n_samples]
    y = np.arange(n_samples) % 2

    # USR and USRCAT fail for molecules with ≤3 atoms → allow NaN fallback
    fp_obj = fp_cls(errors="NaN") if "USR" in fp_name else fp_cls()
    run_checks(fp_name, fp_obj)


@pytest.mark.parametrize("preproc_name, preproc_cls", get_all_preprocessors())
def test_basic_sklearn_checks_for_preprocessors(
    preproc_name, preproc_cls, smallest_smiles_list, smallest_mols_list, fasta_list
):
    global X, y
    y = np.arange(n_samples) % 2

    # skip SDF transformers (different API)
    if preproc_name in {"MolFromSDFTransformer", "MolToSDFTransformer"}:
        return

    if preproc_name in {"MolFromSmilesTransformer", "MolFromInchiTransformer"}:
        X = smallest_smiles_list[:n_samples]
    elif preproc_name == "MolFromAminoseqTransformer":
        X = fasta_list
    else:
        X = smallest_mols_list[:n_samples]

    preproc_obj = preproc_cls()
    run_checks(preproc_name, preproc_obj)


def run_checks(fp_name: str, fp_obj: BaseFingerprintTransformer):
    checks = [
        check_no_attributes_set_in_init,
        check_fit_score_takes_y,
        check_estimators_fit_returns_self,
        check_estimators_overwrite_params,
        check_estimators_pickle,
        check_transformers_unfitted_stateless,
        check_parameters_default_constructible,
        check_get_params_invariance,
        check_set_params,
        check_dont_overwrite_parameters,
    ]
    for check in checks:
        try:
            check = partial(check, fp_name)
            check(fp_obj)
        except Exception:
            print(f"\n{fp_name} failed check {check.func.__name__}")
            raise


def check_fit_score_takes_y(name: str, estimator_orig: BaseFingerprintTransformer):
    """Check that all estimators accept optional y in fit/score, for pipeline compatibility."""
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    for func_name in ["fit", "fit_transform"]:
        func = getattr(estimator, func_name, None)
        if func is None:
            continue

        func(X, y)
        args = [p.name for p in inspect.signature(func).parameters.values()]
        if args[0] == "self":
            args = args[1:]
        if args[1] not in ["y", "Y"]:
            raise AssertionError(
                f"Expected y or Y as 2nd argument for {func_name} of {type(estimator).__name__}, "
                f"got {args}"
            )


def check_estimators_pickle(
    name: str,
    estimator_orig: BaseFingerprintTransformer,
    readonly_memmap: bool = False,
):
    """Check that all estimators can be pickled and give consistent results."""
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    # some outputs are non-deterministic or not comparable
    omit_results_check = {"GETAWAYFingerprint", "ConformerGenerator"}

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    estimator.fit(X, y)

    if readonly_memmap:
        unpickled_estimator = create_memmap_backed_data(estimator)
    else:
        pickled_estimator = pickle.dumps(estimator)
        unpickled_estimator = pickle.loads(pickled_estimator)

    results = {}
    for method in check_methods:
        if hasattr(estimator, method):
            results[method] = getattr(estimator, method)(X)

    for method, result in results.items():
        unpickled_result = getattr(unpickled_estimator, method)(X)

        if isinstance(result, list):
            assert_equal(len(result), len(unpickled_result))
        else:
            assert_equal(result.shape, unpickled_result.shape)
            if name not in omit_results_check:
                assert_allclose(result, unpickled_result, atol=1e-1, equal_nan=True)


def check_transformers_unfitted_stateless(
    name: str, transformer: BaseFingerprintTransformer
):
    """Check that transform() without fit() works for stateless transformers."""
    transformer = clone(transformer)
    X_trans = transformer.transform(X)

    if issubclass(transformer.__class__, BaseFilter):
        assert len(X_trans) <= len(X)
    else:
        assert_equal(len(X), len(X_trans))
