# fingerprint classes are obtained dynamically via inspect, so mypy sees only "object"
# mypy: disable-error-code="call-arg,attr-defined"
import inspect

import pandas as pd
import sklearn

import skfp.fingerprints


def test_individual_set_output(mols_conformers_list):
    mols_conformers_list = mols_conformers_list[:10]

    for name, obj in inspect.getmembers(skfp.fingerprints):
        if not inspect.isclass(obj):
            continue

        # USR and USRCAT don't work for molecules with 3 or fewer atoms, so we use
        # NaNs there
        if "USR" in name:
            fp = obj(errors="NaN")  # type: ignore
        else:
            fp = obj()

        fp.set_output(transform="pandas")  # type: ignore
        output = fp.transform(mols_conformers_list)  # type: ignore
        if not isinstance(output, pd.DataFrame):
            raise AssertionError(
                f"For fingerprint {name} output was of type {type(output)} "
                f"instead of pd.DataFrame"
            )


def test_global_set_output(mols_conformers_list):
    mols_conformers_list = mols_conformers_list[:10]

    with sklearn.config_context(transform_output="pandas"):
        # loop through all fingerprint classes
        for name, obj in inspect.getmembers(skfp.fingerprints):
            if not inspect.isclass(obj):
                continue

            # USR and USRCAT don't work for molecules with 3 or fewer atoms, so we use
            # NaNs there
            if "USR" in name:
                fp = obj(errors="NaN")  # type: ignore
            else:
                fp = obj()

            output = fp.transform(mols_conformers_list)  # type: ignore
            if not isinstance(output, pd.DataFrame):
                raise AssertionError(
                    f"For fingerprint {name} output was of type {type(output)} "
                    f"instead of pd.DataFrame"
                )
