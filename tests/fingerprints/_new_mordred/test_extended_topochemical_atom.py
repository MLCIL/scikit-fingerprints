import importlib

import numpy as np
import pytest
from mordred import Calculator, ExtendedTopochemicalAtom
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import GetMolFrags

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.atomic_properties import get_eta_epsilon
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = [
    "ETA_alpha",
    "AETA_alpha",
    "ETA_shape_p",
    "ETA_shape_y",
    "ETA_shape_x",
    "ETA_beta",
    "AETA_beta",
    "ETA_beta_s",
    "AETA_beta_s",
    "ETA_beta_ns",
    "AETA_beta_ns",
    "ETA_beta_ns_d",
    "AETA_beta_ns_d",
    "ETA_eta",
    "AETA_eta",
    "ETA_eta_L",
    "AETA_eta_L",
    "ETA_eta_R",
    "AETA_eta_R",
    "ETA_eta_RL",
    "AETA_eta_RL",
    "ETA_eta_F",
    "AETA_eta_F",
    "ETA_eta_FL",
    "AETA_eta_FL",
    "ETA_eta_B",
    "AETA_eta_B",
    "ETA_eta_BR",
    "AETA_eta_BR",
    "ETA_dAlpha_A",
    "ETA_dAlpha_B",
    "ETA_epsilon_1",
    "ETA_epsilon_2",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
    "ETA_dBeta",
    "AETA_dBeta",
    "ETA_psi_1",
    "ETA_dPsi_A",
    "ETA_dPsi_B",
]

NO_EXPLICIT_H_ETA_FEATURES = {
    "ETA_epsilon_1",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
}

SMILES = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "c1ccccc1",
    "c1ccncc1",
    "c1ccccc1O",
    "CC(=O)O",
    "CCl",
    "C1CCCCC1",
    "P(Cl)(Cl)(Cl)(Cl)Cl",
    "C.C",
]


@pytest.fixture(scope="module")
def mordred_eta_calc():
    return Calculator(ExtendedTopochemicalAtom, ignore_3D=True)


def _mordred_expected(mol, mordred_eta_calc, feature_names):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_eta_calc.descriptors),
            mordred_eta_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [
            np.nan
            if isinstance(mordred_values[name], Missing)
            else mordred_values[name]
            for name in feature_names
        ],
        dtype=np.float32,
    )


def _alter_mol_without_hydrogens(mol, saturated=False):
    new = Chem.RWMol(Chem.Mol())
    atom_idxs = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        if saturated:
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
        else:
            new_atom = Chem.Atom(6)
        atom_idxs[atom.GetIdx()] = new.AddAtom(new_atom)

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if not saturated and (begin.GetDegree() > 4 or end.GetDegree() > 4):
            return None

        begin_idx = atom_idxs.get(begin.GetIdx())
        end_idx = atom_idxs.get(end.GetIdx())
        if begin_idx is None or end_idx is None:
            continue

        if saturated and (begin.GetAtomicNum() != 6 or end.GetAtomicNum() != 6):
            bond_type = bond.GetBondType()
        else:
            bond_type = Chem.BondType.SINGLE
        new.AddBond(begin_idx, end_idx, bond_type)

    new_mol = Chem.Mol(new)
    if Chem.SanitizeMol(new_mol, catchErrors=True) != 0:
        return None

    Chem.Kekulize(new_mol)
    return new_mol


def _mean_eta_epsilon(mol):
    if mol is None or mol.GetNumAtoms() == 0:
        return np.nan
    return sum(get_eta_epsilon(atom) for atom in mol.GetAtoms()) / mol.GetNumAtoms()


def _no_h_epsilon_expected(mol):
    reference_mol = _alter_mol_without_hydrogens(mol)
    saturated_mol = _alter_mol_without_hydrogens(mol, saturated=True)

    # The new 2D policy suppresses explicit hydrogens for every ETA descriptor.
    # With no explicit H atoms, epsilon_1, epsilon_2, and epsilon_5 all reduce
    # to the heavy-atom epsilon mean.
    epsilon_1 = _mean_eta_epsilon(mol)
    epsilon_2 = epsilon_1
    epsilon_3 = _mean_eta_epsilon(reference_mol)
    epsilon_4 = _mean_eta_epsilon(saturated_mol)
    epsilon_5 = epsilon_2

    return {
        "ETA_epsilon_1": epsilon_1,
        "ETA_epsilon_3": epsilon_3,
        "ETA_epsilon_4": epsilon_4,
        "ETA_epsilon_5": epsilon_5,
        "ETA_dEpsilon_A": epsilon_1 - epsilon_3,
        "ETA_dEpsilon_B": epsilon_1 - epsilon_4,
        "ETA_dEpsilon_C": epsilon_3 - epsilon_4,
        "ETA_dEpsilon_D": epsilon_2 - epsilon_5,
    }


def _expected_with_no_h_policy(mol, mordred_eta_calc, feature_names):
    expected = _mordred_expected(mol, mordred_eta_calc, feature_names)
    if len(GetMolFrags(mol)) != 1:
        return np.full(len(feature_names), np.nan, dtype=np.float32)

    no_h_expected = _no_h_epsilon_expected(preprocess_mol(mol, kekulize=True))
    for i, name in enumerate(feature_names):
        if name in NO_EXPLICIT_H_ETA_FEATURES:
            expected[i] = no_h_expected[name]
    return expected


def test_extended_topochemical_atom_feature_names_are_in_mordred_order():
    eta = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom"
    )

    assert eta.FEATURE_NAMES == FEATURE_NAMES
    assert len(eta.FEATURE_NAMES) == 45


@pytest.mark.parametrize("smiles", SMILES)
def test_extended_topochemical_atom_matches_mordred_with_no_h_policy(
    smiles, mordred_eta_calc
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eta = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom"
    )

    values, feature_names = eta.calc(cache)
    expected = _expected_with_no_h_policy(mol, mordred_eta_calc, feature_names)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_extended_topochemical_atom_columns(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eta = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom"
    )

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected, _ = eta.calc(cache)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_disconnected_molecules_return_nan():
    mol = Chem.MolFromSmiles("C.C")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eta = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom"
    )

    values, feature_names = eta.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert np.isnan(values).all()


@pytest.mark.parametrize("smiles", ["[H]", "[He]"])
def test_single_atom_edge_cases_do_not_raise(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eta = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom"
    )

    values, feature_names = eta.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert values.shape == (len(FEATURE_NAMES),)


def test_reference_molecule_failure_is_descriptor_local():
    mol = Chem.MolFromSmiles("P(Cl)(Cl)(Cl)(Cl)Cl")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    eta = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom"
    )

    values, feature_names = eta.calc(cache)
    value_by_name = dict(zip(feature_names, values, strict=True))

    assert not np.isnan(value_by_name["ETA_alpha"])
    assert not np.isnan(value_by_name["ETA_beta"])
    assert not np.isnan(value_by_name["ETA_epsilon_1"])
    assert np.isnan(value_by_name["ETA_eta_R"])
    assert np.isnan(value_by_name["ETA_dAlpha_A"])
