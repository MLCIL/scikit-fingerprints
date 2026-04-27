import importlib
from collections import deque

import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["fMF"]
Node = tuple[str, int]

SMILES_EXPECTED = [
    ("CC", 0.0),
    ("CCO", 0.0),
    ("c1ccccc1", 1.0),
    ("C1CCCCC1", 1.0),
    ("c1ccc(CCc2ccccc2)cc1", 1.0),
    ("c1ccc2ccccc2c1", 1.0),
    ("c1ccccc1.CC", 0.75),
]


@pytest.fixture(scope="module")
def mordred_framework_calc():
    return Calculator(descriptors.Framework, ignore_3D=True)


def _shortest_path(
    graph: dict[Node, list[Node]], source: Node, target: Node
) -> list[Node]:
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        node, path = queue.popleft()
        if node == target:
            return path

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))

    return []


def _expected_no_h_framework(mol):
    """Compute fMF with Mordred Framework graph semantics and no explicit Hs."""
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return np.nan

    rings = [frozenset(ring) for ring in Chem.GetSymmSSSR(mol)]
    ring_by_atom = {
        atom_idx: ("R", ring_idx)
        for ring_idx, ring in enumerate(rings)
        for atom_idx in ring
    }
    ring_nodes = list(set(ring_by_atom.values()))

    graph: dict[Node, list[Node]] = {}
    for bond in mol.GetBonds():
        begin = ring_by_atom.get(bond.GetBeginAtomIdx(), ("A", bond.GetBeginAtomIdx()))
        end = ring_by_atom.get(bond.GetEndAtomIdx(), ("A", bond.GetEndAtomIdx()))
        graph.setdefault(begin, []).append(end)
        graph.setdefault(end, []).append(begin)

    linkers: set[int] = set()
    for i, source in enumerate(ring_nodes):
        for target in ring_nodes[i + 1 :]:
            path = _shortest_path(graph, source, target)
            linkers.update(atom_idx for node_type, atom_idx in path if node_type == "A")

    ring_atoms = {atom_idx for ring in rings for atom_idx in ring}
    return (len(linkers) + len(ring_atoms)) / n_atoms


@pytest.mark.parametrize(("smiles", "expected"), SMILES_EXPECTED)
def test_framework_calc_uses_no_h_expected_formula(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    framework = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.framework"
    )

    values, feature_names = framework.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, [expected], rtol=1e-6, atol=1e-6)
    assert_allclose(values, [_expected_no_h_framework(mol)], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("smiles", "expected"), SMILES_EXPECTED)
def test_calculator_fills_framework_column_with_no_h_formula(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idx = FEATURE_NAMES_2D.index("fMF")

    assert observed.dtype == np.float32
    assert_allclose(observed[idx], expected, rtol=1e-6, atol=1e-6)
    assert_allclose(observed[idx], _expected_no_h_framework(mol), rtol=1e-6, atol=1e-6)


def test_framework_empty_molecule_returns_nan():
    cache = MordredMolCache.from_mol(Chem.Mol(), use_3D=False)
    framework = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.framework"
    )

    values, feature_names = framework.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert np.isnan(values[0])


def test_default_mordred_uses_explicit_hydrogen_denominator(mordred_framework_calc):
    mol = Chem.MolFromSmiles("c1ccccc1")
    mordred_value = np.asarray(mordred_framework_calc(mol), dtype=np.float32)[0]

    assert_allclose(_expected_no_h_framework(mol), 1.0, rtol=1e-6, atol=1e-6)
    assert_allclose(mordred_value, 0.5, rtol=1e-6, atol=1e-6)
    assert mordred_value != _expected_no_h_framework(mol)
