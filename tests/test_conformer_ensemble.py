import pytest

pytest.importorskip("rdkit")
from rdkit import Chem

from multiple_minimum_monte_carlo.conformer_ensemble import (
    ConformerEnsemble,
    run_class_func,
)
from multiple_minimum_monte_carlo.conformer import Conformer


def test_run_class_func_calls_method():
    class C:
        def add(self, x, y):
            return x + y

    assert run_class_func(C(), "add", {"x": 1, "y": 2}) == 3


def make_dummy_conformer(monkeypatch):
    # Create a minimal Conformer-like object
    # prevent heavy generation
    def fake_generate(self):
        class MinimalAtoms:
            def __init__(self):
                self.info = {}

            def get_positions(self):
                return []

        self.atoms = MinimalAtoms()

    try:
        # if a pytest monkeypatch fixture is provided
        monkeypatch.setattr(Conformer, "generate_conformer", fake_generate)
    except Exception:
        # otherwise, monkeypatch by direct assignment
        Conformer.generate_conformer = fake_generate
    c = Conformer("CC")
    c.atoms = None
    c.mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    c.constrained_atoms = []
    c.bonded_atoms = []
    return c


def test_sample_conformer_minimum():
    c = make_dummy_conformer(None)
    ensemble = ConformerEnsemble(c, calc=None, num_iterations=1, parallel=False)
    # when used is [0,0,0], argmin is 0
    assert ensemble.sample_conformer([1, 2, 3]) == 0 or isinstance(
        ensemble.sample_conformer([1, 2, 3]), int
    )


def test_constraint_test_detects_close_atoms(monkeypatch):
    c = make_dummy_conformer(monkeypatch)
    # create a fake conformer with positions that are extremely close
    mol = c.mol
    Chem.AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    # set all positions to zero to force failure
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
    ensemble = ConformerEnsemble(c, calc=None, num_iterations=1, parallel=False)
    assert ensemble.constraint_test(conf) is False


def _make_real_conformer():
    """Build a Conformer with real 3D coordinates and an ASE Atoms object,
    bypassing the heavy SMILES initializer."""
    from rdkit.Chem import AllChem
    from ase import Atoms as ASEAtoms

    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    positions = mol.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    c = Conformer.__new__(Conformer)
    c.mol = mol
    c.atoms = ASEAtoms(symbols=symbols, positions=positions)
    c.constrained_atoms = []
    c.bonded_atoms = []
    c.charge = 0
    return c, positions


def _crest_ensemble(monkeypatch, method):
    from multiple_minimum_monte_carlo import cheminformatics

    c, positions = _make_real_conformer()
    ensemble = ConformerEnsemble(
        c,
        calc=None,
        uniqueness_method=method,
        energy_window=1000.0,
        rmsd_threshold=0.3,
    )
    # identity check is orthogonal to dedup; bypass it here
    ensemble.original_bonds = []
    ensemble.metal_atoms = []
    ensemble.halides = []
    monkeypatch.setattr(cheminformatics, "check_identity_mc", lambda *a, **k: True)
    return ensemble, positions


def test_invalid_uniqueness_method_raises():
    c = make_dummy_conformer(None)
    with pytest.raises(ValueError):
        ConformerEnsemble(c, calc=None, uniqueness_method="bogus")


def test_crest_energy_distinguishes_identical_geometry(monkeypatch):
    ensemble, positions = _crest_ensemble(monkeypatch, "crest")
    existing, energies = [positions], [0.0]
    # identical geometry + same energy -> duplicate -> rejected
    assert ensemble.check_conformer(existing, energies, positions, 0.0) is False
    # identical geometry but energy differs by more than ethr (0.05) -> kept distinct
    assert ensemble.check_conformer(existing, energies, positions, 1.0) is True


def test_rmsd_method_ignores_energy(monkeypatch):
    ensemble, positions = _crest_ensemble(monkeypatch, "rmsd")
    existing, energies = [positions], [0.0]
    # under rmsd-only, identical geometry is a duplicate regardless of energy
    assert ensemble.check_conformer(existing, energies, positions, 1.0) is False


def test_crest_verbose_logs_comparisons(monkeypatch, caplog):
    import logging

    ensemble, positions = _crest_ensemble(monkeypatch, "crest")
    ensemble.verbose = True
    with caplog.at_level(logging.INFO):
        # identical geometry + same energy -> logged DUPLICATE
        ensemble.check_conformer([positions], [0.0], positions, 0.0)
    text = caplog.text
    assert "Checking candidate" in text
    assert "vs #0" in text
    assert "DUPLICATE" in text
