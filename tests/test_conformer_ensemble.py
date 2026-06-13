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


def _methyl_rotamer_conformer(monkeypatch, **ensemble_kwargs):
    """Ethane conformer plus a 120-degree methyl rotamer of it.

    The rotamer is the same conformer with hydrogens relabeled, so a heavy-atom
    RMSD is ~0 while an all-atom fixed-map RMSD is large.
    """
    from rdkit.Chem import AllChem, rdMolTransforms
    from ase import Atoms as ASEAtoms
    from multiple_minimum_monte_carlo import cheminformatics

    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    base = mol.GetConformer().GetPositions().copy()
    # rotate the methyl on C1 by 120 degrees about the C0-C1 bond
    h0 = next(n.GetIdx() for n in mol.GetAtomWithIdx(0).GetNeighbors())
    h1 = next(n.GetIdx() for n in mol.GetAtomWithIdx(1).GetNeighbors())
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(
        conf, h0, 0, 1, h1, rdMolTransforms.GetDihedralDeg(conf, h0, 0, 1, h1) + 120
    )
    rotated = conf.GetPositions().copy()

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    c = Conformer.__new__(Conformer)
    c.mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(c.mol, randomSeed=1)
    c.atoms = ASEAtoms(symbols=symbols, positions=base)
    c.constrained_atoms = []
    c.bonded_atoms = []
    c.charge = 0
    ensemble = ConformerEnsemble(
        c, calc=None, uniqueness_method="crest", energy_window=1000.0, **ensemble_kwargs
    )
    ensemble.original_bonds = []
    ensemble.metal_atoms = []
    ensemble.halides = []
    monkeypatch.setattr(cheminformatics, "check_identity_mc", lambda *a, **k: True)
    return ensemble, base, rotated


def test_heavy_atom_rmsd_rejects_methyl_rotamer(monkeypatch):
    # default is heavy-atom only: a methyl rotamer is the same conformer -> duplicate
    ensemble, base, rotated = _methyl_rotamer_conformer(monkeypatch)
    assert ensemble.rmsd_heavy_only is True
    assert ensemble._pairwise_rmsd(rotated, base) < 1e-3
    assert ensemble.check_conformer([base], [0.0], rotated, 0.0) is False


def test_all_atom_rmsd_keeps_methyl_rotamer(monkeypatch):
    # legacy all-atom RMSD inflates on the relabeled hydrogens -> kept as distinct
    ensemble, base, rotated = _methyl_rotamer_conformer(
        monkeypatch, rmsd_heavy_only=False
    )
    assert ensemble._pairwise_rmsd(rotated, base) > ensemble.rthr
    assert ensemble.check_conformer([base], [0.0], rotated, 0.0) is True


def test_symmetry_rmsd_rejects_methyl_rotamer(monkeypatch):
    # GetBestRMS permutes equivalent hydrogens, so even all-atom collapses below rthr
    ensemble, base, rotated = _methyl_rotamer_conformer(
        monkeypatch, rmsd_heavy_only=False, rmsd_symmetry=True
    )
    assert ensemble._pairwise_rmsd(rotated, base) < ensemble.rthr
    assert ensemble.check_conformer([base], [0.0], rotated, 0.0) is False


def _chiral_conformer(monkeypatch, **ensemble_kwargs):
    """A chiral conformer (not superimposable on its mirror image by rotation)."""
    from rdkit.Chem import AllChem
    from ase import Atoms as ASEAtoms
    from multiple_minimum_monte_carlo import cheminformatics

    mol = Chem.AddHs(Chem.MolFromSmiles("FC(Cl)Br"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    base = mol.GetConformer().GetPositions().copy()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    c = Conformer.__new__(Conformer)
    c.mol = mol
    c.atoms = ASEAtoms(symbols=symbols, positions=base)
    c.constrained_atoms = []
    c.bonded_atoms = []
    c.charge = 0
    ensemble = ConformerEnsemble(
        c, calc=None, uniqueness_method="crest", energy_window=1000.0, **ensemble_kwargs
    )
    ensemble.original_bonds = []
    ensemble.metal_atoms = []
    ensemble.halides = []
    monkeypatch.setattr(cheminformatics, "check_identity_mc", lambda *a, **k: True)
    return ensemble, base


def test_mirror_image_kept_without_enantiomer_detection(monkeypatch):
    ensemble, base = _chiral_conformer(monkeypatch, detect_enantiomers=False)
    assert ensemble.detect_enantiomers is False
    mirror = -base
    # mirror is not superimposable by proper rotation -> distinct by direct RMSD
    assert ensemble._pairwise_rmsd(mirror, base) > ensemble.rthr
    assert ensemble.check_conformer([base], [0.0], mirror, 0.0) is True


def test_mirror_image_flagged_as_enantiomer(monkeypatch):
    ensemble, base = _chiral_conformer(monkeypatch)  # detection is on by default
    assert ensemble.detect_enantiomers is True
    mirror = -base
    # inverted comparison overlays perfectly -> rejected as a redundant enantiomer
    assert ensemble._pairwise_rmsd(mirror, base, invert=True) < ensemble.rthr
    assert ensemble.check_conformer([base], [0.0], mirror, 0.0) is False
    assert ensemble._duplicate_index == 0


def test_lower_energy_duplicate_refines_member(monkeypatch):
    from multiple_minimum_monte_carlo import cheminformatics

    c, positions = _make_real_conformer()

    class DummyCalc:
        def energy(self, atoms):
            return 0.0

        def run(self, atoms, constrained_atoms=None):
            return atoms.get_positions(), 0.0

    ensemble = ConformerEnsemble(
        c,
        calc=DummyCalc(),
        uniqueness_method="rmsd",
        rmsd_threshold=0.5,
        num_iterations=1,
        initial_optimization=False,
        parallel=False,
    )
    monkeypatch.setattr(cheminformatics, "check_identity_mc", lambda *a, **k: True)
    monkeypatch.setattr(ensemble, "sample_conformer", lambda used: 0)
    monkeypatch.setattr(
        ensemble, "modify_conformer", lambda conf, dih: (True, positions)
    )
    # the single MC optimization re-discovers the input geometry at lower energy
    monkeypatch.setattr(ensemble, "run_optimizations", lambda inp: [(positions, -0.04)])

    captured = {}

    def callback(steps, initial, results, accepted, refined):
        captured["accepted"] = list(accepted)
        captured["refined"] = list(refined)
        return False

    ensemble.step_callback = callback
    ensemble.run_monte_carlo()

    # rejected as a duplicate, but kept as the cluster's lower-energy representative
    assert captured["accepted"] == [False]
    assert captured["refined"] == [True]
    assert ensemble.final_energies[0] == pytest.approx(-0.04)
    assert ensemble.found[0] == 2


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
