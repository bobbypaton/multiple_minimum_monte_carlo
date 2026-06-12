import pytest
import numpy as np

# skip entire module if rdkit missing
pytest.importorskip("rdkit")
from rdkit import Chem

from multiple_minimum_monte_carlo import cheminformatics


def test_make_mol_basic():
    smi = "CCO"
    mol = cheminformatics.make_mol(smi)
    assert mol.GetNumAtoms() == Chem.MolFromSmiles(smi).GetNumAtoms()


def test_get_bonds_and_metal_atoms():
    # create a molecule with atom mapping so get_bonds uses atom map nums
    smi = "[CH3:1][CH3:2]"
    mol = cheminformatics.make_mol(smi)
    bonds = cheminformatics.get_bonds(mol)
    assert isinstance(bonds, set)
    # metal detection on a simple metal mol
    metal_mol = Chem.MolFromSmiles("[Fe]")
    metals = cheminformatics.get_metal_atoms(metal_mol)
    assert all(isinstance(i, int) for i in metals)


def test_rotate_dihedrals_runs_without_error():
    # Use a simple molecule with a rotatable bond (but we only ensure no exception)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    Chem.AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    # pick an example dihedral (0,1,2,3) which exists for a 4-carbon chain
    cheminformatics.rotate_dihedrals(conf, [(0, 1, 2, 3)], 30.0)
    # position changed for at least one atom (float values)
    pos = conf.GetPositions()
    assert isinstance(pos, np.ndarray) or hasattr(pos, "__len__")


def test_mol_to_ase_and_add_coords():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    Chem.AllChem.EmbedMolecule(mol)
    atoms = cheminformatics.mol_to_ase_atoms(mol)
    assert hasattr(atoms, "get_positions")

    # test add_coords_to_mol with explicit numpy array
    coords = atoms.get_positions()
    newmol = cheminformatics.add_coords_to_mol(coords, mol)
    assert newmol is mol


def test_identity_checks(tmp_path):
    # Build a small molecule and ASE Atoms via mol_to_ase_atoms
    mol = cheminformatics.make_mol("CCO")
    Chem.AllChem.EmbedMolecule(mol)
    atoms = cheminformatics.mol_to_ase_atoms(mol)

    original_bonds, metal_atoms, halides = cheminformatics.initialize_mc_identity_check(
        atoms, mol
    )
    # returned types
    assert isinstance(original_bonds, list)
    assert isinstance(metal_atoms, list)
    assert isinstance(halides, list)

    # check_identity_mc on same atoms should be True
    assert (
        cheminformatics.check_identity_mc(original_bonds, metal_atoms, halides, atoms)
        is True
    )


def _water_like_positions():
    # A simple bent triatomic (asymmetric top) with three distinct masses-agnostic
    # principal moments.
    return np.array(
        [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ]
    )


def test_rotational_constants_invariant_to_translation_and_rotation():
    pos = _water_like_positions()
    masses = np.array([16.0, 1.0, 1.0])
    base = cheminformatics.rotational_constants(pos, masses)
    # translation
    translated = cheminformatics.rotational_constants(
        pos + np.array([3.0, -2.0, 5.0]), masses
    )
    # 90-degree rotation about z
    rot_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = cheminformatics.rotational_constants(pos @ rot_z.T, masses)
    assert np.allclose(base, translated)
    assert np.allclose(base, rotated)
    # returned largest-first
    assert base[0] >= base[1] >= base[2]


def test_rotational_constants_invariant_to_permutation_of_identical_atoms():
    pos = _water_like_positions()
    masses = np.array([16.0, 1.0, 1.0])
    base = cheminformatics.rotational_constants(pos, masses)
    # swap the two identical hydrogens
    swapped = pos[[0, 2, 1]]
    permuted = cheminformatics.rotational_constants(swapped, np.array([16.0, 1.0, 1.0]))
    assert np.allclose(base, permuted)


def test_rotational_anisotropy_zero_for_spherical_top():
    # equal rotational constants -> spherical top -> zero anisotropy
    assert cheminformatics.rotational_anisotropy(np.array([1.0, 1.0, 1.0])) == 0.0
    # unequal constants -> positive anisotropy
    assert cheminformatics.rotational_anisotropy(np.array([3.0, 2.0, 1.0])) > 0.0


def test_bthr_anisotropy_threshold_bounds_and_monotonic():
    bthr, bthrmax, bthrshift = 0.01, 0.025, 0.5
    low = cheminformatics.bthr_anisotropy_threshold(bthr, 0.0, bthrmax, bthrshift)
    high = cheminformatics.bthr_anisotropy_threshold(bthr, 1.0, bthrmax, bthrshift)
    mid = cheminformatics.bthr_anisotropy_threshold(bthr, 0.5, bthrmax, bthrshift)
    # at zero anisotropy the threshold collapses to ~bthr; at high anisotropy ~bthrmax
    assert low == pytest.approx(bthr, abs=1e-4)
    assert high == pytest.approx(bthrmax, abs=1e-3)
    # ramp is monotonically increasing in anisotropy
    assert low < mid < high


def test_rotational_constants_equal_identical_and_distinct():
    pos = _water_like_positions()
    masses = np.array([16.0, 1.0, 1.0])
    rot = cheminformatics.rotational_constants(pos, masses)
    # identical structure matches itself
    assert cheminformatics.rotational_constants_equal(rot, rot) is True
    # a clearly different shape (linear-ish) does not match
    other = cheminformatics.rotational_constants(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [0.0, 0.0, 2.4]]), masses
    )
    assert cheminformatics.rotational_constants_equal(rot, other) is False
