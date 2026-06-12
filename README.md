# Multiple Minimum Monte Carlo
[![pypi](https://img.shields.io/pypi/v/multiple-minimum-monte-carlo.svg)](https://pypi.python.org/pypi/multiple-minimum-monte-carlo)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

This package will help you perform a multiple minumum Monte Carlo conformer search as described in [Chang et al., 1989](https://doi.org/10.1021/ja00194a035). It is built to be used with an ASE calculator and ASE optimization tools but user-defined optimization strategies can be employed as well.

### Installation

This package can be installed with pip

```bash
pip install multiple-minimum-monte-carlo
```

### Tutorial

To run a search, you need to initialize Conformer, Calculation, and ConformerEnsemble objects. Conformer objects require either an input xyz or SMILES string. The default Calculation object is ASEOptimization which requires an ASE optimization routine (like FIRE) and an ASE calculator (the example below uses the aimnet calculator which will need to be installed separately from this package). ConformerEnsemble objects require a Conformer and Calculation object.

```python
from ase.optimize.fire import FIRE
from ase.io import write
from aimnet.calculators import AIMNet2ASE
from multiple_minimum_monte_carlo.conformer import Conformer
from multiple_minimum_monte_carlo.calculation import ASEOptimization
from multiple_minimum_monte_carlo.conformer_ensemble import ConformerEnsemble

smiles = "CC(=O)Oc1ccccc1C(=O)O"
conformer = Conformer(smiles=smiles)
optimizer = ASEOptimization(calc=AIMNet2ASE(), optimizer=FIRE)
conformer_ensemble = ConformerEnsemble(conformer=conformer, calc=optimizer)
```

To run the search, call run_monte_carlo with the ConformerEnsemble object

```python
conformer_ensemble.run_monte_carlo()
```

final_ensemble will be a list of coordinate arrays that arranged by their energy (lowest energy first). To read out the minimum energy compound, do this

```python
from ase.io import write
conformer.atoms.set_positions(conformer_ensemble.final_ensemble[0])
write("lowest_energy_conformer.xyz", conformer.atoms, format="xyz")
```

To perform batched calculations (which will perform multiple Monte Carlo steps at once and optimize all of the sampled conformers simultaneously), we will need to use a batched optimizer. In the example below, we will use the torchsim batched calculator with the uma-s-1 MLIP (which both need to be installed separately from this package)

```python
from multiple_minimum_monte_carlo.batch_calculation import TorchSimCalculation
from torch_sim.models.fairchem import FairChemModel
from torch_sim.optimizers import Optimizer

model = FairChemModel(model=None, model_name="uma-s-1",task_name="omol", cpu=True)
calc = TorchSimCalculation(model=model, optimizer=Optimizer.fire, max_cycles=500)
conformer_ensemble = ConformerEnsemble(conformer=conformer, calc=calc)
```

### Using the xtb command-line program

If you have the standalone [xtb](https://github.com/grimme-lab/xtb) executable installed (e.g. `conda install -c conda-forge xtb`), XTBCalculation will run optimizations by shelling out to it directly — no Python bindings (tblite) required. This uses xtb's native optimizer, is often faster than driving GFN2-xTB through an ASE optimizer, and works well on macOS.

```python
from multiple_minimum_monte_carlo.calculation import XTBCalculation

calc = XTBCalculation(charge=0, spin_multiplicity=1)
conformer_ensemble = ConformerEnsemble(conformer=conformer, calc=calc)
```

Optional arguments include `method` ("gfn0", "gfn1", "gfn2", or "gfnff"), `opt_level` (xtb convergence level, e.g. "tight"), and `solvent` (ALPB implicit solvent, e.g. "water"). From the command line, the same backend is available as `mcmm input.xyz --model xtb-cli`.

### Command-line interface

The package installs an `mcmm` command for running a search directly on an XYZ file:

```bash
mcmm input.xyz --steps 100 --model aimnet2
```

This optimizes the input structure, runs the Monte Carlo search, writes the final ensemble to `input_mcmm.xyz` (ordered lowest energy first, with each frame's energy on its comment line), and prints a formatted summary of the unique conformers:

```text
────────────────────── Final Ensemble Information ──────────────────────
output file name               : input_mcmm.xyz
conformer energy window  /kcal :  10.0000
total number unique conformers :        3
lowest energy conformer    /Eh : -649.151439

 #   Erel/kcal      Etot/Eh   weight/tot   found   origin
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1       0.000   -649.15144      0.47811       1   input
 2       0.088   -649.15130      0.41204       1     mc
 3       0.871   -649.15005      0.10985       2     mc
```

`Erel`/`Etot` are the relative and total energies, `weight/tot` is the Boltzmann population at 298 K, `found` is how many times each unique conformer was located during the search, and `origin` marks the starting structure (`input`) versus Monte-Carlo-discovered conformers (`mc`). Run `mcmm --help` for the full list of options (energy window, RMSD threshold, model and optimizer choice, parallel/batched execution, and more).

### Conformer deduplication

By default a new conformer is considered unique when its RMSD to every existing member exceeds `rmsd_threshold`. Passing `uniqueness_method="crest"` to `ConformerEnsemble` (or `--uniqueness-method crest` on the command line) switches to a CREST/CREGEN-style criterion: a conformer is treated as a duplicate only when it matches an existing member in energy, rotational constants, and RMSD simultaneously, which is more robust to symmetry-equivalent atom permutations than RMSD alone.

### A note about parallel calculations

As opposed to batched calculations, you can also do calculations in parallel with a Calculation object by setting parallel=True in the ConformerEnsemble object. However, this requires that the multiprocessing start method "fork" is used which may be incompatible with certain workflows.

### A note about just using input xyz structure

When you only include an input xyz structure, this code uses rdkit to construct the SMILES string. However, occassionally this will fail (i.e. when the structure is more ambiguous like a transition state). In this case, including the SMILES string will fix this issue, but the SMILES string needs to be mapped to the structure. The easiest way to do this is to use an atom-mapped SMILES string (the hydrogens also need to be mapped!) and set mapped=True in your Conformer object

### User-Defined Calculation

To define a Calculation object, a class will need three function: init, run, and energy. init initializes the class with whatever information is necessary. run performs an optimization. It takes an ase.Atoms object and a list of atoms to constrain and returns an np array of cartesian coordinates (in angstroms) and a float with the energy of the conformation (in kcal/mol). energy calculates the energy of a conformer. It takes an ase.Atoms object and returns a float the with energy (in kcal/mol)