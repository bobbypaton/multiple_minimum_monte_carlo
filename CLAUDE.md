# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Open-source implementation of the Multiple Minimum Monte Carlo (MMMC) conformer search algorithm (Chang, Guida & Still, 1989). Published to PyPI as `multiple-minimum-monte-carlo`. The package generates molecular conformer ensembles by iteratively rotating random dihedral angles, optimizing the resulting structures with an ASE-compatible calculator, and filtering by energy window and RMSD uniqueness.

## Commands

This project uses `uv` for dependency management (see `uv.lock`).

```bash
uv run pytest -q                          # run all tests (what CI runs)
uv run pytest tests/test_conformer.py     # run one test file
uv run pytest tests/test_conformer.py::test_name   # run a single test
uv run ruff check .                       # lint
uv run ruff format .                      # format
python -m build                           # build distribution
```

Tests use lightweight dummy calculators/atoms (see `tests/test_calculation.py`) so they don't require ML potentials or GPU; heavy backends (aimnet, torch-sim/fairchem) are intentionally not dependencies and are only needed by end users at runtime. CI runs tests on Python 3.8–3.14.

## Architecture

Three objects compose a search; the user wires them together (see README tutorial):

1. **`Conformer`** (`conformer.py`) — holds the molecule as both an RDKit `Mol` (for connectivity/dihedral manipulation) and an ASE `Atoms` (for optimization). Built from a SMILES string (3D coords generated via ETKDG + UFF), an XYZ file (bonds inferred via RDKit `DetermineBonds`, requires `charge`), or both (atom-mapped SMILES with `mapped=True` when RDKit can't infer bonding, e.g. transition states). Carries charge, spin multiplicity, and `constrained_atoms`.

2. **A calculation object** — two parallel interfaces:
   - `Calculation` (`calculation.py`): one-structure-at-a-time interface — `run(atoms, constrained_atoms) -> (positions, energy)` and `energy(atoms) -> float`. `ASEOptimization` is the provided implementation wrapping any ASE calculator + optimizer. Users can supply their own class with the same three methods (`__init__`, `run`, `energy`).
   - `BatchCalculation` (`batch_calculation.py`): list-in/list-out interface for optimizing many conformers simultaneously. `TorchSimCalculation` is the provided implementation (GPU-accelerated via torch-sim, imported lazily).
   - **Energies are always kcal/mol** (converted from eV via `EV_TO_KCAL = 23.0605`); positions are numpy arrays in Angstroms. Any new calculation backend must respect these units.

3. **`ConformerEnsemble`** (`conformer_ensemble.py`) — the MMMC driver. `run_monte_carlo()` loops: sample a conformer from the ensemble (least-used by default, random with `random_walk=True`), rotate 1–`max_bonds_rotate` random dihedrals by multiples of `angle_step`, reject rotations placing non-bonded atoms closer than ¼ summed vdW radii (`constraint_test`), optimize, then accept only structures that pass an identity check (no bond changes vs. the input; see `cheminformatics.check_identity_mc`), fall within `energy_window` of the minimum, and differ by more than `rmsd_threshold` from every existing member. Results land in `final_ensemble` / `final_energies`, sorted lowest-energy first.

Execution modes are selected automatically/by flag in `ConformerEnsemble`:
- **Serial** (default): one optimization per iteration.
- **Parallel** (`parallel=True`, only with `Calculation`): `num_cpus` optimizations per loop via `multiproc.parallel_run_proc`, which uses torch multiprocessing with each worker running in its own temp batch directory (cleaned up afterward) and a `process_timeout` watchdog. Requires the "fork" start method.
- **Batched** (automatic when `calc` is a `BatchCalculation`): `batch_size` trial structures optimized in one call. Parallel + batch together is rejected (falls back with a warning).

`cheminformatics.py` is the shared utility layer: RDKit↔ASE conversion (atom ordering between `mol` and `atoms` must stay consistent — this is why atom-mapped SMILES matter), dihedral detection via SMARTS, dihedral rotation, and the bond-identity check (metals and halides get special handling there).

## Conventions

- Dihedral/constraint atom indices are 0-based throughout.
- `examples/run_monte_carlo.ipynb` is the runnable end-to-end example.
