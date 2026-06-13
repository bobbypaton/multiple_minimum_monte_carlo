"""Module for running geometry optimizations.

This module provides classes for performing geometry optimizations on molecular
structures using ASE (Atomic Simulation Environment) calculators.
"""

import os
import re
import contextlib
import subprocess
import tempfile
from typing import Optional, List, Tuple
import numpy as np
import ase
import ase.calculators.calculator
from ase.optimize import BFGS
import ase.optimize.optimize
from ase.constraints import FixAtoms

EV_TO_KCAL = 23.0605
"""float: Conversion factor from electron volts to kilocalories per mole."""

HARTREE_TO_KCAL = 627.5094740631
"""float: Conversion factor from hartree to kilocalories per mole."""


class Calculation:
    """Abstract base class for molecular calculations.

    This class defines the interface for performing energy calculations and
    geometry optimizations on molecular structures.
    """

    def __init__(self):
        """Initialize the calculation object."""
        pass

    def run(
        self, atoms: ase.Atoms, constrained_atoms: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        """Run a geometry optimization on the given atoms.

        Args:
            atoms: ASE Atoms object representing the molecule.
            constrained_atoms: List of atom indices to constrain during optimization.

        Returns:
            Tuple containing the optimized positions (np.ndarray) and energy (float).
        """
        pass

    def energy(self, atoms: ase.Atoms) -> float:
        """Calculate the energy of the given atoms.

        Args:
            atoms: ASE Atoms object representing the molecule.

        Returns:
            Energy in kcal/mol.
        """
        pass


class ASEOptimization(Calculation):
    """Geometry optimization using ASE calculators.

    This class wraps ASE's optimization routines to perform geometry optimizations
    with configurable calculator, optimizer, convergence criteria, and constraints.

    Attributes:
        calc: ASE calculator for computing energies and forces.
        optimizer: ASE optimizer class to use for optimization.
        fmax: Maximum force convergence criterion in eV/Angstrom.
        max_cycles: Maximum number of optimization steps.
        verbose: Whether to print optimization progress.
    """

    def __init__(
        self,
        calc: ase.calculators.calculator.Calculator,
        optimizer: Optional[ase.optimize.optimize.Optimizer] = BFGS,
        fmax: Optional[float] = 0.01,
        max_cycles: Optional[int] = 1000,
        verbose: Optional[bool] = False,
    ) -> None:
        """Initialize the ASE optimization calculation.

        Args:
            calc: ASE calculator to use for energy and force calculations.
            optimizer: ASE optimizer class (default: BFGS). Common options include
                BFGS, LBFGS, FIRE, and GPMin.
            fmax: Maximum force convergence criterion in eV/Angstrom. Optimization
                stops when all forces are below this value. Default is 0.01.
            max_cycles: Maximum number of optimization steps. Default is 1000.
            verbose: If True, print optimization progress to stdout. Default is False.
        """
        self.calc = calc
        self.optimizer = optimizer
        self.fmax = fmax
        self.max_cycles = max_cycles
        self.verbose = verbose

    def run(
        self, atoms: ase.Atoms, constrained_atoms: Optional[List[int]] = None
    ) -> Tuple[ase.Atoms, float]:
        """
        Perform constrained optimization using ASE.

        Args:
            atoms (ase.Atoms): Molecule to optimize.
            constrained_atoms: Atomic indices to constrain

        Returns:
            atoms (ase.Atoms): Optimized ASE atoms object.
            energy (float): Energy of the optimized atoms object.
        """
        atoms.calc = self.calc
        if constrained_atoms is not None and len(constrained_atoms) > 0:
            atoms.set_constraint(FixAtoms(constrained_atoms))
        # Perform optimization
        if self.verbose:
            opt = self.optimizer(atoms)
            opt.run(fmax=self.fmax, steps=self.max_cycles)
        else:
            with open(
                os.devnull, "w", encoding="utf-8"
            ) as f, contextlib.redirect_stdout(f):
                opt = self.optimizer(atoms)
                opt.run(fmax=self.fmax, steps=self.max_cycles)
        return atoms.get_positions(), atoms.get_potential_energy() * EV_TO_KCAL

    def energy(self, atoms: ase.Atoms) -> float:
        """
        Return the energy of the input atoms object

        Args:
            atoms (ase.Atoms): Input atoms object

        Returns:
            energy (float): Energy of the atoms object
        """
        atoms.calc = self.calc
        return atoms.get_potential_energy() * EV_TO_KCAL


class XTBCalculation(Calculation):
    """Geometry optimization using the xtb command-line program.

    Each calculation shells out to an ``xtb`` binary in its own temporary
    directory and parses the optimized geometry and energy from the files xtb
    writes. This requires no Python bindings (e.g. tblite) — only an ``xtb``
    executable on the PATH — and uses xtb's native ANC optimizer, which
    typically converges in fewer gradient evaluations than driving the SCF
    through an ASE optimizer.

    Attributes:
        charge: Formal charge of the molecule.
        spin_multiplicity: Spin multiplicity (2S+1) of the molecule.
        method: xTB Hamiltonian, one of "gfn0", "gfn1", "gfn2", "gfnff", or
            "gxtb" (g-xTB; requires an xtb build that supports --gxtb).
        opt_level: xtb optimization convergence level (crude, sloppy, loose,
            normal, tight, vtight, extreme).
        solvent: Implicit solvent name for the ALPB model, or None for gas phase.
        max_cycles: Maximum number of optimization cycles, or None for the
            xtb default (automatic, geometry-dependent).
        n_threads: Number of OpenMP threads for each xtb process.
        xtb_path: Path to the xtb executable.
    """

    METHOD_FLAGS = {
        "gfn0": ["--gfn", "0"],
        "gfn1": ["--gfn", "1"],
        "gfn2": ["--gfn", "2"],
        "gfnff": ["--gfnff"],
        "gxtb": ["--gxtb"],
    }

    def __init__(
        self,
        charge: int = 0,
        spin_multiplicity: int = 1,
        method: str = "gfn2",
        opt_level: str = "normal",
        solvent: Optional[str] = None,
        max_cycles: Optional[int] = None,
        n_threads: int = 1,
        xtb_path: str = "xtb",
        cache_restart: bool = False,
    ) -> None:
        """Initialize the xtb command-line calculation.

        Args:
            charge: Formal charge of the molecule. Default is 0.
            spin_multiplicity: Spin multiplicity 2S+1. Default is 1 (singlet).
            method: xTB Hamiltonian: "gfn0", "gfn1", "gfn2", "gfnff", or "gxtb".
                "gxtb" selects the g-xTB method and requires an xtb build that
                supports the --gxtb flag (set xtb_path accordingly).
                Default is "gfn2".
            opt_level: Convergence level passed to ``xtb --opt``. One of
                crude, sloppy, loose, normal, tight, vtight, extreme.
                Default is "normal".
            solvent: ALPB implicit solvent name (e.g. "water", "toluene"), or
                None for gas phase. Default is None.
            max_cycles: Maximum optimization cycles, or None for the xtb
                default. Default is None.
            n_threads: OpenMP threads per xtb process. Keep at 1 when running
                with parallel=True in ConformerEnsemble to avoid
                oversubscription. Default is 1.
            xtb_path: Path to the xtb executable. Default is "xtb" (found on
                the PATH).
            cache_restart: If True, carry the xtb wavefunction restart file
                (``xtbrestart``) from one call to the next, seeding each SCF with
                the previous result. Because successive Monte Carlo structures are
                geometrically similar, this speeds SCF convergence (most useful for
                the more expensive g-xTB method). The cache lives on the instance,
                so each parallel worker (which holds its own copy) keeps its own
                restart with no cross-process clobbering. Default is False.
        """
        if method not in self.METHOD_FLAGS:
            raise ValueError(
                f"Unknown xTB method: {method!r}. "
                f"Choose from {sorted(self.METHOD_FLAGS)}"
            )
        self.charge = charge
        self.spin_multiplicity = spin_multiplicity
        self.method = method
        self.opt_level = opt_level
        self.solvent = solvent
        self.max_cycles = max_cycles
        self.n_threads = n_threads
        self.xtb_path = xtb_path
        self.cache_restart = cache_restart
        # Cached contents of the previous run's xtbrestart file, reused as the SCF
        # guess for the next call when cache_restart is True.
        self._restart_data = None

    def _execute(self, atoms: ase.Atoms, extra_args: List[str], run_dir: str) -> str:
        """Write the geometry and run xtb in run_dir, returning its stdout."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        with open(os.path.join(run_dir, "input.xyz"), "w", encoding="utf-8") as f:
            f.write(f"{len(symbols)}\n\n")
            for symbol, (x, y, z) in zip(symbols, positions):
                f.write(f"{symbol} {x:.10f} {y:.10f} {z:.10f}\n")
        # Seed the SCF from the previous call's wavefunction; xtb auto-reads a
        # file named "xtbrestart" present in the working directory.
        restart_path = os.path.join(run_dir, "xtbrestart")
        if self.cache_restart and self._restart_data is not None:
            with open(restart_path, "wb") as f:
                f.write(self._restart_data)
        command = [
            self.xtb_path,
            "input.xyz",
            "--chrg",
            str(self.charge),
            "--uhf",
            str(self.spin_multiplicity - 1),
            *self.METHOD_FLAGS[self.method],
        ]
        if self.solvent is not None:
            command += ["--alpb", self.solvent]
        command += extra_args
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = f"{self.n_threads},1"
        env["MKL_NUM_THREADS"] = str(self.n_threads)
        result = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-15:])
            raise RuntimeError(
                f"xtb exited with code {result.returncode}:\n{tail}\n{result.stderr}"
            )
        # Persist the updated wavefunction for the next call's SCF guess.
        if self.cache_restart and os.path.exists(restart_path):
            with open(restart_path, "rb") as f:
                self._restart_data = f.read()
        return result.stdout

    def run(
        self, atoms: ase.Atoms, constrained_atoms: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a geometry optimization with the xtb program.

        Args:
            atoms (ase.Atoms): Molecule to optimize.
            constrained_atoms: Atomic indices (0-based) to fix during
                optimization.

        Returns:
            positions (np.ndarray): Optimized cartesian coordinates in Angstroms.
            energy (float): Energy of the optimized structure in kcal/mol.
        """
        with tempfile.TemporaryDirectory() as run_dir:
            extra_args = ["--opt", self.opt_level]
            if self.max_cycles is not None:
                extra_args += ["--cycles", str(self.max_cycles)]
            if constrained_atoms is not None and len(constrained_atoms) > 0:
                # xtb atom indices are 1-based. Exact fixing only holds with
                # the cartesian lbfgs engine; the default ANC engine zeroes
                # gradients in internal coordinates and lets fixed atoms drift.
                fix_list = ",".join(str(i + 1) for i in constrained_atoms)
                with open(os.path.join(run_dir, "xtb.inp"), "w", encoding="utf-8") as f:
                    f.write(f"$fix\n   atoms: {fix_list}\n$end\n")
                    f.write("$opt\n   engine=lbfgs\n$end\n")
                extra_args += ["--input", "xtb.inp"]
            self._execute(atoms, extra_args, run_dir)
            opt_path = os.path.join(run_dir, "xtbopt.xyz")
            if not os.path.exists(opt_path):
                raise RuntimeError(
                    "xtb optimization did not produce xtbopt.xyz "
                    "(geometry optimization likely failed to converge)"
                )
            with open(opt_path, encoding="utf-8") as f:
                lines = f.readlines()
        num_atoms = int(lines[0])
        match = re.search(r"energy:\s*(-?\d+\.\d+)", lines[1])
        if match is None:
            raise RuntimeError(
                f"Could not parse energy from xtbopt.xyz comment line: {lines[1]!r}"
            )
        positions = np.array(
            [
                [float(value) for value in line.split()[1:4]]
                for line in lines[2 : 2 + num_atoms]
            ]
        )
        return positions, float(match.group(1)) * HARTREE_TO_KCAL

    def energy(self, atoms: ase.Atoms) -> float:
        """
        Return the single-point energy of the input atoms object.

        Args:
            atoms (ase.Atoms): Input atoms object

        Returns:
            energy (float): Energy of the atoms object in kcal/mol.
        """
        with tempfile.TemporaryDirectory() as run_dir:
            stdout = self._execute(atoms, [], run_dir)
        match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", stdout)
        if match is None:
            raise RuntimeError("Could not parse TOTAL ENERGY from xtb output")
        return float(match.group(1)) * HARTREE_TO_KCAL
