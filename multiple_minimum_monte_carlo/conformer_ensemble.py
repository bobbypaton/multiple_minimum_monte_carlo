"""Module for running Multiple Minimum Monte Carlo conformer sampling.

This module implements the Multiple Minimum Monte Carlo (MMMC) algorithm for
generating diverse conformer ensembles. The algorithm combines systematic dihedral
angle rotation with energy minimization and RMSD-based filtering to efficiently
explore conformational space.
"""

import os
import sys
from typing import Callable, Optional, List, Tuple, Union
import random
from copy import copy
import logging
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import PeriodicTable, rdMolAlign
import ase
from multiple_minimum_monte_carlo.conformer import Conformer
from multiple_minimum_monte_carlo.calculation import Calculation
from multiple_minimum_monte_carlo.batch_calculation import BatchCalculation
from multiple_minimum_monte_carlo import cheminformatics, multiproc


def run_class_func(cls, func_name, args):
    """Helper function to run a class method with keyword arguments.

    Args:
        cls: The class instance to call the method on.
        func_name: Name of the method to call.
        args: Dictionary of keyword arguments to pass to the method.

    Returns:
        The return value of the called method.
    """
    func = getattr(cls, func_name)
    return func(**args)


class ConformerEnsemble:
    """Multiple Minimum Monte Carlo conformer ensemble generator.

    This class implements the Multiple Minimum Monte Carlo (MMMC) algorithm for
    generating diverse conformer ensembles. The algorithm iteratively:
    1. Selects a conformer from the current ensemble
    2. Randomly rotates a subset of dihedral angles
    3. Optimizes the resulting structure
    4. Filters based on energy and RMSD criteria

    Supports both serial and parallel/batched execution modes for efficient
    conformer generation.

    Attributes:
        conformer (Conformer): The initial conformer structure.
        calc (Union[Calculation, BatchCalculation]): Calculator for optimizations.
        final_ensemble (List[np.ndarray]): Final set of unique conformer coordinates.
        final_energies (List[float]): Energies corresponding to final_ensemble.
        found (List[int]): Degeneracy of each final conformer, i.e. how many times
            it was located during the search (1 = found once).
        origin (List[str]): Where each final conformer came from: "input" for the
            starting structure, "mc" for ones discovered by Monte Carlo sampling.
    """

    def __init__(
        self,
        conformer: Conformer,
        calc: Union[Calculation, BatchCalculation],
        num_iterations: Optional[int] = 100,
        energy_window: Optional[float] = 10.0,
        max_bonds_rotate: Optional[int] = 3,
        max_attempts: Optional[int] = 1000,
        angle_step: Optional[float] = 30.0,
        rmsd_threshold: Optional[float] = 0.3,
        uniqueness_method: Optional[str] = "rmsd",
        ethr: Optional[float] = 0.05,
        rthr: Optional[float] = 0.125,
        bthr: Optional[float] = 0.01,
        bthrmax: Optional[float] = 0.025,
        bthrshift: Optional[float] = 0.5,
        initial_optimization: Optional[bool] = True,
        random_walk: Optional[bool] = False,
        reduce_angle: Optional[bool] = False,
        reduce_angle_every: Optional[int] = 50,
        reduce_angle_by: Optional[int] = 2,
        only_heavy: Optional[bool] = False,
        parallel: Optional[bool] = False,
        num_cpus: Optional[int] = 0,
        batch_size: Optional[int] = 10,
        verbose: Optional[bool] = False,
        parallel_batch_folder_location: Optional[str] = None,
        process_timeout: Optional[float] = 3600,
        step_callback: Optional[Callable] = None,
        fixed_bonds: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Initialize the conformer ensemble generator.

        Args:
            conformer: The initial conformer structure to start ensemble generation.
            calc: A Calculation or BatchCalculation object to perform energy
                minimizations. Use BatchCalculation for GPU-accelerated optimizations.
            num_iterations: Number of Monte Carlo iterations to perform. Each
                iteration generates one or more trial conformers. Default is 100.
            energy_window: Maximum energy window (in kcal/mol) above the minimum
                energy conformer to retain conformers. Conformers with higher
                energies are discarded. Default is 10.0.
            max_bonds_rotate: Maximum number of rotatable bonds to rotate in each
                step. Randomly selects 1 to max_bonds_rotate bonds. Default is 3.
            max_attempts: Maximum number of attempts to generate a valid rotated
                conformer per iteration. Default is 1000.
            angle_step: Step size (in degrees) for dihedral angle rotation. Angles
                are randomly selected from multiples of this value. Default is 30.0.
            rmsd_threshold: RMSD threshold (in Angstroms) for distinguishing unique
                conformers. Conformers with RMSD below this to any existing conformer
                are discarded as duplicates. Default is 0.3. Only used when
                uniqueness_method is "rmsd".
            uniqueness_method: How to decide whether a new conformer is a duplicate
                of an existing one. "rmsd" (default) discards a conformer whose RMSD
                to any existing member is below rmsd_threshold. "crest" follows the
                CREST/CREGEN criterion: a conformer is a duplicate only if its
                energy, rotational constants, and RMSD all match an existing member
                within ethr, an anisotropy-adjusted bthr, and rthr respectively. The
                rotational-constant test makes "crest" robust to symmetry-equivalent
                atom permutations that an RMSD-only test can miss.
            ethr: Pairwise energy threshold (in kcal/mol) for the "crest" method.
                Two conformers closer than this in energy may be treated as
                duplicates. Default is 0.05.
            rthr: RMSD threshold (in Angstroms) for the "crest" method. Default is
                0.125.
            bthr: Lower-bound relative threshold for the rotational-constant
                comparison in the "crest" method. Dynamically widened up to bthrmax
                based on the anisotropy of each pair's rotational constants. Default
                is 0.01 (1%).
            bthrmax: Upper bound for the anisotropy-adjusted rotational-constant
                threshold in the "crest" method. Default is 0.025 (2.5%).
            bthrshift: Shift of the error-function ramp that maps anisotropy onto the
                rotational-constant threshold in the "crest" method. Default is 0.5.
            initial_optimization: If True, perform a structure optimization on the
                starting conformer before Monte Carlo sampling. Default is True.
            random_walk: If True, randomly select conformers from the ensemble for
                modification. If False, preferentially select less-used conformers.
                Default is False.
            reduce_angle: If True, progressively reduce the angle step size during
                the search to enable finer sampling. Default is False.
            reduce_angle_every: Number of iterations between angle step reductions.
                Only used when reduce_angle is True. Default is 50.
            reduce_angle_by: Factor to divide the angle step by at each reduction.
                Only used when reduce_angle is True. Default is 2.
            only_heavy: If True, only rotate dihedrals between heavy atoms (non-H).
                Reduces conformational space but may miss important H-bonding
                arrangements. Default is False.
            parallel: If True, perform calculations in parallel using multiprocessing.
                Not compatible with BatchCalculation. Default is False.
            num_cpus: Number of CPUs to use for parallel calculations. If 0, use all
                available CPUs. Only used when parallel is True. Default is 0.
            batch_size: Number of conformers to process in each batch when using
                BatchCalculation. Default is 10.
            verbose: If True, log Monte Carlo progress to stdout. Default is False.
            parallel_batch_folder_location: Optional path to a directory where
                temporary batch folders are created during parallel execution. If
                None, falls back to TMPDIR, /tmp, or the current working directory.
            process_timeout: Maximum time in seconds to wait for each parallel batch
                to complete. Processes that exceed this limit are terminated. Default
                is 3600 (1 hour). Pass None to wait indefinitely.
            step_callback: Optional callable invoked after each batch of Monte
                Carlo optimizations with (steps_completed, initial_positions,
                positions_and_energies, accepted), where steps_completed is the
                number of Monte Carlo steps performed so far, initial_positions
                is a list of pre-optimization coordinate arrays for the batch,
                positions_and_energies is the corresponding list of
                (optimized_positions, energy) tuples, and accepted is a list of
                booleans indicating whether each conformer passed the energy,
                identity, and RMSD checks and joined the ensemble. If the
                callback returns a truthy value, the Monte Carlo search stops
                early. Useful for progress reporting and convergence-based
                stopping. Default is None.
            fixed_bonds: Optional list of central-bond atom-index pairs (0-based)
                to exclude from the rotatable dihedral list. A torsion is dropped
                if its central b-c bond matches one of these pairs (in either
                order). Unlike constrained_atoms, the atoms are still free to
                relax during optimization; only the random rotation is skipped.
                Default is None.
        """
        self.conformer = conformer
        self.calc = calc
        self.num_iterations = num_iterations
        self.energy_window = energy_window
        self.max_bonds_rotate = max_bonds_rotate
        self.max_attempts = max_attempts
        self.angle_step = angle_step
        self.rmsd_threshold = rmsd_threshold
        if uniqueness_method not in ("rmsd", "crest"):
            raise ValueError(
                f"uniqueness_method must be 'rmsd' or 'crest', got {uniqueness_method!r}"
            )
        self.uniqueness_method = uniqueness_method
        self.ethr = ethr
        self.rthr = rthr
        self.bthr = bthr
        self.bthrmax = bthrmax
        self.bthrshift = bthrshift
        self.initial_optimization = initial_optimization
        self.random_walk = random_walk
        self.reduce_angle = reduce_angle
        self.reduce_angle_every = reduce_angle_every
        self.reduce_angle_by = reduce_angle_by
        self.only_heavy = only_heavy
        self.parallel = parallel
        self.num_cpus = num_cpus
        self.batch = isinstance(self.calc, BatchCalculation)
        self.batch_size = batch_size
        self.verbose = verbose
        self.parallel_batch_folder_location = parallel_batch_folder_location
        self.process_timeout = process_timeout
        self.step_callback = step_callback
        self.fixed_bonds = (
            {frozenset(bond) for bond in fixed_bonds} if fixed_bonds else set()
        )
        if self.num_cpus == 0:
            self.num_cpus = os.cpu_count()
        if self.verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.final_ensemble = []
        self.final_energies = []
        self.found = []
        self.origin = []
        self._duplicate_index = None
        if self.parallel and self.batch:
            self.parallel = False
            self.log_warning(
                "Parallel calculations not supported with batch calculations"
            )

    def log_info(self, message: str) -> None:
        """
        Logs a message
        Args:
            message (str): Message to log
        """
        if self.verbose:
            logging.info(message)

    def log_warning(self, message: str) -> None:
        """
        Logs a warning message
        Args:
            message (str): Message to log
        """
        if self.verbose:
            logging.warning(message)

    def run_monte_carlo(self) -> None:
        """
        Runs a Monte Carlo search to generate a conformer ensemble by iteratively sampling, modifying, and optimizing molecular conformers.
        The method performs the following steps:
            1. Optionally runs an initial optimization on the starting conformer.
            2. Identifies rotatable dihedral angles, excluding those associated with constrained atoms.
            3. Iteratively samples conformers, applies random dihedral rotations, and optimizes the resulting structures.
            4. Filters out high-energy and duplicate conformers.
            5. Sorts the ensemble by energy and returns the final set of unique, low-energy conformers.
        """
        # Run the initial optimization
        if self.initial_optimization:
            self.log_info("Running intitial optimization")
            positions_and_energies = self.run_optimizations([self.conformer.atoms])
            self.conformer.atoms.positions = positions_and_energies[0][0]
            energy = positions_and_energies[0][1]
        else:
            if self.batch:
                energies = self.calc.energy([self.conformer.atoms])
                energy = energies[0]
            else:
                energy = self.calc.energy(self.conformer.atoms)
        # Get the dihedrals to rotate
        dihedrals = cheminformatics.get_dihedral_matches(
            self.conformer.mol, self.only_heavy
        )
        self.max_bonds_rotate = min(len(dihedrals), self.max_bonds_rotate)

        # Remove any dihedrals associated with constrained atoms or whose central
        # bond was explicitly fixed via fixed_bonds.
        final_dihedrals = []
        for dihedral in dihedrals:
            if self.conformer.constrained_atoms is not None and (
                dihedral[1] in self.conformer.constrained_atoms
                and dihedral[2] in self.conformer.constrained_atoms
            ):
                # If the bond is constrained, we need to remove it from the list of rotatable bonds
                continue
            elif frozenset((dihedral[1], dihedral[2])) in self.fixed_bonds:
                # Central bond explicitly fixed by the user: skip the rotation
                continue
            else:
                final_dihedrals.append(dihedral)
        dihedrals = final_dihedrals

        # Report the rotatable bonds (central b-c atom of each torsion), using
        # 1-based atom indices and element symbols (e.g. "C1-C12").
        mol = self.conformer.mol
        bond_labels = [
            f"{mol.GetAtomWithIdx(b).GetSymbol()}{b + 1}-"
            f"{mol.GetAtomWithIdx(c).GetSymbol()}{c + 1}"
            for _, b, c, _ in dihedrals
        ]
        self.log_info(f"Rotating {len(dihedrals)} bond(s): {', '.join(bond_labels)}")

        # Initialize information for identity checking
        # TODO: halides are weird with bond formation occasionally so they are currently gnore
        self.original_bonds, self.metal_atoms, self.halides = (
            cheminformatics.initialize_mc_identity_check(
                self.conformer.atoms, self.conformer.mol
            )
        )

        final_ensemble = [self.conformer.atoms.get_positions()]
        final_energies = [energy]
        used = [0]
        # found[i] = how many times conformer i was located (1 = the initial
        # discovery); incremented each time a trial is rejected as a duplicate of
        # it. origin[i] records where it came from: "input" or "mc".
        found = [1]
        origin = ["input"]
        current_iter = 0
        samples_per_batch = 1
        if self.batch:
            samples_per_batch = self.batch_size
        elif self.parallel:
            samples_per_batch = self.num_cpus
        while current_iter < self.num_iterations:
            self.log_info(
                f"Iteration: {current_iter} Current min energy: {min(final_energies)}"
            )
            # Reduce angle if reduce_angle true
            if self.reduce_angle:
                if current_iter % self.reduce_angle_every == 0:
                    self.angle_step = self.angle_step / self.reduce_angle_by

            # Sample a conformer, rotate its dihedrals, and optimize it (if parallel, do this num_cpus times)
            calculation_input = []
            for _ in range(samples_per_batch):
                current_iter += 1
                success = False
                index = self.sample_conformer(used)
                success, positions = self.modify_conformer(
                    final_ensemble[index], dihedrals
                )
                if success:
                    used[index] += 1
                    atoms_to_optimize = copy(self.conformer.atoms)
                    atoms_to_optimize.set_positions(positions)
                    calculation_input.append(atoms_to_optimize)
            if len(calculation_input) == 0:
                continue
            # Capture pre-optimization coordinates before run_optimizations
            # mutates the atoms objects in place
            initial_positions = [atoms.get_positions() for atoms in calculation_input]
            positions_and_energies = self.run_optimizations(calculation_input)
            # Filter out high energy and duplicate conformers
            accepted = []
            for positions, energy in positions_and_energies:
                if self.check_conformer(
                    final_ensemble, final_energies, positions, energy
                ):
                    final_ensemble.append(positions)
                    final_energies.append(energy)
                    used.append(0)
                    found.append(1)
                    origin.append("mc")
                    accepted.append(True)
                else:
                    # If it was rejected as a duplicate of an existing member,
                    # bump that member's degeneracy count.
                    if self._duplicate_index is not None:
                        found[self._duplicate_index] += 1
                    accepted.append(False)
            stop_requested = False
            if self.step_callback is not None:
                stop_requested = bool(
                    self.step_callback(
                        min(current_iter, self.num_iterations),
                        initial_positions,
                        positions_and_energies,
                        accepted,
                    )
                )

            # Sort all of the lists by energies
            final_ensemble, used, final_energies, found, origin = zip(
                *sorted(
                    zip(final_ensemble, used, final_energies, found, origin),
                    key=lambda x: x[2],
                )
            )
            final_ensemble = list(final_ensemble)
            used = list(used)
            final_energies = list(final_energies)
            found = list(found)
            origin = list(origin)

            if stop_requested:
                self.log_info("Search stopped early by step_callback")
                break

        self.final_ensemble = final_ensemble
        self.final_energies = final_energies
        self.found = found
        self.origin = origin

    def run_optimizations(
        self, atoms_list: List[ase.Atoms]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Runs optimizations on a list of ASE Atoms objects using the provided calculation method.
        Args:
            atoms_list (List[ase.Atoms]): List of ASE Atoms objects to optimize.
        Returns:
            List[Tuple[np.ndarray, float]]: List of tuples containing optimized positions and energies.
        """
        if self.parallel:
            calculation_input = []
            for atoms in atoms_list:
                calculation_input.append(
                    {
                        "cls": self.calc,
                        "func_name": "run",
                        "args": {
                            "atoms": atoms,
                            "constrained_atoms": self.conformer.constrained_atoms,
                        },
                    }
                )
            workers = min(self.num_cpus, len(calculation_input))
            results = multiproc.parallel_run_proc(
                run_class_func,
                calculation_input,
                workers,
                self.parallel_batch_folder_location,
                self.process_timeout,
            )
        elif self.batch:
            if self.conformer.constrained_atoms is not None:
                constrained_atoms = [self.conformer.constrained_atoms] * len(atoms_list)
            else:
                constrained_atoms = None
            positions_list, energies = self.calc.run(
                atoms_list=atoms_list,
                constrained_atoms_list=constrained_atoms,
            )
            results = list(zip(positions_list, energies))
        else:
            results = []
            for atoms in atoms_list:
                positions, energy = self.calc.run(
                    atoms, self.conformer.constrained_atoms
                )
                results.append((positions, energy))
        return results

    def sample_conformer(self, used: List[int]) -> int:
        """
        Selects a conformer index based on the sampling strategy.
        If random_walk is enabled, randomly selects an index from the provided list of used indices.
        Otherwise, selects the index corresponding to the minimum value in the used list.
        Args:
            used (list or array-like): A list of indices or values representing conformer usage.
        Returns:
            int: The selected conformer index.
        """
        if self.random_walk:
            index = random.choice(range(len(used)))
        else:
            index = np.argmin(np.array(used))
        return index

    def modify_conformer(
        self, conformer: np.ndarray, all_dihedrals: List[Tuple[int, int, int, int]]
    ) -> Tuple[bool, np.ndarray]:
        """
        Attempts to modify a given conformer by randomly rotating a subset of its dihedral angles.
        For a maximum number of attempts (`self.max_attempts`), this method:
            - Copies the input conformer and its associated molecule.
            - Randomly selects a subset of dihedral angles to rotate.
            - Applies a rotation to the selected dihedrals by a fixed angle step (`self.angle_step`).
            - Tests the modified conformer against a set of constraints (`self.constraint_test`).
        If a valid conformer is found that passes the constraint test, the process stops early.
        Args:
            conformer (np.ndarray): The input conformer coordinates to be modified.
            all_dihedrals (List[Tuple[int, int, int, int]]): List of all possible dihedral angles (as atom index tuples) in the molecule.
        Returns:
            Tuple[bool, np.ndarray]:
                - A boolean indicating whether a valid conformer was found.
                - The resulting np.ndarray with the coordinates of the modified conformer.
        """
        success = False
        for _ in range(self.max_attempts):
            temp_mol = copy(self.conformer.mol)
            temp_mol = cheminformatics.add_coords_to_mol(conformer, temp_mol)
            num_dihedrals = random.randint(1, self.max_bonds_rotate)
            dihedrals = random.choices(all_dihedrals, k=num_dihedrals)
            cheminformatics.rotate_dihedrals(
                temp_mol.GetConformer(), dihedrals, self.angle_step
            )
            if self.constraint_test(temp_mol.GetConformer()):
                success = True
                break
        atoms = cheminformatics.mol_to_ase_atoms(temp_mol)
        return success, atoms.get_positions()

    def constraint_test(self, conf: Chem.rdchem.Conformer) -> bool:
        """
        Determine whether the conformer meets the constraint test 2 as defined in CITATION.
        This function checks whether the interatomic distances between non-bonded atoms is under 1/4 of the
        van der Waals radius of the atoms. If the distance is under this threshold, the conformer is considered invalid.

        Parameters
        ----------
        conf : rdkit.Chem.rdchem.Conformer
            The conformer to be tested.
        Returns
        -------
        bool
            True if the conformer passes the constraint test, False otherwise.
        """
        # Get the 3D coordinates of the atoms in the conformer
        coords = conf.GetPositions()
        # Calculate the distance matrix between all pairs of atoms
        dist_matrix = distance_matrix(coords, coords)
        # Get the van der Waals radii for each atom in the conformer
        vdw_radii = [
            PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum())
            for atom in self.conformer.mol.GetAtoms()
        ]
        # Make a matrix of the van der Waals radii
        vdw_matrix = np.array(
            [
                [vdw_radii[i] + vdw_radii[j] for j in range(len(vdw_radii))]
                for i in range(len(vdw_radii))
            ]
        )
        vdw_matrix = vdw_matrix / 4.0  # Scale the van der Waals radii by 1/4
        # Set the distances between bonded atoms to zero (to ignore them in the test)
        for bonded_atom_pair in self.conformer.bonded_atoms:
            i, j = bonded_atom_pair
            vdw_matrix[i][j] = 0.0
            vdw_matrix[j][i] = 0.0
        # Set the diagonal of the van der Waals matrix to zero (to ignore self-distances)
        np.fill_diagonal(vdw_matrix, 0.0)
        # Check if the distance between any two non-bonded atoms is less than 1/4 of the van der Waals radius
        difference_matrix = dist_matrix - vdw_matrix
        # If any distance is less than 0, the conformer fails the constraint test
        if np.any(difference_matrix < 0):
            return False
        # If all distances are greater than or equal to 0, the conformer passes the constraint test
        return True

    def check_conformer(
        self,
        ensemble: List[np.ndarray],
        energies: List[float],
        conf: np.ndarray,
        energy: float,
    ) -> bool:
        """
        Checks whether a given conformer should be added to the ensemble based on energy and structural similarity.
        This function evaluates if the provided conformer (`conf`) with its associated energy (`energy`) is sufficiently low in energy
        and structurally distinct from all conformers already present in the ensemble. The conformer is accepted if:
          - Its energy is within `self.energy_window` of the minimum energy in the current ensemble.
          - It preserves the input bond connectivity (identity check).
          - It is distinct from every existing member under the selected uniqueness method:
              * "rmsd": its RMSD to all members exceeds `self.rmsd_threshold`.
              * "crest": no member matches it on all of energy (`self.ethr`),
                rotational constants (anisotropy-adjusted `self.bthr`), and RMSD
                (`self.rthr`) simultaneously.
        Args:
            ensemble (list): List of conformers cooordinates currently in the ensemble, each represented as np.ndarray objects.
            energies (list): List of energies corresponding to the conformers in the ensemble.
            conf (np.ndarray): The candidate conformer to be evaluated.
            energy (float): The energy of the candidate conformer.
        Returns:
            bool: True if the conformer passes both the energy and RMSD criteria and should be added to the ensemble, False otherwise.
        """

        # Index of the existing member this candidate duplicates, if any. Reset
        # each call; set only when a duplicate match is found below.
        self._duplicate_index = None
        self.log_info(
            f"Checking candidate (E={energy:.4f} kcal/mol) against "
            f"{len(ensemble)} ensemble member(s) using '{self.uniqueness_method}' criterion"
        )
        if energy > min(energies) + self.energy_window:
            self.log_info(
                f"  rejected: energy exceeds window "
                f"(min {min(energies):.4f} + {self.energy_window} kcal/mol)"
            )
            return False
        temp_atoms = copy(self.conformer.atoms)
        temp_atoms.set_positions(conf)
        try:
            identity = cheminformatics.check_identity_mc(
                self.original_bonds, self.metal_atoms, self.halides, temp_atoms
            )
        except Exception as e:
            self.log_warning(
                f"Identity check failed with error: {e}. This may be due to issues with the input structure or the cheminformatics library. The conformer will be rejected."
            )
            return False
        if not identity:
            self.log_info("  rejected: connectivity changed (identity check failed)")
            return False
        temp_mol = copy(self.conformer.mol)
        temp_mol = cheminformatics.add_coords_to_mol(conf, temp_mol)
        temp_reference_mol = copy(self.conformer.mol)
        atom_map = [(i, i) for i in range(len(temp_mol.GetAtoms()))]
        if self.uniqueness_method == "crest":
            masses = self.conformer.atoms.get_masses()
            candidate_rot = cheminformatics.rotational_constants(conf, masses)
            for idx, (reference_conf, reference_energy) in enumerate(
                zip(ensemble, energies)
            ):
                # Cheapest checks first: energy, then rotational constants, then
                # the comparatively expensive RMSD alignment. A member is a
                # duplicate only if all three match.
                delta_e = abs(energy - reference_energy)
                if delta_e >= self.ethr:
                    self.log_info(
                        f"  vs #{idx}: dE={delta_e:.4f} >= ethr {self.ethr} -> distinct"
                    )
                    continue
                reference_rot = cheminformatics.rotational_constants(
                    reference_conf, masses
                )
                if not cheminformatics.rotational_constants_equal(
                    candidate_rot,
                    reference_rot,
                    self.bthr,
                    self.bthrmax,
                    self.bthrshift,
                ):
                    if self.verbose:
                        rot_max = float(
                            np.max(np.abs(candidate_rot / reference_rot - 1.0))
                        )
                        self.log_info(
                            f"  vs #{idx}: dE={delta_e:.4f} < ethr, "
                            f"rot |dB|max={rot_max:.4f} mismatch -> distinct"
                        )
                    continue
                temp_reference_mol = cheminformatics.add_coords_to_mol(
                    reference_conf, temp_reference_mol
                )
                rmsd = rdMolAlign.AlignMol(
                    temp_mol, temp_reference_mol, atomMap=atom_map
                )
                if rmsd < self.rthr:
                    self.log_info(
                        f"  vs #{idx}: dE={delta_e:.4f} < ethr, rot match, "
                        f"rmsd={rmsd:.4f} < rthr {self.rthr} -> DUPLICATE"
                    )
                    self._duplicate_index = idx
                    return False
                self.log_info(
                    f"  vs #{idx}: dE={delta_e:.4f} < ethr, rot match, "
                    f"rmsd={rmsd:.4f} >= rthr {self.rthr} -> distinct"
                )
            self.log_info("  accepted: distinct from all members")
            return True
        for idx, reference_conf in enumerate(ensemble):
            temp_reference_mol = cheminformatics.add_coords_to_mol(
                reference_conf, temp_reference_mol
            )
            rmsd = rdMolAlign.AlignMol(temp_mol, temp_reference_mol, atomMap=atom_map)
            if rmsd < self.rmsd_threshold:
                self.log_info(
                    f"  vs #{idx}: rmsd={rmsd:.4f} < threshold "
                    f"{self.rmsd_threshold} -> DUPLICATE"
                )
                self._duplicate_index = idx
                return False
            self.log_info(
                f"  vs #{idx}: rmsd={rmsd:.4f} >= threshold "
                f"{self.rmsd_threshold} -> distinct"
            )
        self.log_info("  accepted: distinct from all members")
        return True
