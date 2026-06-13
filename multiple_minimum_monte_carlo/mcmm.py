#!/usr/bin/env python
"""CLI for running a Multiple Minimum Monte Carlo conformer search on an XYZ file.

Reads an input XYZ structure, runs the MMMC search with the selected calculator
(see --model; each backend requires its own package, installed separately), and
writes a multi-frame XYZ file containing the final ensemble ordered by energy
(lowest first). Each frame's comment line records its energy in kcal/mol.

Example:
    mcmm example.xyz --charge 0 --steps 30    # writes example_mcmm.xyz

or equivalently:
    python -m multiple_minimum_monte_carlo.mcmm example.xyz --charge 0 --steps 30
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from copy import copy

import numpy as np
from ase.optimize import BFGS, FIRE, LBFGS
from rdkit.Chem import rdMolTransforms
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from multiple_minimum_monte_carlo import cheminformatics
from multiple_minimum_monte_carlo.calculation import ASEOptimization, XTBCalculation
from multiple_minimum_monte_carlo.conformer import Conformer
from multiple_minimum_monte_carlo.conformer_ensemble import ConformerEnsemble

console = Console(highlight=False)

OPTIMIZERS = {"fire": FIRE, "bfgs": BFGS, "lbfgs": LBFGS}

MODELS = {
    "aimnet2": "AIMNet2",
    "mace-off": "MACE-OFF",
    "ani2x": "ANI-2x",
    "xtb": "GFN2-xTB",
    "xtb-cli": "GFN2-xTB (xtb binary)",
    "gxtb-cli": "g-xTB (xtb binary)",
    "uma": "UMA (FairChem)",
}

HARTREE_TO_KCAL = 627.5094740631
"""float: Conversion factor from hartree to kilocalories per mole."""


def build_calculator(name, charge, spin_multiplicity):
    """Construct the ASE calculator for the requested model.

    Imports the backend lazily and exits with an install hint if it is not
    installed. Returns a tuple of (calculator, device_string).
    """
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    if name == "aimnet2":
        try:
            from aimnet.calculators import AIMNet2ASE
        except ImportError:
            sys.exit(
                "AIMNet2 is not installed.\nInstall it with: "
                "pip install git+https://github.com/isayevlab/aimnetcentral.git"
            )
        calc = AIMNet2ASE(charge=charge, mult=spin_multiplicity)
        return calc, str(calc.base_calc.device)
    if name == "mace-off":
        try:
            from mace.calculators import mace_off
        except ImportError:
            sys.exit("MACE is not installed.\nInstall it with: pip install mace-torch")
        if charge != 0 or spin_multiplicity != 1:
            print(
                "Warning: MACE-OFF is trained on neutral singlets; "
                "charge and spin multiplicity are ignored"
            )
        return mace_off(model="medium", device=device), device
    if name == "ani2x":
        try:
            import torch
            import torchani
        except ImportError:
            sys.exit(
                "TorchANI is not installed.\nInstall it with: pip install torchani"
            )
        if charge != 0 or spin_multiplicity != 1:
            print(
                "Warning: ANI-2x supports only neutral singlets; "
                "charge and spin multiplicity are ignored"
            )
        return torchani.models.ANI2x().to(torch.device(device)).ase(), device
    if name == "xtb":
        try:
            from tblite.ase import TBLite
        except ImportError:
            sys.exit("tblite is not installed.\nInstall it with: pip install tblite")
        calc = TBLite(
            method="GFN2-xTB",
            charge=charge,
            multiplicity=spin_multiplicity,
            verbosity=0,
        )
        return calc, "cpu"
    if name == "uma":
        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError:
            sys.exit(
                "FairChem is not installed.\nInstall it with: "
                "pip install fairchem-core (the uma-s-1 model also requires "
                "requesting access on HuggingFace)"
            )
        predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=device)
        return FAIRChemCalculator(predictor, task_name="omol"), device
    raise ValueError(f"Unknown model: {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiple Minimum Monte Carlo conformer search on an XYZ file."
    )
    parser.add_argument("input_xyz", help="Path to the input XYZ structure")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output multi-frame XYZ file with conformers ordered by energy "
        "(default: input filename with an _mcmm suffix, e.g. file.xyz -> file_mcmm.xyz)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Formal charge of the molecule, used to infer bonds from the XYZ (default: 0)",
    )
    parser.add_argument(
        "--smiles",
        default=None,
        help="Optional SMILES string; provide when bond determination from the XYZ fails "
        "(use an atom-mapped SMILES with --mapped for ambiguous structures)",
    )
    parser.add_argument(
        "--mapped",
        action="store_true",
        help="Treat --smiles as atom-mapped to the XYZ (hydrogens must be mapped too)",
    )
    parser.add_argument(
        "--spin-multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity 2S+1 (default: 1, singlet)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of Monte Carlo steps (default: 100)",
    )
    parser.add_argument(
        "--energy-window",
        type=float,
        default=10.0,
        help="Energy window in kcal/mol above the minimum for keeping conformers (default: 10.0)",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.3,
        help="RMSD threshold in Angstroms for duplicate filtering, 'rmsd' method only (default: 0.3)",
    )
    parser.add_argument(
        "--uniqueness-method",
        choices=("rmsd", "crest"),
        default="rmsd",
        help="Duplicate-detection method: 'rmsd' (RMSD only) or 'crest' (energy + rotational constants + RMSD, CREST/CREGEN-style) (default: rmsd)",
    )
    parser.add_argument(
        "--ethr",
        type=float,
        default=0.05,
        help="Pairwise energy threshold in kcal/mol for the 'crest' method (default: 0.05)",
    )
    parser.add_argument(
        "--rthr",
        type=float,
        default=0.125,
        help="RMSD threshold in Angstroms for the 'crest' method (default: 0.125)",
    )
    parser.add_argument(
        "--bthr",
        type=float,
        default=0.01,
        help="Lower-bound relative rotational-constant threshold for the 'crest' method (default: 0.01)",
    )
    parser.add_argument(
        "--bthrmax",
        type=float,
        default=0.025,
        help="Upper-bound relative rotational-constant threshold for the 'crest' method; "
        "the threshold is widened toward this value for anisotropic tops (default: 0.025)",
    )
    parser.add_argument(
        "--bthrshift",
        type=float,
        default=0.5,
        help="Anisotropy shift controlling how quickly the rotational-constant threshold "
        "widens from bthr toward bthrmax for the 'crest' method (default: 0.5)",
    )
    parser.add_argument(
        "--rmsd-all-atom",
        action="store_true",
        help="Include hydrogens in the duplicate-detection RMSD (both 'rmsd' and "
        "'crest' methods). By default the RMSD is heavy-atom only, so methyl/hydroxyl "
        "rotamers are not counted as distinct conformers; pass this to restore the "
        "legacy all-atom RMSD",
    )
    parser.add_argument(
        "--rmsd-symmetry",
        action="store_true",
        help="Permute topologically equivalent atoms (ring flips, equivalent "
        "substituents) when computing the duplicate-detection RMSD, via RDKit's "
        "GetBestRMS. More robust for symmetric molecules but slower (default: off)",
    )
    parser.add_argument(
        "--no-detect-enantiomers",
        action="store_true",
        help="Disable enantiomer detection. By default each candidate's mirror image "
        "is also compared, and a conformer matching an existing member only after "
        "inversion is rejected as a redundant enantiomer (logged as ENANTIOMER rather "
        "than DUPLICATE); pass this to keep such mirror-image conformers",
    )
    parser.add_argument(
        "--max-bonds-rotate",
        type=int,
        default=3,
        help="Maximum number of dihedrals rotated per step (default: 3)",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=60.0,
        help="Dihedral rotation step size in degrees (default: 60.0)",
    )
    parser.add_argument(
        "--fix",
        action="extend",
        nargs="+",
        default=[],
        metavar="BOND",
        help="Rotatable bond(s) to hold fixed (not rotated), named by their "
        "1-indexed central atoms as printed in 'Rotatable bonds', e.g. "
        "--fix C5-O4 (or just 5-4). Repeatable and space-separated. The atoms "
        "still relax during optimization; only the random rotation is skipped",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="aimnet2",
        help="Energy model to use (default: aimnet2). Each backend needs its own "
        "package: aimnet2 (aimnet), mace-off (mace-torch), ani2x (torchani), "
        "xtb (tblite), xtb-cli (the xtb executable, GFN2), gxtb-cli (the xtb "
        "executable run with --gxtb; needs a build supporting it, see --xtb-path), "
        "uma (fairchem-core)",
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZERS),
        default="lbfgs",
        help="ASE optimizer to use (default: lbfgs); "
        "ignored by xtb-cli, which uses xtb's native optimizer",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion in eV/Angstrom (default: 0.05); "
        "ignored by xtb-cli, see --opt-level",
    )
    parser.add_argument(
        "--opt-level",
        choices=("crude", "sloppy", "loose", "normal", "tight", "vtight", "extreme"),
        default="tight",
        help="Optimization convergence level for the xtb-cli and gxtb-cli models "
        "(default: tight); other models use --fmax",
    )
    parser.add_argument(
        "--xtb-path",
        default=None,
        help="Path to the xtb executable for the xtb-cli and gxtb-cli models. "
        "If omitted, xtb-cli uses 'xtb' on PATH; gxtb-cli auto-detects a "
        "g-xTB-capable build, searching $XTBHOME/bin/xtb, ~/xtb/bin/xtb, then PATH",
    )
    parser.add_argument(
        "--no-initial-optimization",
        action="store_true",
        help="Skip the optimization of the input structure before sampling",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run optimizations in parallel (requires the 'fork' multiprocessing start method)",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=0,
        help="Number of conformer optimizations to run concurrently in --parallel "
        "mode; 0 uses all available cores (default: 0). Each runs as its own process",
    )
    parser.add_argument(
        "--xtb-threads",
        type=int,
        default=1,
        help="OpenMP threads per xtb process for the xtb-cli and gxtb-cli models "
        "(default: 1). Use this to give a single optimization more cores; keep it "
        "at 1 with --parallel to avoid oversubscribing (total load is "
        "num-cpus x xtb-threads)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show a progress bar instead of per-step dihedral and energy output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the per-step uniqueness comparisons against previous conformers",
    )
    return parser.parse_args()


def get_dihedral_angles(mol, dihedrals, positions):
    """Measure the dihedral angles (in degrees) of a set of torsions for the
    given coordinates."""
    temp_mol = cheminformatics.add_coords_to_mol(positions, copy(mol))
    conf = temp_mol.GetConformer()
    return [rdMolTransforms.GetDihedralDeg(conf, *dihedral) for dihedral in dihedrals]


def format_angles(angles):
    return "[" + ", ".join(f"{angle:7.1f}" for angle in angles) + "]"


def _xtb_supports_gxtb(exe):
    """Return True if the xtb executable advertises the --gxtb flag."""
    try:
        result = subprocess.run(
            [exe, "--help"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--gxtb" in result.stdout


def resolve_xtb_path(requested, require_gxtb):
    """Resolve the xtb executable for the xtb-cli / gxtb-cli backends.

    If ``requested`` is given it is the only candidate (and is validated). When
    it is None, xtb-cli falls back to ``xtb`` on PATH, while gxtb-cli auto-detects
    a g-xTB-capable build from standard install locations. Exits with a helpful
    message if no suitable executable is found.
    """
    if requested is not None:
        exe = shutil.which(requested)
        if exe is None:
            sys.exit(
                f"xtb executable not found: {requested!r}.\n"
                "Install it with: conda install -c conda-forge xtb, or pass a "
                "valid path with --xtb-path."
            )
        if require_gxtb and not _xtb_supports_gxtb(exe):
            sys.exit(
                f"The xtb at {exe} does not support --gxtb.\n"
                "Point --xtb-path at a build that implements g-xTB "
                "(e.g. ~/xtb/bin/xtb)."
            )
        return exe
    if not require_gxtb:
        exe = shutil.which("xtb")
        if exe is None:
            sys.exit(
                "xtb executable not found on PATH.\nInstall it with: "
                "conda install -c conda-forge xtb, or pass a path with --xtb-path."
            )
        return exe
    # gxtb-cli with no explicit path: search standard locations for a g-xTB build.
    candidates = []
    if os.environ.get("XTBHOME"):
        candidates.append(os.path.join(os.environ["XTBHOME"], "bin", "xtb"))
    candidates.append(os.path.expanduser("~/xtb/bin/xtb"))
    candidates.append("xtb")
    searched = []
    for candidate in candidates:
        exe = shutil.which(candidate)
        if exe is None or exe in searched:
            continue
        searched.append(exe)
        if _xtb_supports_gxtb(exe):
            return exe
    searched_str = ", ".join(searched) if searched else "(none found)"
    sys.exit(
        "No xtb build supporting --gxtb was found.\n"
        f"Searched: {searched_str}.\n"
        "Point --xtb-path at a g-xTB build (e.g. ~/xtb/bin/xtb)."
    )


def print_pairwise_matrix(title, n, value_fn, fmt="{:.3f}"):
    """Print an upper-triangle pairwise matrix over the final conformers.

    value_fn(i, j) returns the scalar shown for the (i, j) pair (i < j); the
    diagonal and lower triangle are left blank.
    """
    console.print(f"\n[bold]{title}[/]")
    matrix = Table(box=box.SIMPLE, header_style="bold magenta", pad_edge=False)
    matrix.add_column("", justify="right", style="dim")
    for j in range(n):
        matrix.add_column(str(j + 1), justify="right")
    for i in range(n):
        cells = [str(i + 1)]
        for j in range(n):
            cells.append(fmt.format(value_fn(i, j)) if j > i else "")
        matrix.add_row(*cells)
    console.print(matrix)


def main():
    args = parse_args()
    if args.output is None:
        root, _ = os.path.splitext(args.input_xyz)
        args.output = f"{root}_mcmm.xyz"

    conformer = Conformer(
        smiles=args.smiles,
        mapped=args.mapped,
        input_xyz=args.input_xyz,
        charge=args.charge,
        spin_multiplicity=args.spin_multiplicity,
    )

    dihedrals = cheminformatics.get_dihedral_matches(conformer.mol, False)

    def bond_label(b, c):
        return (
            f"{conformer.mol.GetAtomWithIdx(b).GetSymbol()}{b + 1}-"
            f"{conformer.mol.GetAtomWithIdx(c).GetSymbol()}{c + 1}"
        )

    # Resolve any --fix bonds (named by 1-indexed central atoms) to their 0-based
    # central-bond pairs, validating against the actual rotatable bonds.
    available = {frozenset((b + 1, c + 1)): (b, c) for _, b, c, _ in dihedrals}
    all_labels = [bond_label(b, c) for _, b, c, _ in dihedrals]
    fixed_bonds = []
    for token in args.fix:
        nums = re.findall(r"\d+", token)
        if len(nums) != 2:
            sys.exit(f"--fix value '{token}' must name two atoms, e.g. C5-O4 or 5-4")
        key = frozenset((int(nums[0]), int(nums[1])))
        if key not in available:
            sys.exit(
                f"--fix bond '{token}' is not a rotatable bond. "
                f"Available: {', '.join(all_labels) or 'none'}"
            )
        fixed_bonds.append(available[key])

    fixed_set = {frozenset(bc) for bc in fixed_bonds}
    dihedrals = [d for d in dihedrals if frozenset((d[1], d[2])) not in fixed_set]
    bond_labels = [bond_label(b, c) for _, b, c, _ in dihedrals]

    console.print(
        f"[bold]Molecular formula:[/] [cyan]{conformer.atoms.get_chemical_formula()}[/]"
    )
    console.print(f"[bold]Rotatable torsions:[/] {len(dihedrals)}")
    if bond_labels:
        console.print(f"[bold]Rotatable bonds:[/] [green]{', '.join(bond_labels)}[/]")
    if fixed_bonds:
        fixed_str = ", ".join(bond_label(b, c) for b, c in fixed_bonds)
        console.print(f"[bold]Fixed bonds (not rotated):[/] [red]{fixed_str}[/]")
    console.print(f"[bold]Estimated conformer space (3^N):[/] {3 ** len(dihedrals)}")

    if args.model in ("xtb-cli", "gxtb-cli"):
        method = "gxtb" if args.model == "gxtb-cli" else "gfn2"
        xtb_exe = resolve_xtb_path(args.xtb_path, require_gxtb=method == "gxtb")
        console.print(f"[bold]xtb binary:[/] [cyan]{xtb_exe}[/]")
        device = "cpu"
        calc = XTBCalculation(
            charge=args.charge,
            spin_multiplicity=args.spin_multiplicity,
            method=method,
            opt_level=args.opt_level,
            n_threads=args.xtb_threads,
            xtb_path=xtb_exe,
        )
    else:
        model_calc, device = build_calculator(
            args.model, args.charge, args.spin_multiplicity
        )
        calc = ASEOptimization(
            calc=model_calc,
            optimizer=OPTIMIZERS[args.optimizer],
            fmax=args.fmax,
        )
    console.print(f"[bold]Model:[/] {MODELS[args.model]}")
    if device.startswith("cuda"):
        console.print(f"[bold]GPU acceleration:[/] [green]yes ({device})[/]")
    else:
        console.print(f"[bold]GPU acceleration:[/] [yellow]no (running on {device})[/]")

    if not args.no_initial_optimization:
        console.print("[bold]Running initial optimization...[/]")
        positions, global_min = calc.run(conformer.atoms)
        conformer.atoms.set_positions(positions)
        console.print(
            f"[bold]Initial structure energy:[/] {global_min / HARTREE_TO_KCAL:.6f} Eh"
        )
    else:
        global_min = calc.energy(conformer.atoms)
        console.print(
            f"[bold]Initial structure energy (unoptimized):[/] "
            f"{global_min / HARTREE_TO_KCAL:.6f} Eh"
        )
    console.print(f"[bold]Monte Carlo steps:[/] {args.steps}")

    bar_width = 40
    acceptance_history = []
    stopped_early = False

    def report_step(steps_done, initial_positions, results, accepted, refined):
        nonlocal global_min, stopped_early
        acceptance_history.extend(accepted)
        overall_rate = sum(acceptance_history) / len(acceptance_history)
        last_10 = acceptance_history[-10:]
        last_10_rate = sum(last_10) / len(last_10)
        # Only structures kept in the ensemble (accepted as new, or a lower-energy
        # duplicate that refined an existing member) can set the global minimum.
        retained_energies = [
            energy
            for (_, energy), acc, ref in zip(results, accepted, refined)
            if acc or ref
        ]
        if args.quiet:
            if retained_energies:
                global_min = min(global_min, *retained_energies)
            filled = bar_width * steps_done // args.steps
            print(
                f"\r[{'#' * filled}{'-' * (bar_width - filled)}] "
                f"{steps_done}/{args.steps} | "
                f"global min: {global_min / HARTREE_TO_KCAL:.6f} Eh | "
                f"accepted: {sum(acceptance_history)} "
                f"({overall_rate:.0%}, last 10: {last_10_rate:.0%})",
                end="",
                flush=True,
            )
        else:
            first_step = steps_done - len(initial_positions) + 1
            for i, (initial, (optimized, energy)) in enumerate(
                zip(initial_positions, results)
            ):
                retained = accepted[i] or refined[i]
                new_minimum = retained and energy < global_min
                if new_minimum:
                    global_min = energy
                guess = get_dihedral_angles(conformer.mol, dihedrals, initial)
                final = get_dihedral_angles(conformer.mol, dihedrals, optimized)
                if accepted[i]:
                    verdict = "[green]accepted[/]"
                elif refined[i]:
                    verdict = "[yellow]refined[/]"
                else:
                    verdict = "[red]rejected[/]"
                console.print(
                    f"[bold]Step {first_step + i}/{args.steps}:[/] "
                    f"energy: {energy / HARTREE_TO_KCAL:.6f} Eh "
                    f"(rel: {energy - global_min:.2f} kcal/mol) \\[{verdict}] "
                    f"(acc: {overall_rate:.0%}, last 10: {last_10_rate:.0%})"
                )
                console.print(
                    f"  [dim]dihedral guess:      {escape(format_angles(guess))}[/]"
                )
                console.print(
                    f"  [dim]optimized dihedrals: {escape(format_angles(final))}[/]"
                )
                if new_minimum:
                    if accepted[i]:
                        console.print("  [bold yellow]*** new global minimum ***[/]")
                    else:
                        console.print(
                            "  [bold yellow]*** new global minimum "
                            "(refined existing conformer) ***[/]"
                        )
        if len(acceptance_history) >= 10 and sum(acceptance_history[-10:]) == 0:
            stopped_early = True
            if args.quiet:
                print()
            console.print(
                "[yellow]No conformers accepted in the last 10 steps - "
                "stopping search early[/]"
            )
            return True
        return False

    ensemble = ConformerEnsemble(
        conformer=conformer,
        calc=calc,
        num_iterations=args.steps,
        energy_window=args.energy_window,
        rmsd_threshold=args.rmsd_threshold,
        uniqueness_method=args.uniqueness_method,
        ethr=args.ethr,
        rthr=args.rthr,
        bthr=args.bthr,
        bthrmax=args.bthrmax,
        bthrshift=args.bthrshift,
        rmsd_heavy_only=not args.rmsd_all_atom,
        rmsd_symmetry=args.rmsd_symmetry,
        detect_enantiomers=not args.no_detect_enantiomers,
        max_bonds_rotate=args.max_bonds_rotate,
        angle_step=args.angle_step,
        initial_optimization=False,
        parallel=args.parallel,
        num_cpus=args.num_cpus,
        verbose=args.verbose,
        step_callback=report_step,
        fixed_bonds=fixed_bonds,
    )
    ensemble.run_monte_carlo()
    if args.quiet and not stopped_early:
        print()

    symbols = conformer.atoms.get_chemical_symbols()
    with open(args.output, "w", encoding="utf-8") as f:
        for positions, energy in zip(ensemble.final_ensemble, ensemble.final_energies):
            f.write(f"{len(symbols)}\n")
            f.write(f"energy_kcal_mol={energy:.6f}\n")
            for symbol, (x, y, z) in zip(symbols, positions):
                f.write(f"{symbol:2s} {x:18.10f} {y:18.10f} {z:18.10f}\n")

    boltzmann_kcal = 0.0019872041  # kcal/(mol K)
    temperature = 298.15
    e_min = ensemble.final_energies[0]
    rel_energies = [e - e_min for e in ensemble.final_energies]
    weights = [math.exp(-de / (boltzmann_kcal * temperature)) for de in rel_energies]
    total_weight = sum(weights)

    console.print()
    console.rule("[bold cyan]Final Ensemble Information[/]")
    console.print(f"[bold]output file name[/]               : [cyan]{args.output}[/]")
    console.print(
        f"[bold]conformer energy window  /kcal[/] : {args.energy_window:8.4f}"
    )
    console.print(
        f"[bold]total number unique conformers[/] : {len(ensemble.final_ensemble):8d}"
    )
    console.print(
        f"[bold]lowest energy conformer    /Eh[/] : "
        f"[green]{e_min / HARTREE_TO_KCAL:.6f}[/]"
    )

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Erel/kcal", justify="right")
    table.add_column("Etot/Eh", justify="right", style="cyan")
    table.add_column("weight/tot", justify="right", style="green")
    table.add_column("found", justify="right", style="yellow")
    table.add_column("origin", justify="center")
    for i, (energy, de, weight) in enumerate(
        zip(ensemble.final_energies, rel_energies, weights)
    ):
        origin = ensemble.origin[i]
        origin_cell = f"[blue]{origin}[/]" if origin == "input" else origin
        table.add_row(
            str(i + 1),
            f"{de:.3f}",
            f"{energy / HARTREE_TO_KCAL:.5f}",
            f"{weight / total_weight:.5f}",
            str(ensemble.found[i]),
            origin_cell,
            style="bold" if i == 0 else None,
        )
    console.print(table)

    if args.verbose and len(ensemble.final_ensemble) >= 2:
        n = len(ensemble.final_ensemble)
        masses = ensemble.conformer.atoms.get_masses()
        rots = [
            cheminformatics.rotational_constants(positions, masses)
            for positions in ensemble.final_ensemble
        ]
        print_pairwise_matrix(
            "Pairwise |dB|max (max relative rotational-constant difference)",
            n,
            lambda i, j: float(np.max(np.abs(rots[i] / rots[j] - 1.0))),
            fmt="{:.4f}",
        )
        print_pairwise_matrix(
            "Pairwise RMSD / Angstrom",
            n,
            lambda i, j: ensemble._pairwise_rmsd(
                ensemble.final_ensemble[i], ensemble.final_ensemble[j]
            ),
            fmt="{:.3f}",
        )


if __name__ == "__main__":
    main()
