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
import shutil
import sys
from copy import copy

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
        "--model",
        choices=sorted(MODELS),
        default="aimnet2",
        help="Energy model to use (default: aimnet2). Each backend needs its own "
        "package: aimnet2 (aimnet), mace-off (mace-torch), ani2x (torchani), "
        "xtb (tblite), xtb-cli (the xtb executable on PATH), uma (fairchem-core)",
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
        default="normal",
        help="Optimization convergence level for the xtb-cli model "
        "(default: normal); other models use --fmax",
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
        help="CPUs for parallel mode; 0 uses all available (default: 0)",
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

    console.print(
        f"[bold]Molecular formula:[/] [cyan]{conformer.atoms.get_chemical_formula()}[/]"
    )
    console.print(f"[bold]Rotatable torsions:[/] {len(dihedrals)}")
    console.print(f"[bold]Estimated conformer space (3^N):[/] {3 ** len(dihedrals)}")

    if args.model == "xtb-cli":
        if shutil.which("xtb") is None:
            sys.exit(
                "xtb executable not found on PATH.\nInstall it with: "
                "conda install -c conda-forge xtb"
            )
        device = "cpu"
        calc = XTBCalculation(
            charge=args.charge,
            spin_multiplicity=args.spin_multiplicity,
            opt_level=args.opt_level,
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

    def report_step(steps_done, initial_positions, results, accepted):
        nonlocal global_min, stopped_early
        acceptance_history.extend(accepted)
        overall_rate = sum(acceptance_history) / len(acceptance_history)
        last_10 = acceptance_history[-10:]
        last_10_rate = sum(last_10) / len(last_10)
        if args.quiet:
            global_min = min(global_min, *(energy for _, energy in results))
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
                new_minimum = energy < global_min
                if new_minimum:
                    global_min = energy
                guess = get_dihedral_angles(conformer.mol, dihedrals, initial)
                final = get_dihedral_angles(conformer.mol, dihedrals, optimized)
                if accepted[i]:
                    verdict = "[green]accepted[/]"
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
                    console.print("  [bold yellow]*** new global minimum ***[/]")
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
        max_bonds_rotate=args.max_bonds_rotate,
        angle_step=args.angle_step,
        initial_optimization=False,
        parallel=args.parallel,
        num_cpus=args.num_cpus,
        verbose=args.verbose,
        step_callback=report_step,
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


if __name__ == "__main__":
    main()
