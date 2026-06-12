import numpy as np
import pytest

from multiple_minimum_monte_carlo.calculation import HARTREE_TO_KCAL, XTBCalculation

# Stand-in for the xtb binary: records its arguments and any xcontrol input to
# FAKE_XTB_LOG, then mimics xtb's outputs (xtbopt.xyz for --opt runs, the
# TOTAL ENERGY stdout line for single points).
FAKE_XTB = """#!/bin/sh
if [ -n "$FAKE_XTB_LOG" ]; then
    echo "$@" >> "$FAKE_XTB_LOG"
    if [ -f xtb.inp ]; then cat xtb.inp >> "$FAKE_XTB_LOG"; fi
fi
case " $* " in
    *" --opt "*)
        cat > xtbopt.xyz <<EOF
2
 energy: -1.500000000000 gnorm: 0.000337685554 xtb: 6.7.1 (fake)
H         0.0000000000    0.0000000000    0.1000000000
H         0.0000000000    0.0000000000    0.8500000000
EOF
        ;;
    *)
        echo "          | TOTAL ENERGY              -1.500000000000 Eh   |"
        ;;
esac
"""

FAILING_XTB = """#!/bin/sh
echo "[ERROR] Program stopped due to fatal error"
exit 1
"""


class DummyAtoms:
    def get_chemical_symbols(self):
        return ["H", "H"]

    def get_positions(self):
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    script = tmp_path / "xtb"
    script.write_text(FAKE_XTB)
    script.chmod(0o755)
    log = tmp_path / "xtb_calls.log"
    monkeypatch.setenv("FAKE_XTB_LOG", str(log))
    return script, log


def test_run_parses_positions_and_energy(fake_xtb):
    script, log = fake_xtb
    calc = XTBCalculation(xtb_path=str(script))
    positions, energy = calc.run(DummyAtoms())
    assert positions.shape == (2, 3)
    assert positions[1, 2] == pytest.approx(0.85)
    assert energy == pytest.approx(-1.5 * HARTREE_TO_KCAL)
    args = log.read_text()
    assert "--chrg 0" in args
    assert "--uhf 0" in args
    assert "--gfn 2" in args
    assert "--opt normal" in args


def test_run_passes_charge_spin_and_options(fake_xtb):
    script, log = fake_xtb
    calc = XTBCalculation(
        charge=-1,
        spin_multiplicity=3,
        method="gfnff",
        opt_level="tight",
        solvent="water",
        max_cycles=50,
        xtb_path=str(script),
    )
    calc.run(DummyAtoms())
    args = log.read_text()
    assert "--chrg -1" in args
    assert "--uhf 2" in args
    assert "--gfnff" in args
    assert "--opt tight" in args
    assert "--alpb water" in args
    assert "--cycles 50" in args


def test_run_writes_constraint_input(fake_xtb):
    script, log = fake_xtb
    calc = XTBCalculation(xtb_path=str(script))
    calc.run(DummyAtoms(), constrained_atoms=[0])
    contents = log.read_text()
    assert "--input xtb.inp" in contents
    # 0-based indices must be converted to xtb's 1-based numbering
    assert "atoms: 1" in contents
    assert "engine=lbfgs" in contents


def test_energy_parses_total_energy(fake_xtb):
    script, log = fake_xtb
    calc = XTBCalculation(xtb_path=str(script))
    energy = calc.energy(DummyAtoms())
    assert energy == pytest.approx(-1.5 * HARTREE_TO_KCAL)
    assert "--opt" not in log.read_text()


def test_failed_run_raises(tmp_path):
    script = tmp_path / "xtb"
    script.write_text(FAILING_XTB)
    script.chmod(0o755)
    calc = XTBCalculation(xtb_path=str(script))
    with pytest.raises(RuntimeError, match="fatal error"):
        calc.run(DummyAtoms())


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="gfn3"):
        XTBCalculation(method="gfn3")
