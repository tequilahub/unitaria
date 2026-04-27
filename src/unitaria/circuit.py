"""
Representation quantum circuits.
"""

from __future__ import annotations

import copy
import tempfile

import tequila as tq
import numpy as np
from dataclasses import dataclass

from tequila import BitNumbering

from unitaria.util import is_ipython


@dataclass
class Circuit:
    """
    Representation of a quantum circuit.

    This is just a wrapper around the Tequila
    :external:py:class:`~tequila.circuit.circuit.QCircuit` class.

    :param tq_circuit:
        The representation of the circuit for the tequila backend.
    """

    _tq_circuit: tq.QCircuit
    # TODO: This could be removed if tequila handles the n_qubits = 0 case properly
    n_qubits: int = 0

    def __init__(self, tq_circuit: tq.QCircuit | None = None):
        if tq_circuit is not None:
            self._tq_circuit = tq_circuit
        else:
            self._tq_circuit = tq.QCircuit()

    def simulate(self, input: np.ndarray | int = 0, **kwargs) -> np.ndarray:
        """
        Simulate this circuit. For additional arguments see
        :external:py:func:`~tequila.simulators.simulator_api.simulate`.

        :param input:
            The initial state from which the circuit should be simulated.
            If ``input`` is a vector, it will be interpreted as amplitudes of
            the computational basis states and its dimension should be ``2 **
            n_qubits``. If it is an integer ``i``, it will be interpreted as the
            ``i``-th computational basis state.
        """
        if len(self._tq_circuit.qubits) == 0:
            if isinstance(input, np.ndarray):
                return input
            else:
                assert isinstance(input, (int, np.integer))
                result = np.zeros(2**self.n_qubits, dtype=complex)
                result[input] = 1
                return result
        if isinstance(input, np.ndarray):
            input = tq.QubitWaveFunction.from_array(input, BitNumbering.LSB)
        elif isinstance(input, (int, np.integer)):
            input = tq.QubitWaveFunction.from_basis_state(max(1, self.n_qubits), input, BitNumbering.LSB)

        padded = self._padded()

        result = tq.simulate(padded, initial_state=input, **kwargs)
        return result.to_array(BitNumbering.LSB, copy=False)

    # TODO: This function is necessary because tequila has problems with unused qubits
    def _padded(self) -> tq.QCircuit:
        copy = tq.QCircuit(gates=self._tq_circuit.gates.copy())
        for bit in range(max(1, self.n_qubits)):
            if bit not in self._tq_circuit.qubits:
                copy += tq.gates.I(bit)
        return copy

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __iadd__(self, other):
        if isinstance(other, Circuit):
            self._tq_circuit += other._tq_circuit
        elif isinstance(other, tq.QCircuit):
            self._tq_circuit += other
        else:
            raise TypeError(f"Cannot add {type(other)} to Circuit")
        return self

    def adjoint(self) -> Circuit:
        """
        Gives the inverse circuit (corresponding to the adjoint unitary).
        """
        adj = Circuit(self._tq_circuit.dagger())
        # TODO: this should maybe be included in tequila
        adj.n_qubits = self.n_qubits
        return adj

    def add_controls(self, controls):
        return Circuit(self._tq_circuit.add_controls(control=controls))

    def map_qubits(self, map):
        return Circuit(self._tq_circuit.map_qubits(map))

    def depth(self) -> int:
        return self._tq_circuit.depth

    def _remove_global_phase(self) -> tuple[Circuit, float]:
        tq_circuit = tq.QCircuit()
        global_phase = 0
        for gate in self._tq_circuit.gates:
            if isinstance(gate, tq.circuit._gates_impl.GlobalPhaseGateImpl):
                global_phase += gate.parameter
            elif (
                isinstance(gate, tq.circuit._gates_impl.RotationGateImpl | tq.circuit._gates_impl.PhaseGateImpl)
                and gate.parameter == 0
            ):
                pass
            else:
                tq_circuit += gate
        result = Circuit(tq_circuit)
        result.n_qubits = self.n_qubits
        return result, global_phase

    def draw(self) -> str:
        """
        Draw this circuit and return a string representation.

        If qpic is installed, this will generate a temporary file containing a
        pdf of the circuit and return a ``file://`` url to the pdf, which should
        be printed to the user.
        """
        if tq.circuit.qpic.system_has_qpic:
            _handle, file = tempfile.mkstemp(suffix=".pdf")
            circuit, _global_phase = self._remove_global_phase()
            tq.circuit.qpic.export_to(circuit._padded(), file, style="standard")
            if is_ipython():
                import IPython
                import subprocess

                png_file = file[:-4] + ".png"

                subprocess.run(
                    ["gs", "-dSAFER", "-r200", "-sDEVICE=pngalpha", "-o", png_file, file], stdout=subprocess.DEVNULL
                )

                with open(png_file, "rb") as f:
                    data = f.read().rstrip()
                    image = IPython.display.Image(data, unconfined=True)
                    IPython.display.display(image)
            else:
                return f"Circuit stored at file://{file}"
        else:
            return self._tq_circuit.__str__()
