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


@dataclass
class Circuit:
    """
    Representation of a quantum circuit.

    This is just a wrapper around the Tequila
    :external:py:class:`~tequila.circuit.circuit.QCircuit` class.

    :param tq_circuit:
        The representation of the circuit for the tequila backend.
    """

    tq_circuit: tq.QCircuit

    def __init__(self, tq_circuit: tq.QCircuit | None = None):
        if tq_circuit is not None:
            self.tq_circuit = tq_circuit
        else:
            self.tq_circuit = tq.QCircuit()

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
        if len(self.tq_circuit.qubits) == 0:
            if isinstance(input, np.ndarray):
                return input
            else:
                assert isinstance(input, (int, np.integer))
                result = np.zeros(2**self.tq_circuit.n_qubits)
                result[input] = 1
                return result
        if isinstance(input, np.ndarray):
            input = tq.QubitWaveFunction.from_array(input, BitNumbering.LSB)
        elif isinstance(input, (int, np.integer)):
            input = tq.QubitWaveFunction.from_basis_state(self.tq_circuit.n_qubits, input, BitNumbering.LSB)

        padded = self._padded()

        result = tq.simulate(padded, initial_state=input, **kwargs)
        return result.to_array(BitNumbering.LSB, copy=False)

    # TODO: This function is necessary because tequila has problems with unused qubits
    def _padded(self) -> tq.QCircuit:
        copy = tq.QCircuit(gates=self.tq_circuit.gates.copy())
        for bit in range(self.tq_circuit.n_qubits):
            if bit not in self.tq_circuit.qubits:
                copy += tq.gates.Phase(bit, angle=0)
        return copy

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __iadd__(self, other):
        if isinstance(other, Circuit):
            self.tq_circuit += other.tq_circuit
        elif isinstance(other, tq.QCircuit):
            self.tq_circuit += other
        else:
            raise TypeError(f"Cannot add {type(other)} to Circuit")
        return self

    def adjoint(self) -> Circuit:
        """
        Gives the inverse circuit (corresponding to the adjoint unitary).
        """
        adj = self.tq_circuit.dagger()
        # TODO: this should maybe be included in tequila
        adj.n_qubits = self.tq_circuit.n_qubits
        return Circuit(adj)

    def add_controls(self, controls):
        return Circuit(self.tq_circuit.add_controls(control=controls))

    def map_qubits(self, map):
        return Circuit(self.tq_circuit.map_qubits(map))

    def draw(self) -> str:
        """
        Draw this circuit and return a string representation.

        If qpic is installed, this will generate a temporary file containing a
        pdf of the circuit and return a ``file://`` url to the pdf, which should
        be printed to the user.
        """
        if tq.circuit.qpic.system_has_qpic:
            # TODO: Use IPython if available
            _handle, file = tempfile.mkstemp(suffix=".pdf")
            tq.circuit.qpic.export_to(self.tq_circuit, file, always_use_generators=True)
            return f"Circuit stored at file://{file}"
        else:
            return self.tq_circuit.__str__()
