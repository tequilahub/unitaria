from __future__ import annotations
import tequila as tq
import numpy as np
from dataclasses import dataclass

from tequila import BitNumbering


@dataclass
class Circuit:
    tq_circuit: tq.QCircuit

    def __init__(self, tq_circuit: tq.QCircuit | None = None):
        if tq_circuit is not None:
            self.tq_circuit = tq_circuit
        else:
            self.tq_circuit = tq.QCircuit()

    def simulate(self, input: np.ndarray | int = 0, **kwargs) -> np.ndarray:
        if len(self.tq_circuit.qubits) == 0:
            if isinstance(input, np.ndarray):
                return input
            else:
                assert isinstance(input, (int, np.integer))
                result = np.zeros(2 ** self.tq_circuit.n_qubits)
                result[input] = 1
                return result
        if isinstance(input, np.ndarray):
            input = tq.QubitWaveFunction.from_array(input, BitNumbering.LSB)
        elif isinstance(input, (int, np.integer)):
            input = tq.QubitWaveFunction.from_basis_state(self.tq_circuit.n_qubits, input, BitNumbering.LSB)

        padded = self.padded()

        result = tq.simulate(padded, initial_state=input, **kwargs)
        return result.to_array(BitNumbering.LSB, copy=False)

    # TODO: This function is necessary because tequila has problems with unused qubits
    def padded(self) -> tq.QCircuit:
        copy = tq.QCircuit(gates=self.tq_circuit.gates.copy())
        for bit in range(self.tq_circuit.n_qubits):
            if bit not in self.tq_circuit.qubits:
                copy += tq.gates.Phase(bit, angle=0)
        return copy

    def __add__(self, other):
        return self.tq_circuit + other.tq_circuit

    def __iadd__(self, other):
        self.tq_circuit += other.tq_circuit
        return self

    def adjoint(self):
        # TODO: this should maybe be included in tequila
        adj = self.tq_circuit.dagger()
        adj.n_qubits = self.tq_circuit.n_qubits
        return Circuit(adj)

    @staticmethod
    def from_qiskit(circuit):
        from qiskit.qasm2 import dumps
        qasm = dumps(circuit)
        qasm = qasm.replace("\nu(", "\nU(")
        tq_circuit = tq.import_open_qasm(qasm)
        if circuit.num_qubits > 0:
            tq_circuit.n_qubits = circuit.num_qubits
        return Circuit(tq_circuit)
