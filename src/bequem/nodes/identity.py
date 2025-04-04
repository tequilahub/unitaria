import numpy as np
import tequila as tq

from bequem.circuit import Circuit
from bequem.qubit_map import QubitMap
from .node import Node


class Identity(Node):

    def __init__(self, qubits: QubitMap):
        self.qubits = qubits
    
    def qubits_in(self) -> QubitMap:
        return self.qubits

    def qubits_out(self) -> QubitMap:
        return self.qubits

    def normalization(self) -> float:
        return 1

    def compute(self, input: np.array) -> np.array:
        return input

    def circuit(self) -> Circuit:
        return Circuit(tq.QCircuit())
