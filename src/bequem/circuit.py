from __future__ import annotations
import tequila as tq
import numpy as np
from dataclasses import dataclass

from tequila import BitNumbering


@dataclass
class Circuit:

    def __init__(self, tq_circuit: tq.QCircuit | None = None):
        if tq_circuit is not None:
            self.tq_circuit = tq_circuit
        else:
            self.tq_circuit = tq.QCircuit()

    def simulate(self, input: np.array | None = None, **kwargs) -> np.array:
        input = tq.QubitWaveFunction.from_array(input, BitNumbering.LSB)
        result = tq.simulate(self.tq_circuit, initial_state=input, **kwargs)
        return result.to_array(BitNumbering.LSB, copy=False)

    def __add__(self, other):
        return self.tq_circuit + other.tq_circuit

    def __iadd__(self, other):
        self.tq_circuit += other.tq_circuit
        return self
