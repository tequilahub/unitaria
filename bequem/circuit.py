import tequila as tq
import numpy as np
from dataclasses import dataclass


@dataclass
class Circuit:
    circuit: tq.QCircuit

    def simulate(self, input: str | None=None) -> np.array:
        raise NotImplementedError
