import qiskit as qk

@dataclass
class Circuit:
    circuit: qk.QuantumCircuit

    def simulate(self, input: str | None=None) -> np.array:
        raise NotImplementedError
