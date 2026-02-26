from unitaria.nodes.block_encoding import BlockEncoding
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit
import tequila as tq
from unitaria.verifier import verify


def create_block_encoding():
    # initialize a block encoding with a simple X gate on the first qubit
    subspace_in = Subspace(bits=1, zero_qubits=1)
    subspace_out = Subspace(bits=2)
    tequila_circuit = tq.QCircuit()
    tequila_circuit += tq.gates.X(0)
    tequila_circuit.n_qubits = 2
    circuit = Circuit(tequila_circuit)
    normalization = 1.0
    return BlockEncoding(circuit, subspace_in, subspace_out, normalization)


def test_block_encoding_x_gate_on_first_qubit():
    block_encoding = create_block_encoding()
    verify(block_encoding)
