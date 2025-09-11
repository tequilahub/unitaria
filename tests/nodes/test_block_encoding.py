import numpy as np
from unitaria.nodes.block_encoding import BlockEncoding
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit
import tequila as tq


def create_block_encoding():
    # initialize a block encoding with a simple X gate on the first qubit
    subspace_in = Subspace(2)
    subspace_out = Subspace(2)
    tequila_circuit = tq.QCircuit()
    tequila_circuit += tq.gates.X(0)
    tequila_circuit.n_qubits = 2
    circuit = Circuit(tequila_circuit)
    normalization = 1.0
    return BlockEncoding(circuit, subspace_in, subspace_out, normalization)


def test_block_encoding_x_gate_on_first_qubit():
    block_encoding = create_block_encoding()
    input_vec = np.array([1, 0, 0, 0], dtype=np.complex128)
    expected_output = np.array([0, 1, 0, 0], dtype=np.complex128)
    expected_output_adj = np.array([0, 1, 0, 0], dtype=np.complex128)

    output = block_encoding.compute(input_vec)
    output_adj = block_encoding.compute_adjoint(input_vec)

    assert np.array_equal(output, expected_output), "Output does not match expected result after X gate."
    assert np.array_equal(output_adj, expected_output_adj), (
        "Adjoint output does not match expected result after X gate."
    )
