import tequila as tq
import unitaria as ut


def create_block_encoding():
    # initialize a block encoding with a simple X gate on the first qubit
    subspace_in = ut.Subspace(bits=1, zero_qubits=1)
    subspace_out = ut.Subspace(bits=2)
    tequila_circuit = tq.QCircuit()
    tequila_circuit += tq.gates.X(0)
    tequila_circuit.n_qubits = 2
    circuit = ut.Circuit(tequila_circuit)
    normalization = 1.0
    return ut.BlockEncoding(circuit, subspace_in, subspace_out, normalization)


def test_block_encoding_x_gate_on_first_qubit():
    block_encoding = create_block_encoding()
    ut.verify(block_encoding)
