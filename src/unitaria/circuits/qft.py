import tequila as tq


def qft_circuit(size: int) -> tq.QCircuit():
    U = tq.QCircuit()

    for i in reversed(range(size)):
        U += tq.gates.H(target=i)
        for j in range(i):
            U += tq.gates.Z(target=i, control=j, power=2 ** (j - i))

    for i in range(size // 2):
        U += tq.gates.SWAP(i, size - i - 1)

    return U
