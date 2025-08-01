import numpy as np
import numpy.typing as npt
from typing import Sequence
import tequila as tq
import scipy

from unitaria.circuits.multiplexed_rot import multiplexed_Rz, multiplexed_Ry


def generic_unitary(U: npt.NDArray[complex], target: Sequence[int]) -> tq.QCircuit:
    """
    Constructs a circuit that implements any unitary matrix.
    Reference: https://arxiv.org/abs/quant-ph/0406176, chapter 5

    :param U: The unitary to be implemented.
    :param target: The target qubits in MSB ordering.
    :return: A circuit implementing the unitary.
    """
    n = len(target)
    assert U.shape == (2**n, 2**n)
    U = U.astype(complex)

    if n == 0:
        return tq.gates.GlobalPhase(angle=np.angle(U[0, 0]))

    (a1, b1), theta, (a2, b2) = scipy.linalg.cossin(U, p=2 ** (n - 1), q=2 ** (n - 1), separate=True)

    circuit = tq.QCircuit()
    if n > 1:
        circuit += _multiplexed_unitary(U1=a2, U2=b2, target=target[1:], control=target[0])
        circuit += multiplexed_Ry(angles=2 * theta, target=target[0], controls=target[1:])
        circuit += _multiplexed_unitary(U1=a1, U2=b1, target=target[1:], control=target[0])
    else:
        a1, b1, a2, b2 = np.angle((a1[0, 0], b1[0, 0], a2[0, 0], b2[0, 0]))
        circuit += tq.gates.Rz(angle=b2 - a2, target=target[0])
        circuit += tq.gates.GlobalPhase((b2 + a2) / 2)
        circuit += tq.gates.Ry(angle=2 * theta[0], target=target[0])
        circuit += tq.gates.Rz(angle=b1 - a1, target=target[0])
        circuit += tq.gates.GlobalPhase((b1 + a1) / 2)
    return circuit


def _multiplexed_unitary(
    U1: npt.NDArray[complex], U2: npt.NDArray[complex], target: Sequence[int], control: int
) -> tq.QCircuit():
    eigenvalues, eigenvectors = np.linalg.eig(U1 @ U2.T.conjugate())
    V = eigenvectors
    D = np.diag(np.sqrt(eigenvalues))
    W = D @ V.T.conjugate() @ U2

    circuit = tq.QCircuit()
    circuit += generic_unitary(U=W, target=target)
    circuit += multiplexed_Rz(angles=-2 * np.angle(D.diagonal()), target=control, controls=target)
    circuit += generic_unitary(U=V, target=target)
    return circuit
