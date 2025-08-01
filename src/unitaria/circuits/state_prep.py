from typing import Sequence
import numpy as np
import numpy.typing as npt
import tequila as tq

from unitaria.circuits.multiplexed_rot import multiplexed_Ry, multiplexed_Rz


def prepare_state(state: npt.NDArray[complex], target: Sequence[int]) -> tq.QCircuit:
    """
    Implements a circuit that prepares an arbitrary state.
    Reference: https://arxiv.org/abs/quant-ph/0406176, chapter 4

    :param state: The state to be prepared.
    :param target: Indices of the target qubits in MSB ordering.
    Can be in any state and will be returned to this state by the end of the circuit.
    :return: A circuit implementing the state preparation.
    """
    n = len(target)
    assert state.shape == (2**n,)
    assert np.isclose(np.linalg.norm(state), 1)
    state = state.astype(complex)

    theta = dict()
    phi = dict()
    combined = dict()
    for bit in reversed(range(n)):
        for i in range(2**bit):
            a0 = state[2 * i] if bit == n - 1 else combined[bit + 1, 2 * i]
            a1 = state[2 * i + 1] if bit == n - 1 else combined[bit + 1, 2 * i + 1]
            r = np.hypot(np.abs(a0), np.abs(a1))
            theta[bit, i] = 2 * np.arccos(np.abs(a0) / r) if r != 0 else 0
            phi[bit, i] = np.angle(a1) - np.angle(a0)
            combined[bit, i] = np.exp(((np.angle(a0) + np.angle(a1)) / 2) * 1j) * r

    U = tq.QCircuit()

    for bit in range(n):
        U += multiplexed_Ry(np.array([theta[bit, i] for i in range(2**bit)]), target=target[bit], controls=target[:bit])
        if not np.allclose(np.array([phi[bit, i] for i in range(2**bit)]), 0.0):
            U += multiplexed_Rz(
                np.array([phi[bit, i] for i in range(2**bit)]), target=target[bit], controls=target[:bit]
            )

    U += tq.gates.GlobalPhase(angle=np.angle(combined[0, 0]))

    return U
