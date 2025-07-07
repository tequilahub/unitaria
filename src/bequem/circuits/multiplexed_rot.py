from typing import Sequence

import numpy as np
import numpy.typing as npt
import tequila as tq


def multiplexed_Ry(
    angles: npt.NDArray[float], target: int, controls: Sequence[int], top_level: bool = True, flipped: bool = False
) -> tq.QCircuit:
    """
    Implements a multiplexed Y-rotation.
    Depending on the state |k> of the control qubits, a rotation with angle angles[k] is applied to the target qubit.
    Requires len(angles) == 2 ** len(controls).
    Reference: https://arxiv.org/abs/quant-ph/0406176, chaper 3

    :param angles: The angles for the rotations.
    :param target: The target qubit.
    :param controls: The control qubits in MSB ordering.
    :param top_level: Used internally to determine if the call is at the first recursion level, should not be used externally.
    :param flipped: Used internally to selectively change the order of the rotations, should not be used externally.
    :return: A circuit implementing the multiplexed rotation.
    """

    assert len(angles) == 2 ** len(controls)

    if len(controls) == 0:
        return tq.gates.Ry(angle=angles[0], target=target)

    (angles0, angles1) = np.split(angles, 2)
    U = tq.QCircuit()
    if flipped:
        U += multiplexed_Ry((angles0 - angles1) / 2, target, controls[1:], False, False)
        U += tq.gates.CNOT(control=controls[0], target=target)
    U += multiplexed_Ry((angles0 + angles1) / 2, target, controls[1:], False, flipped)
    if not flipped:
        U += tq.gates.CNOT(control=controls[0], target=target)
        U += multiplexed_Ry((angles0 - angles1) / 2, target, controls[1:], False, True)
    if top_level:
        U += tq.gates.CNOT(control=controls[0], target=target)
    return U


def multiplexed_Rz(
    angles: npt.NDArray[float], target: int, controls: Sequence[int], top_level: bool = True, flipped: bool = False
) -> tq.QCircuit:
    """
    Implements a multiplexed Z-rotation.
    Depending on the state |k> of the control qubits, a rotation with angle angles[k] is applied to the target qubit.
    Requires len(angles) == 2 ** len(controls).
    Reference: https://arxiv.org/abs/quant-ph/0406176, chaper 3

    :param angles: The angles for the rotations.
    :param target: The target qubit.
    :param controls: The control qubits in MSB ordering.
    :param flipped: Used internally to selectively change the order of the rotations, should not be used externally.
    :return: A circuit implementing the multiplexed rotation.
    :return: A circuit implementing the multiplexed rotation.
    """

    assert len(angles) == 2 ** len(controls)

    if len(controls) == 0:
        return tq.gates.Rz(angle=angles[0], target=target)

    (angles0, angles1) = np.split(angles, 2)
    U = tq.QCircuit()
    if flipped:
        U += multiplexed_Rz((angles0 - angles1) / 2, target, controls[1:], False, False)
        U += tq.gates.CNOT(control=controls[0], target=target)
    U += multiplexed_Rz((angles0 + angles1) / 2, target, controls[1:], False, flipped)
    if not flipped:
        U += tq.gates.CNOT(control=controls[0], target=target)
        U += multiplexed_Rz((angles0 - angles1) / 2, target, controls[1:], False, True)
    if top_level:
        U += tq.gates.CNOT(control=controls[0], target=target)
    return U
