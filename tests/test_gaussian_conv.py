import numpy as np
import scipy

from unitaria.nodes import (
    ConstantVector,
    IntegerAddition,
    Identity,
    Adjoint,
    ConstantIntegerAddition,
    Mul,
)
from unitaria.verifier import verify
from unitaria import Subspace
from unitaria.nodes.basic.projection import Projection


def test_1d_gaussian_conv():
    x = np.arange(-3, 4) / 2
    gaussian = np.exp(-(x**2))
    prep = ConstantVector(np.append([0], np.sqrt(gaussian)))
    conv = (
        Projection(subspace_from=Subspace(4), subspace_to=Subspace(3, 1))
        @ (Adjoint(prep) & Identity(4))
        @ (Identity(3) & ConstantIntegerAddition(bits=4, constant=-4))
        @ IntegerAddition(source_bits=3, target_bits=4)
        @ (prep & Identity(4))
        @ Projection(subspace_from=Subspace(3, 1), subspace_to=Subspace(4))
    )
    verify(conv)

    input = np.linspace(0.0, 1.0, 8)
    input /= np.linalg.norm(input)
    result = conv.compute(input)
    expected = scipy.signal.convolve(input, gaussian, mode="same")
    assert np.allclose(result, expected)


def test_2d_gaussian_conv():
    x = np.arange(-1, 2)
    gaussian = np.exp(-(x**2))
    prep = ConstantVector(np.append([0], np.sqrt(gaussian)))

    # TODO: It should be possible to write this more neatly using the @ operator in the future,
    # but currently this increases the number of qubits and significantly slows down the test
    one_dim_conv = Mul(
        Mul(
            Mul(
                Projection(subspace_from=Subspace(2, 1), subspace_to=Subspace(3)),
                prep & Identity(3),
                skip_projection=True,
            ),
            IntegerAddition(source_bits=2, target_bits=3),
            skip_projection=True,
        ),
        Mul(
            Mul(
                Identity(2) & ConstantIntegerAddition(bits=3, constant=-2),
                Adjoint(prep) & Identity(3),
                skip_projection=True,
            ),
            Projection(subspace_from=Subspace(3), subspace_to=Subspace(2, 1)),
            skip_projection=True,
        ),
        skip_projection=True,
    )
    two_dim_conv = one_dim_conv & one_dim_conv
    verify(two_dim_conv)

    input = np.outer(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))
    input /= np.linalg.norm(input)
    result = two_dim_conv.compute(input.flatten()).reshape((4, 4))
    expected = scipy.signal.convolve(input, np.outer(gaussian, gaussian), mode="same")
    assert np.allclose(result, expected)
