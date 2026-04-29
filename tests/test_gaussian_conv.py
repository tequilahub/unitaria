import numpy as np
import scipy
import unitaria as ut


def test_1d_gaussian_conv():
    x = np.arange(-3, 4) / 2
    gaussian = np.exp(-(x**2))
    prep = ut.ConstantVector(np.append([0], np.sqrt(gaussian)))
    conv = (
        (ut.Identity(ut.Subspace.from_dim(16)) & ut.Adjoint(prep))
        @ (ut.ConstantIntegerAddition(bits=4, constant=-4) & ut.Identity(ut.Subspace.from_dim(8)))
        @ ut.IntegerAddition(source_bits=3, target_bits=4)
        @ (ut.Identity(ut.Subspace.from_dim(16)) & prep)
    )[:8, :8]
    ut.verify(conv)

    input = np.linspace(0.0, 1.0, 8)
    input /= np.linalg.norm(input)
    result = conv.compute(input)
    expected = scipy.signal.convolve(input, gaussian, mode="same")
    assert np.allclose(result, expected)


def test_2d_gaussian_conv():
    x = np.arange(-1, 2)
    gaussian = np.exp(-(x**2))
    prep = ut.ConstantVector(np.append([0], np.sqrt(gaussian)))

    one_dim_conv = (
        (ut.Identity(ut.Subspace.from_dim(8)) & ut.Adjoint(prep))
        @ (ut.ConstantIntegerAddition(bits=3, constant=-2) & ut.Identity(ut.Subspace.from_dim(4)))
        @ ut.IntegerAddition(source_bits=2, target_bits=3)
        @ (ut.Identity(ut.Subspace.from_dim(8)) & prep)
    )[:4, :4]

    two_dim_conv = one_dim_conv & one_dim_conv
    print(f"qubits: {two_dim_conv.circuit().n_qubits}")
    ut.verify(two_dim_conv)

    input = np.outer(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))
    input /= np.linalg.norm(input)
    result = two_dim_conv.compute(input.flatten()).reshape((4, 4))
    expected = scipy.signal.convolve(input, np.outer(gaussian, gaussian), mode="same")
    assert np.allclose(result, expected)
