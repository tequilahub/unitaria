import numpy as np
import scipy
import unitaria as ut


def test_1d_gaussian_conv():
    kernel = np.exp(-((np.arange(-3, 4) / 4) ** 2))
    padded_kernel = np.append(kernel, 0)

    prep = ut.Identity(ut.Subspace.from_dim(16)) & ut.ConstantVector(np.sqrt(padded_kernel))
    add = ut.IntegerAddition(source_bits=3, target_bits=4)
    const_add = ut.ConstantIntegerAddition(bits=4, constant=-3) & ut.Identity(ut.Subspace.from_dim(8))
    unprep = ut.Adjoint(prep)

    conv = (unprep @ const_add @ add @ prep)[:8, :8]
    ut.verify(conv)

    input = np.linspace(0.0, 1.0, 8)
    input /= np.linalg.norm(input)
    result = conv.compute(input)
    expected = scipy.signal.convolve(input, kernel, mode="same")
    assert np.allclose(result, expected)


def test_2d_gaussian_conv():
    kernel = np.exp(-((np.arange(-1, 2) / 4) ** 2))
    padded_kernel = np.append(kernel, 0)

    prep = ut.Identity(ut.Subspace.from_dim(8)) & ut.ConstantVector(np.sqrt(padded_kernel))
    add = ut.IntegerAddition(source_bits=2, target_bits=3)
    const_add = ut.ConstantIntegerAddition(bits=3, constant=-1) & ut.Identity(ut.Subspace.from_dim(4))
    unprep = ut.Adjoint(prep)
    one_dim_conv = (unprep @ const_add @ add @ prep)[:4, :4]

    two_dim_conv = one_dim_conv & one_dim_conv
    print(f"qubits: {two_dim_conv.circuit().n_qubits}")
    ut.verify(two_dim_conv)

    input = np.outer(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))
    input /= np.linalg.norm(input)
    result = two_dim_conv.compute(input.flatten()).reshape((4, 4))
    expected = scipy.signal.convolve(input, np.outer(kernel, kernel), mode="same")
    assert np.allclose(result, expected)
