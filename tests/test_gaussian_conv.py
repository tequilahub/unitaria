import numpy as np
import scipy
import unitaria as ut


def test_1d_gaussian_conv():
    x = np.arange(-3, 4) / 2
    gaussian = np.exp(-(x**2))
    prep = ut.ConstantVector(np.append([0], np.sqrt(gaussian)))
    conv = (
        ut.Projection(subspace_from=ut.Subspace("####"), subspace_to=ut.Subspace("0###"))
        @ (ut.Identity(ut.Subspace("####")) & ut.Adjoint(prep))
        @ (ut.ConstantIntegerAddition(bits=4, constant=-4) & ut.Identity(ut.Subspace("0###")))
        @ ut.IntegerAddition(source_bits=3, target_bits=4)
        @ (ut.Identity(ut.Subspace("####")) & prep)
        @ ut.Projection(subspace_from=ut.Subspace("0###"), subspace_to=ut.Subspace("####"))
    )
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

    # TODO: It should be possible to write this more neatly using the @ operator in the future,
    # but currently this increases the number of qubits and significantly slows down the test
    one_dim_conv = ut.Mul(
        ut.Mul(
            ut.Projection(subspace_from=ut.Subspace("###"), subspace_to=ut.Subspace("0##")),
            ut.Mul(
                ut.Identity(ut.Subspace("###")) & ut.Adjoint(prep),
                ut.ConstantIntegerAddition(bits=3, constant=-2) & ut.Identity(ut.Subspace("##")),
                skip_projection=True,
            ),
            skip_projection=True,
        ),
        ut.Mul(
            ut.IntegerAddition(source_bits=2, target_bits=3),
            ut.Mul(
                ut.Identity(ut.Subspace("###")) & prep,
                ut.Projection(subspace_from=ut.Subspace("0##"), subspace_to=ut.Subspace("###")),
                skip_projection=True,
            ),
            skip_projection=True,
        ),
        skip_projection=True,
    )
    two_dim_conv = one_dim_conv & one_dim_conv
    ut.verify(two_dim_conv)

    input = np.outer(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))
    input /= np.linalg.norm(input)
    result = two_dim_conv.compute(input.flatten()).reshape((4, 4))
    expected = scipy.signal.convolve(input, np.outer(gaussian, gaussian), mode="same")
    assert np.allclose(result, expected)
