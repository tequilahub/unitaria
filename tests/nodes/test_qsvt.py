import numpy as np
import unitaria as ut


def test_qsvt_coefficients():
    # Identity
    c = ut.QSVTCoefficients(np.array([0, 0]), "angles")
    np.testing.assert_allclose(c.angles, np.array([0, 0]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    c = ut.QSVTCoefficients(np.array([0, 1]), "chebyshev")
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1

    # Scaled identity
    c = ut.QSVTCoefficients(np.array([0, 2]), "chebyshev")
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 2]))
    assert c.output_normalization == 2

    # Amplitude amplificiation
    c = ut.QSVTCoefficients(np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2]), "angles")
    np.testing.assert_allclose(c.angles, np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    c = ut.QSVTCoefficients(np.array([0, 0, 0, 1]), "chebyshev")
    b = ut.QSVTCoefficients(c.angles, "angles")
    np.testing.assert_allclose(b.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)


def test_qsvt_grover():
    # Define v so it has some subnormalization
    v = 0.45 * (
        ut.Projection(ut.Subspace("#"), ut.Subspace("0"))
        @ ut.ConstantUnitary(2 ** (-1 / 2) * np.array([[1, 1], [1, -1]]))
        @ ut.Projection(ut.Subspace("0"), ut.Subspace("#"))
    )
    A = ut.QSVT(v, np.array(4 * [np.pi]))
    ut.verify(A)


def test_qsvt_with_ancillas():
    # Define v so it has some subnormalization
    A = 9.0 * ut.Identity((ut.Subspace("#") | ut.Subspace("0")) & ut.Subspace("#"))
    assert A.subspace_in.total_qubits + A.subspace_in.clean_ancilla_count() > 2
    B = ut.QSVT(A, np.array(4 * [np.pi]))
    ut.verify(B)


def test_qsvt_with_polynomial():
    A = 1.23 * ut.Increment(bits=2)
    B = ut.QSVT(A, np.polynomial.Chebyshev([1, 0, 1], domain=[-1.23, 1.23]))
    ut.verify(B)
