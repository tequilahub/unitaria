import numpy as np
import unitaria as ut


def test_qsvt_coefficients():
    # Identity
    c = ut.QSVTCoefficients(np.array([0, 0]), "angles", 1)
    np.testing.assert_allclose(c.angles, np.array([0, 0]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    c = ut.QSVTCoefficients(np.array([0, 1]), "monomial", 1)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1

    c = ut.QSVTCoefficients(np.array([0, 1]), "chebyshev", 1)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1

    # Scaled identity
    c = ut.QSVTCoefficients(np.array([0, 2]), "chebyshev", 1)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 2]))
    assert c.output_normalization == 2

    c = ut.QSVTCoefficients(np.array([0, 1]), "chebyshev", 2)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 2

    # Amplitude amplificiation
    c = ut.QSVTCoefficients(np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2]), "angles", 1)
    np.testing.assert_allclose(c.angles, np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    # Test that input normalization does nothing to the normalized polynomial or
    # the output normalization if angles are supplied
    c = ut.QSVTCoefficients(np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2]), "angles", 1.23)
    np.testing.assert_allclose(c.angles, np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    c = ut.QSVTCoefficients(np.array([0, 0, 0, 1]), "chebyshev", 1)
    b = ut.QSVTCoefficients(c.angles, "angles", 1)
    np.testing.assert_allclose(b.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)


def test_qsvt_grover():
    # Define v so it has some subnormalization
    v = (
        ut.Projection(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))
        @ ut.ConstantUnitary(2 ** (-1 / 2) * np.array([[1, 1], [1, -1]]))
        @ ut.Projection(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))
    )
    A = ut.QSVT(v, np.array(4 * [np.pi]), "angles")
    ut.verify(A)


def test_qsvt_with_ancillas():
    # Define v so it has some subnormalization
    A = ut.Identity(
        subspace=ut.Subspace([ut.ID, ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))])
    )
    assert A.subspace.total_qubits + A.subspace.clean_ancilla_count() > 2
    B = ut.QSVT(A, np.array(4 * [np.pi]), "angles")
    ut.verify(B)


def test_qsvt_with_polynomial():
    A = ut.Increment(bits=2)
    B = ut.QSVT(A, np.array([1, 0, 1]), "chebyshev")
    ut.verify(B)
