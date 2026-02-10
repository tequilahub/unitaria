from unitaria.nodes.classical.increment import Increment
from unitaria.nodes.qsvt.qsvt import QSVTCoefficients, QSVT
from unitaria.nodes.constants.constant_unitary import ConstantUnitary
from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.basic.projection import Projection
from unitaria.subspace import Subspace, ID, ControlledSubspace
from unitaria.verifier import verify

import numpy as np


def test_qsvt_coefficients():
    # Identity
    c = QSVTCoefficients(np.array([0, 0]), "angles", 1)
    np.testing.assert_allclose(c.angles, np.array([0, 0]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1

    c = QSVTCoefficients(np.array([0, 1]), "monomial", 1)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1

    c = QSVTCoefficients(np.array([0, 1]), "chebyshev", 1)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1

    # Scaled identity
    c = QSVTCoefficients(np.array([0, 2]), "chebyshev", 1)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 2

    c = QSVTCoefficients(np.array([0, 1]), "chebyshev", 2)
    np.testing.assert_allclose(c.angles[0], -c.angles[1], atol=1e-5)
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 1]))
    assert c.output_normalization == 1 / 2

    # Amplitude amplificiation
    c = QSVTCoefficients(np.array(4 * [np.pi]), "angles", 1)
    np.testing.assert_allclose(c.angles, np.array(4 * [np.pi]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    c = QSVTCoefficients(np.array(4 * [np.pi]), "angles", 1.23)
    np.testing.assert_allclose(c.angles, np.array(4 * [np.pi]))
    np.testing.assert_allclose(c.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)
    assert c.output_normalization == 1

    c = QSVTCoefficients(np.array([0, 0, 0, 1]), "chebyshev", 1)
    b = QSVTCoefficients(c.angles, "angles", 1)
    np.testing.assert_allclose(b.polynomial.coef, np.array([0, 0, 0, 1]), atol=1e-5)


def test_qsvt_grover():
    # Define v so it has some subnormalization
    v = (
        Projection(1, Subspace(0, 1))
        @ ConstantUnitary(2 ** (-1 / 2) * np.array([[1, 1], [1, -1]]))
        @ Projection(Subspace(0, 1), Subspace(1))
    )
    A = QSVT(v, np.array(4 * [np.pi]), "angles")
    verify(A)


def test_qsvt_with_ancillas():
    # Define v so it has some subnormalization
    A = Identity(Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))]))
    assert A.subspace.total_qubits + A.subspace.clean_ancilla_count() > 2
    B = QSVT(A, np.array(4 * [np.pi]), "angles")
    verify(B)


def test_qsvt_with_polynomial():
    A = Increment(2)
    B = QSVT(A, np.array([1, 0, 1]), "chebyshev")
    verify(B)
