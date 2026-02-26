from typing import Sequence

import numpy as np
import scipy as sp
import tequila as tq

from unitaria.nodes import Node
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit
from unitaria.util import poly_sup_norm


def interpolation_points(reduced_degree):
    return np.cos(np.pi * np.arange(reduced_degree) / (2 * reduced_degree))


def DF(reduced_phases, parity, xs):
    state = np.zeros((3, len(xs)))
    xs_sqrt = np.sqrt(1 - xs**2)
    xs_main_diagonal = 2 * xs**2 - 1
    xs_off_diagonal = 2 * xs * xs_sqrt
    xs_operator = np.array([[xs_main_diagonal, -xs_off_diagonal], [xs_off_diagonal, xs_main_diagonal]])
    xs_operator = np.moveaxis(xs_operator, -1, 0)
    if parity == 0:
        state[0] = 1
    else:
        state[0] = xs
        state[2] = xs_sqrt
    cos_phase = np.cos(2 * reduced_phases[0])
    sin_phase = np.sin(2 * reduced_phases[0])
    op = np.array([[cos_phase, -sin_phase], [sin_phase, cos_phase]])
    state[[0, 1]] = op @ state[[0, 1]]
    for i in range(1, len(reduced_phases)):
        cos_phase = np.cos(2 * reduced_phases[i])
        sin_phase = np.sin(2 * reduced_phases[i])
        op = np.array([[cos_phase, -sin_phase], [sin_phase, cos_phase]])
        state[[0, 2]] = np.matvec(xs_operator, state[[0, 2]].T).T
        state[[0, 1]] = op @ state[[0, 1]]

    value = state[1].copy()

    dual_state = np.zeros((3, len(xs)))
    dual_state[1] = 2

    derivative = np.zeros((len(xs), len(reduced_phases)))

    xs_operator = np.swapaxes(xs_operator, -1, -2)
    for i in range(len(reduced_phases) - 1, 0, -1):
        derivative[:, i] = dual_state[1] * state[0] - dual_state[0] * state[1]
        phase = reduced_phases[i]
        cos_phase = np.cos(2 * phase)
        sin_phase = np.sin(2 * phase)
        op = np.array([[cos_phase, sin_phase], [-sin_phase, cos_phase]])
        state[[0, 1]] = op @ state[[0, 1]]
        dual_state[[0, 1]] = op @ dual_state[[0, 1]]
        state[[0, 2]] = np.matvec(xs_operator, state[[0, 2]].T).T
        dual_state[[0, 2]] = np.matvec(xs_operator, dual_state[[0, 2]].T).T

    derivative[:, 0] = dual_state[1] * state[0] - dual_state[0] * state[1]

    return value, derivative


def compute_angles_internal(poly):
    degree = poly.degree()
    reduced_degree = degree // 2 + 1
    parity = degree % 2

    xs = interpolation_points(reduced_degree)
    samples = poly(xs)
    reduced_phases = poly.coef[parity::2] / 2
    assert len(reduced_phases) == reduced_degree

    def fun(x):
        value, derivative = DF(x, parity, xs)
        return value - samples, derivative

    res = sp.optimize.root(fun, reduced_phases, jac=True)
    if not res.success:
        return None
    reduced_phases = res.x

    # Turn angles to encode polynomial in real part
    reduced_phases[-1] -= np.pi / 4
    full_phases = np.zeros(poly.degree() + 1)
    full_phases[reduced_degree - 1 :: -1] = reduced_phases
    full_phases[-reduced_degree::] += reduced_phases

    return full_phases


def compute_angles(poly):
    """
    Computes angles for given polynomial

    This takes into account the parity and normalization of the polynomial.
    It returns a list of one (if the polynomial has definite partiy) or two
    tuples. The first element of the tuples is an angle sequence for QSVT
    and the second element is the corresponding weight, such that the sum of
    QSVT polynomials equals the input to this function.
    """
    poly = poly.convert(kind=np.polynomial.Chebyshev)

    parity = poly.degree() % 2
    is_parity = np.allclose(poly.coef[(1 - parity) :: 2], 0, atol=1e-8)

    coefs_parity = poly.coef.copy()
    coefs_parity[1 - parity :: 2] = 0
    poly_parity = np.polynomial.Chebyshev(coefs_parity)
    polys = [poly_parity]
    if not is_parity:
        coefs_non_parity = poly.coef[:-1].copy()
        coefs_non_parity[parity::2] = 0
        poly_non_parity = np.polynomial.Chebyshev(coefs_non_parity)
        polys.append(poly_non_parity)

    result = []
    for subpoly in polys:
        normalization = poly_sup_norm(subpoly)
        angles = None
        max_tries = 100
        for i in range(max_tries):
            angles = compute_angles_internal(subpoly / normalization)
            if angles is not None:
                break
            normalization += 1e-5 * normalization
        if angles is None:
            raise RuntimeError(f"Could not compute angles for {subpoly}")

        # Convert angles to Rx convention
        angles[:] -= np.pi / 2
        angles[0] += len(angles) * np.pi / 4
        angles[-1] += len(angles) * np.pi / 4

        result.append((subpoly / normalization, angles, normalization))

    return result


def poly_conj(p: np.polynomial.Chebyshev) -> np.polynomial.Chebyshev:
    return np.polynomial.Chebyshev(np.conj(p.coef))


class QSVTCoefficients:
    """
    Class for computing between different formats of polynomials and QSVT angles

    :param data:
        Coefficients of a polynomial in a basis determined by ``format`` or angles
        in R convention
    :param format:
        One of ``"angles"``, ``"chebyshev``, or ``"monomial"``
    :param input_normalization":
        See below

    :ivar polynomial:
        The normalized polynomial
    :ivar input_normalization:
        Factor by which the x-axis of the polynomial is normalized, i.e. the
        normalization of the block encoding that is transformed
    :ivar output_normalization:
        Factor by which the y-axis of the polynomial is normalized, i.e. the
        normalization of the resulting block encoding
    :ivar angles:
        The angles in R-convention. Will be symmetric if ``data`` are polynomial
        coefficients
    """

    polynomial: np.polynomial.Chebyshev
    input_normalization: float
    output_normalization: float
    angles: np.ndarray

    def __init__(self, data: np.ndarray, format: str, input_normalization: float):
        self.data = data.astype(np.float64)
        self.format = format
        self.input_normalization = input_normalization

        if format == "angles":
            # TODO: Symmetricise angles
            self.angles = self.data

            angles_Wx = self.angles + np.pi / 2
            angles_Wx[0] -= len(self.angles) * np.pi / 4
            angles_Wx[-1] -= len(self.angles) * np.pi / 4

            parity = 1 - len(angles_Wx) % 2
            reduced_phases = angles_Wx[(-len(angles_Wx) + 1) // 2 :].copy()
            if parity == 0:
                reduced_phases[0] /= 2

            reduced_phases[-1] += np.pi / 4

            xs = np.cos(np.pi * (np.arange(len(angles_Wx)) + 0.5) / len(angles_Wx))
            value, _derivative = DF(reduced_phases, parity, xs)

            self.polynomial = np.polynomial.Chebyshev.fit(xs, value, len(angles_Wx) - 1, domain=[-1, 1])

            self.output_normalization = 1
        else:
            if format == "monomial":
                self.polynomial = np.polynomial.Polynomial(self.data).convert(kind=np.polynomial.Chebyshev)
            elif format == "chebyshev":
                self.polynomial = np.polynomial.Chebyshev(self.data)
            else:
                raise ValueError(
                    f'format may only be one of "angles", "monomial", or "chebyshev" (passed format={format})'
                )

            self.polynomial = self.polynomial(np.polynomial.Chebyshev([0, 1 / input_normalization]))

            parity = self.polynomial.degree() % 2
            np.testing.assert_allclose(
                self.polynomial.coef[1 - parity :: 2], 0, atol=1e-8, err_msg="Polynomial does not have definite parity"
            )
            self.polynomial.coef[1 - parity :: 2] = 0.0

            subpolys = compute_angles(self.polynomial)
            assert len(subpolys) == 1, "Polynomial does not have definite parity"

            self.polynomial, self.angles, self.output_normalization = subpolys[0]

    def degree(self) -> int:
        return self.polynomial.degree()


class QSVT(Node):
    def __init__(self, A: Node, coefficients: np.ndarray, format: str = "monomial"):
        self.coefficients = QSVTCoefficients(coefficients, format, A.normalization)
        if self.coefficients.degree() % 2 == 0:
            super().__init__(A.dimension_in, A.dimension_in)
        else:
            super().__init__(A.dimension_in, A.dimension_out)
        self.A = A

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {
            "coefficients": self.coefficients.data,
            "format": self.coefficients.format,
            "normalization": self.normalization,
        }

    def _normalization(self) -> float:
        return self.coefficients.output_normalization

    def _subspace_in(self) -> Subspace:
        return Subspace(registers=self.A.subspace_in.registers, zero_qubits=1)

    def _subspace_out(self) -> Subspace:
        if self.coefficients.degree() % 2 == 0:
            return Subspace(registers=self.A.subspace_in.registers, zero_qubits=1)
        else:
            return Subspace(registers=self.A.subspace_out.registers, zero_qubits=1)

    def _compute_internal(self, input: np.ndarray, compute, compute_adjoint) -> np.ndarray:
        # TODO: For now, the polynomial should either be odd or even
        # Probably this should be tested somewhere else
        Tnp = input
        if self.coefficients.degree() % 2 == 0:
            output = Tnp * self.coefficients.polynomial.coef[0]
        else:
            assert np.isclose(self.coefficients.polynomial.coef[0], 0)
            output = np.zeros_like(compute(input))
        if self.coefficients.degree() >= 0:
            Tn = compute(Tnp) / self.A.normalization
        for n in range(1, self.coefficients.degree() + 1):
            if (self.coefficients.degree() + n) % 2 == 0:
                output += Tn * self.coefficients.polynomial.coef[n]
            else:
                assert np.isclose(self.coefficients.polynomial.coef[n], 0)
            if n < self.coefficients.degree():
                if n % 2 == 0:
                    Tnp, Tn = Tn, 2 * (compute(Tn) / self.A.normalization) - Tnp
                else:
                    Tnp, Tn = Tn, 2 * (compute_adjoint(Tn) / self.A.normalization) - Tnp

        output *= self.coefficients.output_normalization
        return output

    def compute(self, input: np.ndarray) -> np.ndarray:
        return self._compute_internal(input, self.A.compute, self.A.compute_adjoint)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return self._compute_internal(input, self.A.compute_adjoint, self.A.compute)

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        circuit = Circuit()
        rotation_bit = target[-1]
        circuit += tq.gates.H(rotation_bit)

        node_circuit = self.A.circuit(target[:-1], clean_ancillae, borrowed_ancillae)
        subspace_in_circuit = self.A.subspace_in.circuit(target, flag=target[-1], ancillae=clean_ancillae)
        subspace_out_circuit = self.A.subspace_out.circuit(target, flag=target[-1], ancillae=clean_ancillae)

        for i, angle in enumerate(self.coefficients.angles[1:]):
            if i % 2 == 0:
                circuit += node_circuit
                circuit += subspace_out_circuit
            else:
                circuit += node_circuit.adjoint()
                circuit += subspace_in_circuit

            # TODO: Do not use projection circuits for last rotation
            # Combine last and first angle into one rotation
            if i == len(self.coefficients.angles) - 2:
                circuit += tq.gates.Rz(2 * np.real(angle + self.coefficients.angles[0]), rotation_bit)
            else:
                circuit += tq.gates.Rz(2 * np.real(angle), rotation_bit)

            if i % 2 == 0:
                circuit += subspace_out_circuit.adjoint()
            else:
                circuit += subspace_in_circuit.adjoint()

        circuit += tq.gates.H(rotation_bit)

        return circuit

    def clean_ancilla_count(self) -> int:
        return max(
            self.A.subspace_in.clean_ancilla_count(),
            self.A.subspace_out.clean_ancilla_count(),
            self.A.clean_ancilla_count(),
        )

    def borrowed_ancilla_count(self) -> int:
        return 0
