from typing import Sequence

import numpy as np
import tequila as tq
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

from unitaria.nodes import Node
from unitaria.subspace import Subspace
from unitaria.circuit import Circuit


def poly_conj(p: np.polynomial.Chebyshev) -> np.polynomial.Chebyshev:
    return np.polynomial.Chebyshev(np.conj(p.coef))


class QSVTCoefficients:
    def __init__(self, data: np.ndarray, format: str, input_normalization: float):
        self.data = data
        self.format = format
        self.input_normalization = input_normalization

        if format == "angles":
            self.angles = self.data
            X = np.polynomial.Chebyshev(np.array([0, 1], dtype=np.complex128))
            state = [
                np.polynomial.Chebyshev(np.array([1], dtype=np.complex128)),
                np.polynomial.Chebyshev(np.array([0], dtype=np.complex128)),
            ]
            for angle in self.angles[:-1]:
                state[0] *= np.exp(angle * 1j)
                state[1] *= np.exp(angle * 1j)
                state = [
                    X * state[0] + (X**2 - 1) * poly_conj(state[1]),
                    X * state[1] + poly_conj(state[0]),
                ]
            state[0] *= np.exp(self.angles[-1] * 1j)
            state[1] *= np.exp(self.angles[-1] * 1j)
            self.polynomial = np.polynomial.Chebyshev(state[0].coef.real)

            self.output_normalization = 1
        else:
            if format == "monomial":
                self.polynomial = np.polynomial.Polynomial(self.data).convert(kind=np.polynomial.Chebyshev)
            elif format == "chebyshev":
                self.polynomial = np.polynomial.Chebyshev(self.data)
            else:
                raise ValueError(
                    f"format may only be one of 'angles', 'monomial', or 'chebyshev' (passed format={format})"
                )

            self.polynomial = self.polynomial(np.polynomial.Chebyshev([0, 1 / input_normalization]))
            maxima = self.polynomial.deriv().roots()
            maxima = maxima[np.abs(np.imag(maxima)) < 1e-6]
            maxima = maxima[np.abs(maxima) <= 1]
            self.output_normalization = np.max(np.abs(self.polynomial(np.concatenate((maxima, [-1, 1])))))
            self.polynomial /= self.output_normalization

            # TODO: Reimplement, because this generates a lot of print output
            self.angles, _, _ = QuantumSignalProcessingPhases(self.polynomial, method="sym_qsp", chebyshev_basis=True)
            # "sym_qsp" method approximates polynomial in imaginary part.
            # By this extra rotation, we turn it into the real part.
            self.angles[0] -= np.pi / 2

    def degree(self) -> int:
        return self.polynomial.degree()

    def angles_to_r_convention(self) -> np.ndarray:
        result = self.angles[:-1] - np.pi / 2
        result[0] += self.angles[-1] + self.degree() * np.pi / 2
        print(result)
        return result


class QSVT(Node):
    def __init__(self, A: Node, coefficients: np.ndarray, format: str = "monomial", normalization: float = 1):
        self.coefficients = QSVTCoefficients(coefficients, format, A.normalization)
        if self.coefficients.degree() % 2 == 0:
            super().__init__(A.dimension_in, A.dimension_in)
        else:
            super().__init__(A.dimension_in, A.dimension_out)
        self.A = A
        self.normalization = normalization

    def children(self) -> list[Node]:
        return [self.A]

    def parameters(self) -> dict:
        return {
            "coefficients": self.coefficients.data,
            "format": self.coefficients.format,
            "normalization": self.normalization,
        }

    def _normalization(self) -> float:
        return self.normalization * self.coefficients.output_normalization

    def _subspace_in(self) -> Subspace:
        return Subspace(self.A.subspace_in.registers, 1)

    def _subspace_out(self) -> Subspace:
        if self.coefficients.degree() % 2 == 0:
            return Subspace(self.A.subspace_in.registers, 1)
        else:
            return Subspace(self.A.subspace_out.registers, 1)

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

        for i, angle in enumerate(reversed(self.coefficients.angles_to_r_convention())):
            if i % 2 == 0:
                circuit += node_circuit
                circuit += subspace_out_circuit
            else:
                circuit += node_circuit.adjoint()
                circuit += subspace_in_circuit

            # TODO: Do not use projection circuits for last rotation
            circuit += tq.gates.Rz(-2 * np.real(angle), rotation_bit)

            if i % 2 == 0:
                circuit += subspace_out_circuit
            else:
                circuit += subspace_in_circuit

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
