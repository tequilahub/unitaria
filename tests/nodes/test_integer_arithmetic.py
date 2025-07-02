import pytest

from bequem.nodes.integer_arithmetic import Increment, IntegerAddition, ConstantIntegerAddition, ConstantIntegerMultiplication
from bequem.verifier import verify


@pytest.mark.parametrize("bits", [-2, 0])
def test_increment_constructor(bits):
    with pytest.raises(ValueError):
        Increment(bits)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_increment(bits):
    A = Increment(bits)
    verify(A)


@pytest.mark.parametrize(
    "source_bits, target_bits",
    [(2, 2), (2, 3), (3, 3), (3, 4)]
)
def test_integer_addition(source_bits, target_bits):
    A = IntegerAddition(source_bits, target_bits)
    verify(A)


@pytest.mark.parametrize(
    "bits, constant",
    [(1, 0), (1, 1), (2, 2), (4, 3)]
)
def test_constant_integer_addition(bits, constant):
    A = ConstantIntegerAddition(bits, constant)
    verify(A)

@pytest.mark.parametrize(
    "bits, constant",
    [(1, 1), (2, 1), (2, 3), (4, 5)]
)
def test_constant_integer_multiplication(bits, constant):
    A = ConstantIntegerMultiplication(bits, constant)
    verify(A)
