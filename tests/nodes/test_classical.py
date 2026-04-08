import pytest
import unitaria as ut


# TODO: We should check that, if the generic compute and compute_adjoint
#   implementations are overriden for performance, they match the generic ones


@pytest.mark.parametrize("bits", [-2, 0])
def test_increment_constructor(bits):
    with pytest.raises(ValueError):
        ut.Increment(bits=bits)
    with pytest.raises(TypeError):
        ut.Increment()


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_increment(bits):
    A = ut.Increment(bits=bits)
    ut.verify(A)


@pytest.mark.parametrize("source_bits, target_bits", [(2, 2), (2, 3), (3, 3), (3, 4)])
def test_integer_addition(source_bits, target_bits):
    A = ut.IntegerAddition(source_bits=source_bits, target_bits=target_bits)
    ut.verify(A)


@pytest.mark.parametrize("bits, constant", [(1, 0), (1, 1), (2, 2), (4, 3)])
def test_constant_integer_addition(bits, constant):
    A = ut.ConstantIntegerAddition(bits=bits, constant=constant)
    ut.verify(A)


@pytest.mark.parametrize("bits, constant", [(1, 1), (2, 1), (2, 3), (4, 5)])
def test_constant_integer_multiplication(bits, constant):
    A = ut.ConstantIntegerMultiplication(bits=bits, constant=constant)
    ut.verify(A)
