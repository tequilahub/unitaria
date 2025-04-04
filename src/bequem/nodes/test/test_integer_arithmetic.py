import pytest

from ..integer_arithmetic import Increment


@pytest.mark.parametrize("bits", [-2, 0])
def test_increment_constructor(bits):
    with pytest.raises(ValueError):
        Increment(bits)

@pytest.mark.parametrize("bits", [1, 4])
def test_increment(bits):
    A = Increment(bits)
    A.verify()

