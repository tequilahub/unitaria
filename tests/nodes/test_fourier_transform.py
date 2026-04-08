import pytest
import unitaria as ut


@pytest.mark.parametrize("bits", range(1, 5))
def test_fourier_transform(bits: int):
    A = ut.FourierTransform(bits)
    ut.verify(A)
