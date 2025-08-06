import pytest

from unitaria import verify
from unitaria.nodes.fourier_transform.fourier_transform import FourierTransform


@pytest.mark.parametrize("bits", range(1, 5))
def test_fourier_transform(bits: int):
    A = FourierTransform(bits)
    verify(A)
