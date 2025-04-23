import pytest

import numpy as np
from bequem.nodes.constant import ConstantVector


def test_constant_matrix():
    A = ConstantVector(np.array([1, 2j, 1 / 3, -1j / 4]))
    A.verify()

@pytest.mark.xfail
def test_global_phase():
    A = ConstantVector(np.array([1, -1]))
    A.verify()
    A = ConstantVector(np.array([-1, 1]))
    A.verify()
