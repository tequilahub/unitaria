import numpy as np

from unitaria.nodes import AbstractNode


def test_abstract_node():
    np.testing.assert_allclose(
        AbstractNode(2, 2, lambda x: x[::-1], lambda x: x[::-1]).toarray(), np.array([[0, 1], [1, 0]])
    )
