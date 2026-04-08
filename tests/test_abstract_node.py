import numpy as np
import unitaria as ut


def test_abstract_node():
    np.testing.assert_allclose(
        ut.AbstractNode(2, 2, lambda x: x[::-1], lambda x: x[::-1]).toarray(), np.array([[0, 1], [1, 0]])
    )
