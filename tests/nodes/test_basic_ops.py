from bequem.nodes.basic_ops import Tensor
from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic import Increment
from bequem.qubit_map import QubitMap


def test_tensor():
    A = Increment(1)
    B = Identity(QubitMap(1, 1))

    # Tensor(A, B).verify()
    # Tensor(B, A).verify()
    # Tensor(B, Tensor(A, B)).verify()
    Tensor(Tensor(B, A), B).verify()
