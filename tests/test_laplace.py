from bequem.nodes import Increment, Identity, Add, Scale, Adjoint

N = 4


def test_add():
    C = ConstantIntegerAddition(N, 1)

    assert C.verify()

# A = Add(Scale(Identity(N), 2), Scale(Add(C, Adjoint(C)), -1))

# assert A.verify()
