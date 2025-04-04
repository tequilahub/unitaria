from bequem.nodes import Increment, Identity, Add, Scale, Adjoint

N = 4


def test_increment():
    C = Increment(N)
    C.verify()

# A = Add(Scale(Identity(N), 2), Scale(Add(C, Adjoint(C)), -1))

# assert A.verify()
