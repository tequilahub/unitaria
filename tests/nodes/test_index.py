import numpy as np
import unitaria as ut


def random_index(dim, rng):
    start = rng.integers(0, dim)
    stop = rng.integers(start + 1, dim + 1)
    step = rng.integers(1, max(1, stop - start) + 1)

    return start, stop, step


def test_index():
    rng = np.random.default_rng(0)
    for i in range(1, 5):
        n, m = rng.integers(1, 2**i, size=2)
        mat = rng.standard_normal((n, m))

        start1, stop1, step1 = random_index(n, rng)
        start2, stop2, step2 = random_index(m, rng)

        ut.verify(ut.Index(n, slice(start1, stop1, step1)))
        ut.verify(ut.Index(m, slice(start2, stop2, step2)))

        ut.verify(ut.ConstantMatrix(mat)[start1], mat[start1 : start1 + 1, :])
        ut.verify(ut.ConstantMatrix(mat)[start1, start2], mat[start1, start2])
        ut.verify(ut.ConstantMatrix(mat)[start1:stop1:step1], mat[start1:stop1:step1])
        ut.verify(
            ut.ConstantMatrix(mat)[start1:stop1:step1, start2:stop2:step2], mat[start1:stop1:step1, start2:stop2:step2]
        )
