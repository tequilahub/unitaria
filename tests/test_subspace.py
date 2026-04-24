import numpy as np
import pytest
import unitaria as ut


def test_eq():
    assert ut.Subspace() == ut.Subspace()
    assert ut.Subspace("#") == ut.Subspace("#")
    assert ut.Subspace("0") == ut.Subspace("0")
    assert ut.Subspace("#") != ut.Subspace("#0")
    assert ut.Subspace("#") & ut.Subspace("0") == ut.Subspace("#0")
    assert ut.Subspace("#") == ut.Subspace([ut.ControlledSubspace(ut.Subspace(), ut.Subspace())])
    c = ut.Subspace() | ut.Subspace()
    assert ut.Subspace("#") == c
    c = ut.Subspace("#") | ut.Subspace("0")
    assert c == c
    assert c != ut.Subspace("#")


def test_from_dim():
    rng = np.random.default_rng(0)
    for n in range(1, 5):
        dim = rng.integers(0, 2**n)
        np.testing.assert_equal(ut.Subspace.from_dim(dim, bits=n).enumerate_basis(), np.arange(dim))


def test_basis():
    assert ut.Subspace("0").test_basis(0)
    assert not ut.Subspace("0").test_basis(1)

    np.testing.assert_allclose(ut.Subspace().enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(ut.Subspace("0").enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(ut.Subspace("#").enumerate_basis(), np.array([0, 1]))
    np.testing.assert_allclose(
        ut.Subspace([ut.ControlledSubspace(ut.Subspace("#"), ut.Subspace("0"))]).enumerate_basis(),
        np.array([0, 1, 2]),
    )
    # TODO
    # circuit = Circuit()
    np.testing.assert_allclose(
        ut.Subspace("0#").enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_qubits():
    assert ut.Subspace().total_qubits == 0
    assert ut.Subspace("0").total_qubits == 1
    assert ut.Subspace("#").total_qubits == 1
    assert (ut.Subspace("#") | ut.Subspace("0")).total_qubits == 2
    assert (ut.Subspace("#") & (ut.Subspace("0") | ut.Subspace("0")) & ut.Subspace("00")).total_qubits == 5


examples = [
    ut.Subspace(),
    ut.Subspace("0#"),
    (ut.Subspace("#") | ut.Subspace("0")) & ut.Subspace("#"),
    ((ut.Subspace("#") & (ut.Subspace("0") | ut.Subspace("#"))) | ut.Subspace("00#")) & ut.Subspace("#"),
]


@pytest.mark.parametrize(
    "subspace",
    examples,
)
def test_circuit(subspace: ut.Subspace):
    subspace.verify_circuit()


@pytest.mark.parametrize("subspace", examples)
def test_initial_zeros(subspace):
    initial_zeros = subspace.initial_zeros()
    assert len(subspace.tensor_factors) >= initial_zeros
    assert initial_zeros == len(subspace.tensor_factors) or isinstance(
        subspace.tensor_factors[-(initial_zeros + 1)], ut.ControlledSubspace
    )


def test_case_zero():
    assert ut.Subspace().case_zero() is None
    assert ut.Subspace("0#").case_zero() == ut.Subspace()
    assert ((ut.Subspace("#") | ut.Subspace("0")) & ut.Subspace("#")).case_zero() == ut.Subspace("##")

    subspace = (ut.Subspace("00") | (ut.Subspace("0") | ut.Subspace("#"))) & ut.Subspace("#")
    case_zero = ut.Subspace("00#")
    assert subspace.case_zero() == case_zero


@pytest.mark.parametrize(
    ("subspace", "expected"),
    [
        (ut.Subspace(), "<zero qubit subspace>"),
        (ut.Subspace("0#"),
            "  │\n"
            "1 0\n"
            "  │\n"
            "0 #"
        ),
        ((ut.Subspace("#") | ut.Subspace("0")) & ut.Subspace("#"),
            "  │\n"
            "2 ?─┬─┐\n"
            "    │ │\n"
            "1   # 0\n"
            "  ╔═╩═╛\n"
            "0 #"),
        (((ut.Subspace("#") & (ut.Subspace("0") | ut.Subspace("#"))) | ut.Subspace("00#")) & ut.Subspace("#"),
            "  │\n"
            "4 ?─┬─────┐\n"
            "    │     │\n"
            "3   #     0\n"
            "    ║     │\n"
            "2   ?─┬─┐ 0\n"
            "      │ │ │\n"
            "1     0 # #\n"
            "  ╔═══╧═╩═╝\n"
            "0 #"),
    ],
)  # fmt: skip
def test_str(subspace: ut.Subspace, expected: str):
    assert str(subspace) == expected
