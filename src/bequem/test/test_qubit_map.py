import pytest

from ..qubit_map import QubitMap, IdBit, ZeroBit


def test_reduce():
    assert QubitMap([]).reduce() == QubitMap([])
    assert QubitMap([ZeroBit()]).reduce() == QubitMap([])
    assert QubitMap([IdBit()]).reduce() == QubitMap([IdBit()])
    assert QubitMap([ZeroBit(), IdBit(), ZeroBit(), ZeroBit(), IdBit(), ZeroBit()]).reduce() == QubitMap([IdBit(), IdBit()])
