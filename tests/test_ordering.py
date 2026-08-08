import numpy as np
import pytest

from scaled_vecchia.ordering import find_ordered_nn, maximin_order


def test_maximin_order_is_a_permutation():
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    order = maximin_order(X)
    assert sorted(order.tolist()) == list(range(len(X)))


def test_maximin_order_first_point_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    X = rng.random((50, 2))
    o1 = maximin_order(X)
    o2 = maximin_order(X)
    np.testing.assert_array_equal(o1, o2)


def test_find_ordered_nn_matches_brute_force():
    rng = np.random.default_rng(2)
    n, d, m = 120, 3, 8
    X = rng.random((n, d))
    order = maximin_order(X)
    Xo = X[order]
    nn = find_ordered_nn(Xo, m)

    for i in range(1, n):
        k = min(i, m)
        dist = np.linalg.norm(Xo[:i] - Xo[i], axis=1)
        truth = set(np.argsort(dist)[:k].tolist())
        found = set(nn[i, 1:k + 1].tolist())
        assert truth == found, f"mismatch at row {i}"


def test_find_ordered_nn_self_column_and_padding():
    rng = np.random.default_rng(3)
    X = rng.random((30, 2))
    m = 5
    nn = find_ordered_nn(X, m)
    assert np.array_equal(nn[:, 0], np.arange(30))
    # row 0 has no earlier neighbours: everything after the self column is -1
    assert np.all(nn[0, 1:] == -1)
    # row 2 (0-indexed) can only have up to 2 earlier neighbours
    assert np.sum(nn[2, 1:] != -1) == 2


def test_find_ordered_nn_start_param_skips_rows():
    rng = np.random.default_rng(4)
    X = rng.random((60, 2))
    m = 6
    nn_full = find_ordered_nn(X, m, start=0)
    nn_partial = find_ordered_nn(X, m, start=40)
    # Rows >= start should agree regardless of `start`, up to the order
    # within the conditioning set: find_ordered_nn only guarantees the *set*
    # of the m nearest neighbours for i > m, not a stable order among them
    # (see its docstring). `start` changes how candidates are batched
    # against the KD-tree, so ties/near-ties can legitimately land in a
    # different order via np.argpartition without changing which neighbours
    # are selected.
    assert np.array_equal(nn_full[40:, 0], nn_partial[40:, 0])
    for i in range(40, 60):
        assert set(nn_full[i, 1:].tolist()) == set(nn_partial[i, 1:].tolist()), \
            f"neighbour set mismatch at row {i}"
