import numpy as np
import pytest

from src.maxkcut import make_random_edges, make_weight_matrix, min_penalty


class TestMakeRandomEdges:
    def test_returns_list(self):
        edges = make_random_edges(n_vertices=5, seed=0)
        assert isinstance(edges, list)

    def test_no_self_loops(self):
        edges = make_random_edges(n_vertices=10, edge_prob=1.0, seed=0)
        assert all(e[0] != e[1] for e in edges)

    def test_no_duplicate_edges(self):
        edges = make_random_edges(n_vertices=10, edge_prob=1.0, seed=0)
        pairs = [(e[0], e[1]) for e in edges]
        assert len(pairs) == len(set(pairs))

    def test_complete_graph_when_prob_1(self):
        n = 5
        edges = make_random_edges(n_vertices=n, edge_prob=1.0, seed=0)
        assert len(edges) == n * (n - 1) // 2

    def test_empty_graph_when_prob_0(self):
        edges = make_random_edges(n_vertices=5, edge_prob=0.0, seed=0)
        assert len(edges) == 0

    def test_unweighted_has_no_weight(self):
        edges = make_random_edges(n_vertices=5, edge_prob=1.0, seed=0)
        assert all(len(e) == 2 for e in edges)

    def test_weighted_has_weight(self):
        edges = make_random_edges(n_vertices=5, edge_prob=1.0, weight_range=(1.0, 5.0), seed=0)
        assert all(len(e) == 3 for e in edges)

    def test_weight_within_range(self):
        lo, hi = 2.0, 8.0
        edges = make_random_edges(n_vertices=10, edge_prob=1.0, weight_range=(lo, hi), seed=0)
        assert all(lo <= e[2] <= hi for e in edges)

    def test_reproducible_with_seed(self):
        e1 = make_random_edges(n_vertices=8, edge_prob=0.5, seed=42)
        e2 = make_random_edges(n_vertices=8, edge_prob=0.5, seed=42)
        assert e1 == e2

    def test_different_seeds_differ(self):
        e1 = make_random_edges(n_vertices=8, edge_prob=0.5, seed=0)
        e2 = make_random_edges(n_vertices=8, edge_prob=0.5, seed=1)
        assert e1 != e2


class TestMakeWeightMatrix:
    def test_unweighted(self):
        W = make_weight_matrix([(0, 1), (1, 2)], n_vertices=3)
        assert W[0, 1] == 1.0
        assert W[1, 0] == 1.0
        assert W[1, 2] == 1.0
        assert W[0, 2] == 0.0

    def test_weighted(self):
        W = make_weight_matrix([(0, 1, 3.5)], n_vertices=2)
        assert W[0, 1] == 3.5
        assert W[1, 0] == 3.5

    def test_symmetric(self):
        edges = [(0, 1, 2.0), (1, 2, 5.0), (0, 2, 1.0)]
        W = make_weight_matrix(edges, n_vertices=3)
        assert np.allclose(W, W.T)

    def test_shape(self):
        W = make_weight_matrix([], n_vertices=4)
        assert W.shape == (4, 4)


class TestMinPenalty:
    def test_exceeds_max_weight(self):
        W = make_weight_matrix([(0, 1, 3.0), (1, 2, 5.0)], n_vertices=3)
        assert min_penalty(W) > W.max()

    def test_unit_weight(self):
        W = make_weight_matrix([(0, 1)], n_vertices=2)
        assert min_penalty(W) > 1.0
