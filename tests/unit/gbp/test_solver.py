import numpy as np
import pytest

from src.gbp import (
    build_gbp_hamiltonian_qutip,
    compute_cut_value,
    exact_min_penalty,
    make_weight_matrix,
    solve_gbp_qutip,
)


def _build(edges, n_vertices):
    W = make_weight_matrix(edges, n_vertices)
    H_cost, H_penalty = build_gbp_hamiltonian_qutip(W, n_vertices)
    return W, H_cost, H_penalty


def _optimal_cuts(edges, n_vertices):
    W, H_cost, H_penalty = _build(edges, n_vertices)
    A = exact_min_penalty(H_cost, H_penalty)
    assignments = solve_gbp_qutip(H_cost, H_penalty, n_vertices, penalty=A + 1e-6)
    return [compute_cut_value(a, W) for a in assignments]


class TestSolveGbp:
    def test_only_returns_balanced_assignments(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        _, H_cost, H_penalty = _build(edges, n_vertices=4)
        A = exact_min_penalty(H_cost, H_penalty)
        assignments = solve_gbp_qutip(H_cost, H_penalty, n_vertices=4, penalty=A + 1e-6)
        assert all(sum(a) == 2 for a in assignments)

    def test_path_graph_min_cut_is_1(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        assert all(c == 1.0 for c in _optimal_cuts(edges, n_vertices=4))

    def test_complete_graph_min_cut(self):
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        # For a balanced bipartition of the complete graph, every partition has the same number of cut edges (2x2 = 4)
        assert all(c == 4.0 for c in _optimal_cuts(edges, n_vertices=4))

    def test_weighted_graph(self):
        edges = [(0, 1, 5.0), (1, 2, 1.0), (2, 3, 5.0), (0, 3, 1.0)]
        # The best partition keeps the weight-5.0 edge uncut: {0,1} / {2,3}
        assert min(_optimal_cuts(edges, n_vertices=4)) == pytest.approx(2.0)
