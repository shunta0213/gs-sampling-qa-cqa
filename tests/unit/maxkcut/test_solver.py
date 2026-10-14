import numpy as np
import pytest

from src.maxkcut import (
    build_maxkcut_hamiltonian_qutip,
    compute_cut_value,
    exact_min_penalty,
    make_weight_matrix,
    min_penalty,
    solve_maxkcut_qutip,
)


def _build(edges, n_vertices, k):
    W = make_weight_matrix(edges, n_vertices)
    H_cost, H_penalty = build_maxkcut_hamiltonian_qutip(W, n_vertices, k)
    return W, H_cost, H_penalty


def _optimal_cuts(edges, n_vertices, k):
    W, H_cost, H_penalty = _build(edges, n_vertices, k)
    A = exact_min_penalty(H_cost, H_penalty)
    assignments = solve_maxkcut_qutip(H_cost, H_penalty, n_vertices, k, penalty=A + 1e-6)
    return [compute_cut_value(a, W) for a in assignments]


# ---------------------------------------------------------------------------
# exact_min_penalty
# ---------------------------------------------------------------------------

class TestExactMinPenalty:
    def _ground_is_feasible(self, H_cost, H_penalty, A):
        """Check that the ground state (minimum energy) under penalty A is feasible."""
        diag_c = np.real(H_cost.diag())
        diag_p = np.real(H_penalty.diag())
        feasible = np.abs(diag_p) < 1e-10
        total = diag_c + A * diag_p
        return total[feasible].min() < total[~feasible].min()

    def test_above_threshold_ground_is_feasible(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        _, H_cost, H_penalty = _build(edges, n_vertices=3, k=3)
        A_star = exact_min_penalty(H_cost, H_penalty)
        assert self._ground_is_feasible(H_cost, H_penalty, A_star + 1e-6)

    def test_below_threshold_ground_is_infeasible(self):
        # A graph for which A* > 0 is guaranteed (for k=2 on a complete graph, infeasible states can have lower cost)
        edges = [(0, 1, 5.0), (1, 2, 5.0), (0, 2, 5.0)]
        _, H_cost, H_penalty = _build(edges, n_vertices=3, k=2)
        A_star = exact_min_penalty(H_cost, H_penalty)
        if A_star > 1e-8:
            assert not self._ground_is_feasible(H_cost, H_penalty, A_star - 1e-6)

    def test_tighter_than_heuristic(self):
        edges = [(0, 1), (1, 2)]
        W, H_cost, H_penalty = _build(edges, n_vertices=3, k=2)
        assert exact_min_penalty(H_cost, H_penalty) <= min_penalty(W)

    @pytest.mark.parametrize("k", [2, 3])
    def test_nonnegative(self, k):
        edges = [(0, 1), (1, 2)]
        _, H_cost, H_penalty = _build(edges, n_vertices=3, k=k)
        assert exact_min_penalty(H_cost, H_penalty) >= 0.0

    def test_weighted_graph(self):
        edges = [(0, 1, 3.0), (1, 2, 2.0), (0, 2, 1.0)]
        _, H_cost, H_penalty = _build(edges, n_vertices=3, k=2)
        A_star = exact_min_penalty(H_cost, H_penalty)
        assert self._ground_is_feasible(H_cost, H_penalty, A_star + 1e-6)


# ---------------------------------------------------------------------------
# solve_maxkcut_qutip
# ---------------------------------------------------------------------------

class TestSolveMaxkcut:
    def test_k3_triangle_all_edges_cut(self):
        assert all(c == 3.0 for c in _optimal_cuts([(0, 1), (1, 2), (0, 2)], 3, 3))

    def test_k3_triangle_k2_max_cut_is_2(self):
        assert all(c == 2.0 for c in _optimal_cuts([(0, 1), (1, 2), (0, 2)], 3, 2))

    def test_k4_complete_k3_max_cut_is_5(self):
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        assert all(c == 5.0 for c in _optimal_cuts(edges, 4, 3))

    def test_path_graph_k2_all_edges_cut(self):
        assert all(c == 3.0 for c in _optimal_cuts([(0, 1), (1, 2), (2, 3)], 4, 2))

    def test_single_edge_k2(self):
        assert all(c == 1.0 for c in _optimal_cuts([(0, 1)], 2, 2))

    def test_weighted_triangle_k2_max_cut_is_5(self):
        edges = [(0, 1, 3.0), (1, 2, 2.0), (0, 2, 1.0)]
        assert max(_optimal_cuts(edges, 3, 2)) == 5.0

    def test_returns_only_valid_assignments(self):
        edges = [(0, 1), (1, 2)]
        W, H_cost, H_penalty = _build(edges, n_vertices=3, k=2)
        A = exact_min_penalty(H_cost, H_penalty)
        assignments = solve_maxkcut_qutip(H_cost, H_penalty, n_vertices=3, k=2, penalty=A + 1e-6)
        assert all(-1 not in a for a in assignments)
