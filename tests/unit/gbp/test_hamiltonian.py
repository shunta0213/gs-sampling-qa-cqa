import numpy as np
import pytest
import qutip as qt

from src.gbp import build_gbp_hamiltonian_qutip, gbp_hamiltonian_scale, make_weight_matrix


def _basis(bits: list[int]) -> qt.Qobj:
    return qt.tensor([qt.basis(2, b) for b in bits])


def _walsh_max_abs_coefficient(H: qt.Qobj, n_qubits: int) -> float:
    """Compute exactly (via the Walsh-Hadamard transform of the diagonal) the maximum of
    |h_i| and |J_ij| for a diagonal Ising Hamiltonian H = c + Σ h_i s_i + Σ J_ij s_i s_j.

    If H is truly at most quadratic (linear + ZZ), each coefficient is exactly
        coeff = 2^-n * Σ_s H_diag(s) * (product of the corresponding s_i)
    (not an approximation). Used to verify the scale independently from the actually
    built operator, without relying on hand-derived formulas.
    """
    diag = np.real(H.diag())
    n_states = 2**n_qubits
    idx = np.arange(n_states)
    bits = (idx[:, None] >> (n_qubits - 1 - np.arange(n_qubits))) & 1
    spins = 1 - 2 * bits  # bit=0 -> s=+1, bit=1 -> s=-1 (same convention as build_*_hamiltonian_qutip)

    coeffs = []
    for i in range(n_qubits):
        coeffs.append(float(np.mean(diag * spins[:, i])))
        for j in range(i + 1, n_qubits):
            coeffs.append(float(np.mean(diag * spins[:, i] * spins[:, j])))
    return max((abs(c) for c in coeffs), default=0.0)


class TestReturnType:
    def test_returns_two_qobjs(self):
        W = make_weight_matrix([(0, 1)], n_vertices=2)
        result = build_gbp_hamiltonian_qutip(W, n_vertices=2)
        assert len(result) == 2
        assert all(isinstance(h, qt.Qobj) for h in result)

    def test_dimension(self):
        n = 4
        W = make_weight_matrix([(0, 1)], n_vertices=n)
        H_cost, H_penalty = build_gbp_hamiltonian_qutip(W, n_vertices=n)
        expected_dim = 2**n
        assert H_cost.shape == (expected_dim, expected_dim)
        assert H_penalty.shape == (expected_dim, expected_dim)

    def test_hermitian(self):
        W = make_weight_matrix([(0, 1), (1, 2)], n_vertices=4)
        H_cost, H_penalty = build_gbp_hamiltonian_qutip(W, n_vertices=4)
        for H in (H_cost, H_penalty):
            assert (H - H.dag()).norm() < 1e-10

    def test_rejects_odd_n_vertices(self):
        W = make_weight_matrix([], n_vertices=3)
        with pytest.raises(ValueError):
            build_gbp_hamiltonian_qutip(W, n_vertices=3)


class TestPenalty:
    def _expect(self, bits: list[int]) -> float:
        n = len(bits)
        W = make_weight_matrix([], n_vertices=n)
        _, H_penalty = build_gbp_hamiltonian_qutip(W, n_vertices=n)
        return float(np.real(qt.expect(H_penalty, _basis(bits))))

    def test_balanced_split_gives_zero_penalty(self):
        assert abs(self._expect([0, 0, 1, 1])) < 1e-10

    def test_unbalanced_split_gives_positive_penalty(self):
        assert self._expect([0, 0, 0, 1]) > 0

    def test_all_same_group_gives_max_penalty(self):
        assert self._expect([0, 0, 0, 0]) == pytest.approx(16.0)


class TestCostHamiltonian:
    def _expect(self, bits: list[int], edges, n_vertices: int) -> float:
        W = make_weight_matrix(edges, n_vertices=n_vertices)
        H_cost, _ = build_gbp_hamiltonian_qutip(W, n_vertices=n_vertices)
        return float(np.real(qt.expect(H_cost, _basis(bits))))

    def test_noncut_is_cheaper_than_cut(self):
        edges = [(0, 1)]
        cost_cut = self._expect([0, 1, 0, 0], edges, n_vertices=4)
        cost_noncut = self._expect([0, 0, 1, 1], edges, n_vertices=4)
        assert cost_noncut < cost_cut
        assert abs(cost_cut - cost_noncut - 1.0) < 1e-10

    def test_weighted_edge_cost_difference(self):
        edges = [(0, 1, 3.0)]
        cost_cut = self._expect([0, 1, 0, 0], edges, n_vertices=4)
        cost_noncut = self._expect([0, 0, 1, 1], edges, n_vertices=4)
        assert abs(cost_cut - cost_noncut - 3.0) < 1e-10


class TestHamiltonianScale:
    @pytest.mark.parametrize("n_vertices,edges,penalty", [
        (4, [(0, 1, 2.5), (1, 2, 1.0), (2, 3, 4.0)], 1.25),
        (2, [(0, 1, 4.5)], 0.83),
        (4, [(0, 1, 1.0), (0, 2, 3.0), (1, 3, 0.5)], 2.0),
        (4, [], 0.5),
    ])
    def test_matches_exact_coefficients(self, n_vertices, edges, penalty):
        """scale exactly matches the maximum absolute ZZ coefficient of the actually built
        H_problem = H_cost + penalty*H_penalty (coefficients are extracted independently
        from the diagonal via the Walsh-Hadamard transform and compared).

        Unlike Max-k-Cut, in GBP H_cost and H_penalty act on the ZZ term of the same vertex
        pair, so taking the max of each term independently misses cases where terms of
        opposite sign cancel (this test guards against that regression).
        """
        W = make_weight_matrix(edges, n_vertices=n_vertices)
        H_cost, H_penalty = build_gbp_hamiltonian_qutip(W, n_vertices=n_vertices)
        H_problem = H_cost + penalty * H_penalty

        expected = _walsh_max_abs_coefficient(H_problem, n_vertices)
        actual = gbp_hamiltonian_scale(W, penalty)
        assert actual == pytest.approx(max(expected, 1.0), abs=1e-8)

    def test_cancellation_makes_scale_tighter_than_naive_max(self):
        """When the H_cost and H_penalty coefficients cancel, scale can be smaller than
        max(|W|/2, |2*penalty|) (but still not below the lower bound 1.0).

        Every vertex pair is made an edge so that -w/2 + 2*penalty is close to exactly 0
        for all pairs (a pair without an edge would keep the coefficient 2*penalty, which
        could coincidentally equal naive_upper_bound and blur the intent of the test).
        """
        n_vertices = 4
        w = 2.72
        edges = [(i, j, w) for i in range(n_vertices) for j in range(i + 1, n_vertices)]
        W = make_weight_matrix(edges, n_vertices=n_vertices)
        penalty = w / 4.0  # Chosen so that -w/2 + 2*penalty = 0

        naive_upper_bound = max(float(np.max(np.abs(W))) / 2.0, abs(2.0 * penalty), 1.0)
        actual = gbp_hamiltonian_scale(W, penalty)
        assert actual < naive_upper_bound
        assert actual == pytest.approx(1.0)  # All vertex pairs cancel, so it hits the lower bound 1.0

    def test_never_below_one(self):
        W = make_weight_matrix([(0, 1, 0.01)], n_vertices=2)
        assert gbp_hamiltonian_scale(W, penalty=0.0) == pytest.approx(1.0)
