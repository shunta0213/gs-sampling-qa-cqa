import numpy as np
import pytest
import qutip as qt

from src.maxkcut import build_maxkcut_hamiltonian_qutip, make_weight_matrix, maxkcut_hamiltonian_scale


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
        result = build_maxkcut_hamiltonian_qutip(W, n_vertices=2, k=2)
        assert len(result) == 2
        assert all(isinstance(h, qt.Qobj) for h in result)

    def test_dimension(self):
        n, k = 3, 3
        W = make_weight_matrix([(0, 1)], n_vertices=n)
        H_cost, H_penalty = build_maxkcut_hamiltonian_qutip(W, n_vertices=n, k=k)
        expected_dim = 2 ** (n * k)
        assert H_cost.shape == (expected_dim, expected_dim)
        assert H_penalty.shape == (expected_dim, expected_dim)

    def test_hermitian(self):
        W = make_weight_matrix([(0, 1), (1, 2)], n_vertices=3)
        H_cost, H_penalty = build_maxkcut_hamiltonian_qutip(W, n_vertices=3, k=2)
        for H in (H_cost, H_penalty):
            assert (H - H.dag()).norm() < 1e-10


class TestPenalty:
    def _expect(self, bits: list[int], n_vertices: int, k: int) -> float:
        W = make_weight_matrix([], n_vertices=n_vertices)
        _, H_penalty = build_maxkcut_hamiltonian_qutip(W, n_vertices=n_vertices, k=k)
        return float(np.real(qt.expect(H_penalty, _basis(bits))))

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_valid_one_hot_gives_zero_penalty(self, k):
        bits = [0] * k
        bits[0] = 1
        assert abs(self._expect(bits, n_vertices=1, k=k)) < 1e-10

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_no_color_gives_positive_penalty(self, k):
        bits = [0] * k
        assert self._expect(bits, n_vertices=1, k=k) > 0

    def test_two_colors_gives_positive_penalty(self):
        assert self._expect([1, 1, 0], n_vertices=1, k=3) > 0


class TestCostHamiltonian:
    def _expect(self, bits: list[int], edges, n_vertices: int, k: int) -> float:
        W = make_weight_matrix(edges, n_vertices=n_vertices)
        H_cost, _ = build_maxkcut_hamiltonian_qutip(W, n_vertices=n_vertices, k=k)
        return float(np.real(qt.expect(H_cost, _basis(bits))))

    def test_cut_lower_than_noncut_k2(self):
        edges = [(0, 1)]
        cost_cut    = self._expect([1, 0, 0, 1], edges, n_vertices=2, k=2)
        cost_noncut = self._expect([1, 0, 1, 0], edges, n_vertices=2, k=2)
        assert cost_cut < cost_noncut
        assert abs(cost_noncut - cost_cut - 1.0) < 1e-10

    def test_cut_lower_than_noncut_k3(self):
        edges = [(0, 1)]
        cost_cut    = self._expect([1, 0, 0, 0, 1, 0], edges, n_vertices=2, k=3)
        cost_noncut = self._expect([1, 0, 0, 1, 0, 0], edges, n_vertices=2, k=3)
        assert cost_cut < cost_noncut
        assert abs(cost_noncut - cost_cut - 1.0) < 1e-10

    def test_weighted_edge_cost_difference(self):
        edges = [(0, 1, 3.0)]
        cost_cut    = self._expect([1, 0, 0, 1], edges, n_vertices=2, k=2)
        cost_noncut = self._expect([1, 0, 1, 0], edges, n_vertices=2, k=2)
        assert abs(cost_noncut - cost_cut - 3.0) < 1e-10


class TestHamiltonianScale:
    @pytest.mark.parametrize("n_vertices,k,edges,penalty", [
        (3, 3, [(0, 1, 2.5), (1, 2, 1.0)], 1.25),
        (2, 3, [(0, 1, 4.5)], 0.83),
        (3, 4, [(0, 1, 1.0), (0, 2, 3.0), (1, 2, 0.5)], 2.0),
        (2, 2, [], 0.5),
    ])
    def test_matches_exact_coefficients(self, n_vertices, k, edges, penalty):
        """scale exactly matches the maximum absolute linear Z / ZZ coefficient of the
        actually built H_problem = H_cost + penalty*H_penalty (coefficients are extracted
        independently from the diagonal via the Walsh-Hadamard transform and compared)."""
        W = make_weight_matrix(edges, n_vertices=n_vertices)
        H_cost, H_penalty = build_maxkcut_hamiltonian_qutip(W, n_vertices=n_vertices, k=k)
        H_problem = H_cost + penalty * H_penalty

        expected = _walsh_max_abs_coefficient(H_problem, n_vertices * k)
        actual = maxkcut_hamiltonian_scale(W, k, penalty)
        assert actual == pytest.approx(max(expected, 1.0), abs=1e-8)

    def test_cost_only_scale_for_cqa(self):
        """CQA does not use the penalty in its dynamics, so call with penalty=0.0."""
        edges = [(0, 1, 2.0), (1, 2, 4.0)]
        W = make_weight_matrix(edges, n_vertices=3)
        H_cost, _ = build_maxkcut_hamiltonian_qutip(W, n_vertices=3, k=3)

        expected = _walsh_max_abs_coefficient(H_cost, 3 * 3)
        actual = maxkcut_hamiltonian_scale(W, k=3, penalty=0.0)
        assert actual == pytest.approx(max(expected, 1.0), abs=1e-8)

    def test_never_below_one(self):
        W = make_weight_matrix([(0, 1, 0.01)], n_vertices=2)
        assert maxkcut_hamiltonian_scale(W, k=2, penalty=0.0) == pytest.approx(1.0)
