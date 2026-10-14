import numpy as np
import qutip as qt

from src.maxkcut import (
    build_maxkcut_feasible_driver_ground_state_qutip,
    build_maxkcut_feasible_uniform_state_qutip,
    build_maxkcut_xy_driver_qutip,
)


def _basis(bits: list[int]) -> qt.Qobj:
    return qt.tensor([qt.basis(2, bit) for bit in bits])


def _vertex_weights(idx: int, n_vertices: int, k: int) -> tuple[int, ...]:
    n_qubits = n_vertices * k
    weights = []
    for i in range(n_vertices):
        total = 0
        for c in range(k):
            q = i * k + c
            total += (idx >> (n_qubits - 1 - q)) & 1
        weights.append(total)
    return tuple(weights)


class TestMaxkcutXYDriver:
    def test_dimension(self):
        n_vertices, k = 2, 3
        h_driver = build_maxkcut_xy_driver_qutip(n_vertices, k)
        expected_dim = 2 ** (n_vertices * k)
        assert h_driver.shape == (expected_dim, expected_dim)

    def test_hermitian(self):
        h_driver = build_maxkcut_xy_driver_qutip(n_vertices=2, k=3)
        assert (h_driver - h_driver.dag()).norm() < 1e-10

    def test_preserves_vertex_excitation_counts(self):
        n_vertices, k = 2, 3
        h_driver = build_maxkcut_xy_driver_qutip(n_vertices, k)
        rows, cols = np.nonzero(np.abs(h_driver.full()) > 1e-12)

        for row, col in zip(rows, cols):
            assert _vertex_weights(row, n_vertices, k) == _vertex_weights(
                col, n_vertices, k
            )

    def test_moves_one_hot_color_with_negative_amplitude(self):
        h_driver = build_maxkcut_xy_driver_qutip(n_vertices=1, k=3)
        source = _basis([1, 0, 0])
        target = _basis([0, 1, 0])

        amplitude = target.dag() * h_driver * source
        assert abs(complex(amplitude) + 1.0) < 1e-10

    def test_chain_only_connects_adjacent_colors(self):
        h_driver = build_maxkcut_xy_driver_qutip(
            n_vertices=1, k=4, topology="chain"
        )
        source = _basis([1, 0, 0, 0])

        adjacent = _basis([0, 1, 0, 0]).dag() * h_driver * source
        non_adjacent = _basis([0, 0, 0, 1]).dag() * h_driver * source

        assert abs(complex(adjacent) + 1.0) < 1e-10
        assert abs(complex(non_adjacent)) < 1e-10

    def test_ring_adds_periodic_boundary_edge(self):
        h_driver = build_maxkcut_xy_driver_qutip(
            n_vertices=1, k=4, topology="ring"
        )
        source = _basis([1, 0, 0, 0])
        periodic_neighbor = _basis([0, 0, 0, 1])

        amplitude = periodic_neighbor.dag() * h_driver * source
        assert abs(complex(amplitude) + 1.0) < 1e-10

    def test_ring_does_not_duplicate_two_color_edge(self):
        h_driver = build_maxkcut_xy_driver_qutip(
            n_vertices=1, k=2, topology="ring"
        )
        amplitude = _basis([0, 1]).dag() * h_driver * _basis([1, 0])
        assert abs(complex(amplitude) + 1.0) < 1e-10

    def test_rejects_unknown_topology(self):
        with np.testing.assert_raises_regex(ValueError, "Unknown XY topology"):
            build_maxkcut_xy_driver_qutip(n_vertices=1, k=3, topology="grid")


class TestMaxkcutFeasibleUniformState:
    def test_normalized(self):
        psi0 = build_maxkcut_feasible_uniform_state_qutip(n_vertices=2, k=3)
        assert abs(psi0.norm() - 1.0) < 1e-10

    def test_uniform_over_feasible_one_hot_states(self):
        n_vertices, k = 2, 3
        psi0 = build_maxkcut_feasible_uniform_state_qutip(n_vertices, k)
        probs = np.abs(psi0.full().flatten()) ** 2
        nonzero = np.where(probs > 1e-12)[0]

        assert len(nonzero) == k**n_vertices
        assert np.allclose(probs[nonzero], 1.0 / (k**n_vertices))
        assert all(_vertex_weights(idx, n_vertices, k) == (1, 1) for idx in nonzero)


class TestMaxkcutFeasibleDriverGroundState:
    def test_is_eigenstate_of_each_topology_driver(self):
        for topology in ("complete", "chain", "ring"):
            h_driver = build_maxkcut_xy_driver_qutip(
                n_vertices=2, k=4, topology=topology
            )
            psi0 = build_maxkcut_feasible_driver_ground_state_qutip(
                n_vertices=2, k=4, topology=topology
            )
            energy = qt.expect(h_driver, psi0)

            assert (h_driver * psi0 - energy * psi0).norm() < 1e-10

    def test_chain_has_sine_shaped_color_probabilities(self):
        psi0 = build_maxkcut_feasible_driver_ground_state_qutip(
            n_vertices=1, k=4, topology="chain"
        )
        probs = np.abs(psi0.full().flatten()) ** 2

        assert np.allclose(probs[[8, 4, 2, 1]], [0.1381966, 0.3618034, 0.3618034, 0.1381966])

    def test_complete_and_ring_use_uniform_state(self):
        uniform = build_maxkcut_feasible_uniform_state_qutip(n_vertices=1, k=4)
        for topology in ("complete", "ring"):
            psi0 = build_maxkcut_feasible_driver_ground_state_qutip(
                n_vertices=1, k=4, topology=topology
            )
            assert abs(complex(uniform.dag() * psi0)) == 1.0
