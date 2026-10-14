import numpy as np
import pytest
import qutip as qt

from src.gbp import (
    build_gbp_feasible_driver_ground_state_qutip,
    build_gbp_feasible_uniform_state_qutip,
    build_gbp_xy_driver_qutip,
)


def _basis(bits: list[int]) -> qt.Qobj:
    return qt.tensor([qt.basis(2, b) for b in bits])


def _z_total(n_qubits: int) -> qt.Qobj:
    return sum(
        qt.tensor([qt.sigmaz() if q == i else qt.qeye(2) for q in range(n_qubits)])
        for i in range(n_qubits)
    )


class TestGbpXYDriver:
    def test_dimension(self):
        h_driver = build_gbp_xy_driver_qutip(n_vertices=4)
        assert h_driver.shape == (16, 16)

    @pytest.mark.parametrize("topology", ["complete", "chain", "ring"])
    def test_hermitian(self, topology):
        h_driver = build_gbp_xy_driver_qutip(n_vertices=4, topology=topology)
        assert (h_driver - h_driver.dag()).norm() < 1e-10

    @pytest.mark.parametrize("topology", ["complete", "chain", "ring"])
    def test_preserves_total_magnetization(self, topology):
        h_driver = build_gbp_xy_driver_qutip(n_vertices=4, topology=topology)
        z_total = _z_total(4)
        assert (h_driver * z_total - z_total * h_driver).norm() < 1e-10

    def test_chain_only_connects_adjacent_vertices(self):
        h_driver = build_gbp_xy_driver_qutip(n_vertices=4, topology="chain")
        source = _basis([1, 0, 0, 0])

        adjacent = _basis([0, 1, 0, 0]).dag() * h_driver * source
        non_adjacent = _basis([0, 0, 1, 0]).dag() * h_driver * source

        assert abs(complex(adjacent) + 1.0) < 1e-10
        assert abs(complex(non_adjacent)) < 1e-10

    def test_ring_adds_periodic_boundary_edge(self):
        h_driver = build_gbp_xy_driver_qutip(n_vertices=4, topology="ring")
        source = _basis([1, 0, 0, 0])
        periodic_neighbor = _basis([0, 0, 0, 1])

        amplitude = periodic_neighbor.dag() * h_driver * source
        assert abs(complex(amplitude) + 1.0) < 1e-10

    def test_ring_does_not_duplicate_two_vertex_edge(self):
        h_driver = build_gbp_xy_driver_qutip(n_vertices=2, topology="ring")
        amplitude = _basis([0, 1]).dag() * h_driver * _basis([1, 0])
        assert abs(complex(amplitude) + 1.0) < 1e-10

    def test_rejects_unknown_topology(self):
        with np.testing.assert_raises_regex(ValueError, "Unknown XY topology"):
            build_gbp_xy_driver_qutip(n_vertices=4, topology="grid")


class TestGbpFeasibleUniformState:
    def test_normalized(self):
        psi0 = build_gbp_feasible_uniform_state_qutip(n_vertices=4)
        assert abs(psi0.norm() - 1.0) < 1e-10

    def test_uniform_over_balanced_states(self):
        n = 4
        psi0 = build_gbp_feasible_uniform_state_qutip(n_vertices=n)
        probs = np.abs(psi0.full().flatten()) ** 2
        nonzero = np.where(probs > 1e-12)[0]

        expected_count = 6  # C(4, 2)
        assert len(nonzero) == expected_count
        assert np.allclose(probs[nonzero], 1.0 / expected_count)
        assert all(bin(idx)[2:].zfill(n).count("1") == n // 2 for idx in nonzero)

    def test_is_ground_state_of_complete_driver(self):
        n = 4
        h_driver = build_gbp_xy_driver_qutip(n_vertices=n)
        psi0 = build_gbp_feasible_uniform_state_qutip(n_vertices=n)
        energy = qt.expect(h_driver, psi0)

        assert (h_driver * psi0 - energy * psi0).norm() < 1e-10
        assert energy == pytest.approx(np.real(h_driver.eigenenergies()).min())


class TestGbpFeasibleDriverGroundState:
    @pytest.mark.parametrize("topology", ["complete", "chain", "ring"])
    def test_is_balanced_eigenstate_of_its_driver(self, topology):
        n = 4
        h_driver = build_gbp_xy_driver_qutip(n_vertices=n, topology=topology)
        psi0 = build_gbp_feasible_driver_ground_state_qutip(n_vertices=n, topology=topology)
        energy = qt.expect(h_driver, psi0)

        assert abs(psi0.norm() - 1.0) < 1e-10
        assert (h_driver * psi0 - energy * psi0).norm() < 1e-10

        probs = np.abs(psi0.full().flatten()) ** 2
        nonzero = np.where(probs > 1e-12)[0]
        assert all(bin(idx).count("1") == n // 2 for idx in nonzero)

    def test_complete_uses_uniform_state(self):
        n = 4
        uniform = build_gbp_feasible_uniform_state_qutip(n_vertices=n)
        psi0 = build_gbp_feasible_driver_ground_state_qutip(n_vertices=n, topology="complete")
        assert abs(complex(uniform.dag() * psi0)) == pytest.approx(1.0)

    def test_chain_ground_state_is_lowest_among_balanced_energies(self):
        n = 4
        h_driver = build_gbp_xy_driver_qutip(n_vertices=n, topology="chain")
        psi0 = build_gbp_feasible_driver_ground_state_qutip(n_vertices=n, topology="chain")
        energy = qt.expect(h_driver, psi0)

        balanced_indices = [idx for idx in range(2**n) if bin(idx).count("1") == n // 2]
        sub_matrix = h_driver.full()[np.ix_(balanced_indices, balanced_indices)]
        min_balanced_energy = np.linalg.eigvalsh(sub_matrix).min()

        assert energy == pytest.approx(min_balanced_energy)
