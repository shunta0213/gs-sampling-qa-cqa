import numpy as np
import qutip as qt


def solve_gbp_qutip(
    H_cost: qt.Qobj,
    H_penalty: qt.Qobj,
    n_vertices: int,
    penalty: float,
) -> list[list[int]]:
    """
    Enumerate the optimal partition assignments of the Graph Bipartitioning Problem.

    H_cost and H_penalty are both diagonal in the computational basis (ZZ terms only),
    so the minimum diagonal element is searched directly without eigendecomposition.
    Complexity: O(2^n)

    Parameters
    ----------
    H_cost, H_penalty : qt.Qobj
        Return values of build_gbp_hamiltonian_qutip
    n_vertices : int
    penalty : float
        Penalty coefficient A (a value larger than exact_min_penalty's return value is recommended)

    Returns
    -------
    list[list[int]]
        Partition assignments of all degenerate optimal solutions (0/1, only those satisfying the balance constraint)
    """
    diag_c = np.real(H_cost.diag())
    diag_p = np.real(H_penalty.diag())
    return solve_gbp_from_diag(diag_c, diag_p, n_vertices, penalty)


def solve_gbp_from_diag(
    diag_c: np.ndarray,
    diag_p: np.ndarray,
    n_vertices: int,
    penalty: float,
) -> list[list[int]]:
    """Core logic of solve_gbp_qutip. Enumerates directly from numpy arrays of the
    diagonals without going through qt.Qobj (use together with hamiltonian.gbp_diag_numpy).
    """
    total = diag_c + penalty * diag_p

    ground_energy = total.min()
    ground_indices = np.where(np.abs(total - ground_energy) < 1e-6)[0]

    return [_decode_from_index(int(idx), n_vertices) for idx in ground_indices]


def _decode_from_index(idx: int, n_vertices: int) -> list[int]:
    """Decode the partition assignment from a computational basis state index."""
    return [(idx >> (n_vertices - 1 - i)) & 1 for i in range(n_vertices)]
