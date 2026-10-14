import numpy as np
import qutip as qt


def solve_maxkcut_qutip(
    H_cost: qt.Qobj,
    H_penalty: qt.Qobj,
    n_vertices: int,
    k: int,
    penalty: float,
) -> list[list[int]]:
    """
    Enumerate the optimal color assignments of Max-k-cut.

    H_cost and H_penalty are both diagonal in the computational basis (ZZ terms only),
    so the minimum diagonal element is searched directly without eigendecomposition.
    Complexity: O(2^{n*k})

    Parameters
    ----------
    H_cost, H_penalty : qt.Qobj
        Return values of build_maxkcut_hamiltonian_qutip
    n_vertices : int
    k : int
    penalty : float
        Penalty coefficient A (a value larger than exact_min_penalty's return value is recommended)

    Returns
    -------
    list[list[int]]
        Color assignments of all degenerate optimal solutions (deduplicated, one-hot violations excluded)
    """
    diag_c = np.real(H_cost.diag())
    diag_p = np.real(H_penalty.diag())
    total = diag_c + penalty * diag_p

    ground_energy = total.min()
    ground_indices = np.where(np.abs(total - ground_energy) < 1e-6)[0]

    n_qubits = n_vertices * k
    assignments = []
    seen: set[tuple[int, ...]] = set()

    for idx in ground_indices:
        a = _decode_from_index(int(idx), n_qubits, n_vertices, k)
        key = tuple(a)
        if key in seen or -1 in a:
            continue
        seen.add(key)
        assignments.append(a)

    return assignments


def exact_min_penalty(H_cost: qt.Qobj, H_penalty: qt.Qobj) -> float:
    """
    For H = H_cost + A * H_penalty, compute exactly the minimum penalty coefficient A*
    at which the energies of infeasible and feasible solutions coincide.

    H_cost and H_penalty are both diagonal in the computational basis (ZZ terms only),
    so the threshold is derived analytically by directly comparing the energies of each
    basis state.

        A* = max_{u: E_P(u)>0, E_C(u)≤E_f*} (E_f* - E_C(u)) / E_P(u)
        E_f* = min_{f: E_P(f)=0} E_C(f)

    For A > A*, every feasible solution has lower energy than every infeasible solution.
    For A = A*, the worst infeasible solution has the same energy as the best feasible solution.

    Parameters
    ----------
    H_cost, H_penalty : qt.Qobj
        Return values of build_maxkcut_hamiltonian_qutip (without penalty coefficient)

    Returns
    -------
    float  — A*
    """
    diag_c = np.real(H_cost.diag())
    diag_p = np.real(H_penalty.diag())
    return exact_min_penalty_from_diag(diag_c, diag_p)


def exact_min_penalty_from_diag(diag_c: np.ndarray, diag_p: np.ndarray) -> float:
    """Core logic of exact_min_penalty. Computes A* directly from numpy arrays of the
    diagonals without going through qt.Qobj (use when n is large and building qt.Qobj
    itself becomes the bottleneck, e.g. diag_c/diag_p returned by gbp.hamiltonian.gbp_diag_numpy).
    """
    feasible = np.abs(diag_p) < 1e-10
    if not np.any(feasible):
        raise ValueError("No feasible state exists")

    E_f_star = diag_c[feasible].min()

    infeasible = ~feasible
    diag_c_u = diag_c[infeasible]
    diag_p_u = diag_p[infeasible]

    # Infeasible solutions that require a threshold: E_C(u) ≤ E_f*
    needs = diag_c_u <= E_f_star + 1e-10
    if not np.any(needs):
        return 0.0

    return float(np.max((E_f_star - diag_c_u[needs]) / diag_p_u[needs]))


def compute_cut_value(assignment: list[int], weight_matrix: np.ndarray) -> float:
    """Compute the cut value (total weight of cut edges) from a color assignment and the graph weight matrix."""
    n = len(assignment)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if assignment[i] != assignment[j]:
                cut += weight_matrix[i, j]
    return cut


def _decode_from_index(idx: int, n_qubits: int, n_vertices: int, k: int) -> list[int]:
    """Decode the color assignment from a computational basis state index."""
    assignment = []
    for i in range(n_vertices):
        color = None
        for c in range(k):
            bit = (idx >> (n_qubits - 1 - (i * k + c))) & 1
            if bit == 1:
                color = c if color is None else -1
        assignment.append(-1 if color is None else color)
    return assignment
