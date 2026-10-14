import numpy as np
import qutip as qt


def build_gbp_hamiltonian_qutip(
    weight_matrix: np.ndarray,
    n_vertices: int,
) -> tuple[qt.Qobj, qt.Qobj]:
    """
    Build the Ising Hamiltonian of the Graph Bipartitioning Problem.

    Qubit layout
    ------------
    qubit i = vertex i
    z_i = +1 → vertex i is in group 0
    z_i = -1 → vertex i is in group 1

    Cost Hamiltonian (minimization = cut minimization)
    --------------------------------------------------
    H_C = Σ_{i<j} W[i,j] (I - Z_i Z_j) / 2

    Penalty Hamiltonian (balance constraint: Σ_i Z_i = 0, without coefficient)
    --------------------------------------------------------------------------
    H_P = (Σ_i Z_i)²

    The full Hamiltonian is H_C + A * H_P (A is the penalty coefficient).

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (n_vertices, n_vertices)
    n_vertices : int — must be even (for a balanced bipartition)

    Returns
    -------
    H_cost : qt.Qobj
    H_penalty : qt.Qobj
    """
    if n_vertices % 2 != 0:
        raise ValueError("n_vertices must be even for balanced bipartitioning")

    I = qt.tensor([qt.qeye(2)] * n_vertices)
    Z = [_single_qubit_z(q, n_vertices) for q in range(n_vertices)]

    H_cost = 0.0 * I
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            w = weight_matrix[i, j]
            if w == 0.0:
                continue
            H_cost += (w / 2.0) * (I - Z[i] * Z[j])

    S = sum(Z)
    H_penalty = S * S

    return H_cost, H_penalty


def gbp_diag_numpy(
    weight_matrix: np.ndarray,
    n_vertices: int,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the same diagonals (diag_c, diag_p) as the H_cost / H_penalty returned by
    build_gbp_hamiltonian_qutip, directly with numpy and without going through qt.Qobj.

    The qutip version builds and sums a 2^n-dimensional sparse operator (tensor product)
    for every vertex pair, so for uses that only need the diagonal (exact_min_penalty,
    ground-state enumeration) the tensor-product construction cost itself dominates and
    becomes extremely slow for large n_vertices (about 10 s per call at n=20; at n=24 it
    does not finish in practical time).

    This version generates all 2^n computational basis states at once via bit expansion
    and computes energies with matrix products, so it is 1-2 orders of magnitude faster
    for the same n_vertices (under 1 s at n=20). It especially pays off in cases like
    search_problem, which rebuilds the Hamiltonian many times just to check the degeneracy.

    Time and memory are both O(2^n), same as the qutip version (unavoidable when exactly
    scanning all 2^n states). For even larger n_vertices, chunk_size processes the basis
    states block by block to reduce peak memory.

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (n_vertices, n_vertices)
    n_vertices : int — must be even (for a balanced bipartition)
    chunk_size : int | None
        Maximum number of basis states expanded at once. If None, all 2**n_vertices are processed at once.

    Returns
    -------
    diag_c, diag_p : np.ndarray, shape (2**n_vertices,)
    """
    if n_vertices % 2 != 0:
        raise ValueError("n_vertices must be even for balanced bipartitioning")

    n_states = 2**n_vertices
    total_w = float(weight_matrix.sum())
    shifts = np.arange(n_vertices - 1, -1, -1, dtype=np.uint64)

    diag_c = np.empty(n_states, dtype=np.float64)
    diag_p = np.empty(n_states, dtype=np.float64)

    step = chunk_size or n_states
    for start in range(0, n_states, step):
        stop = min(start + step, n_states)
        idx = np.arange(start, stop, dtype=np.uint64)
        bits = ((idx[:, None] >> shifts) & 1).astype(np.int8)
        z = (1 - 2 * bits).astype(np.float64)
        quad_full = (z @ weight_matrix * z).sum(axis=1)
        diag_c[start:stop] = (total_w - quad_full) / 4.0
        diag_p[start:stop] = z.sum(axis=1) ** 2

    return diag_c, diag_p


def _single_qubit_z(qubit_idx: int, n_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2)] * n_qubits
    ops[qubit_idx] = qt.sigmaz()
    return qt.tensor(ops)


def gbp_hamiltonian_scale(weight_matrix: np.ndarray, penalty: float) -> float:
    """
    Return the maximum absolute Z_iZ_j coefficient of H_cost + penalty * H_penalty (no linear terms).

    Using the same expansion as build_gbp_hamiltonian_qutip, dividing the operator used
    for the time evolution by this value puts all coefficients within [-1, 1] (same idea
    as the scale computed by build_ising in sa-maxkcut.py).

    Derivation (see build_gbp_hamiltonian_qutip)
    --------------------------------------------
    The ZZ coefficient for each vertex pair (i,j) is -w_ij / 2 + 2 * penalty (H_penalty
    acts on every vertex pair regardless of whether an edge exists, so it adds up with
    the H_cost term on the same pair). Unlike Max-k-Cut, H_cost and H_penalty do not act
    on separate qubit pairs, so instead of taking the max of each term independently, the
    actual combined coefficient must be computed per vertex pair before taking the max
    absolute value (otherwise cases where terms of opposite sign cancel would be
    overestimated). There are no linear terms.

    When the penalty is not used in the dynamics, as in CQA, pass penalty=0.0
    (the scale is then computed from the H_cost terms only).

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (n, n)
    penalty : float — coefficient in H_problem = H_cost + penalty * H_penalty (0.0 if unused)

    Returns
    -------
    float — scale (never less than 1.0)
    """
    n = weight_matrix.shape[0]
    i, j = np.triu_indices(n, k=1)
    combined_zz = -weight_matrix[i, j] / 2.0 + 2.0 * penalty
    max_zz = float(np.max(np.abs(combined_zz))) if combined_zz.size else 0.0
    return max(max_zz, 1.0)
