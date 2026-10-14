from itertools import combinations

import numpy as np
import qutip as qt


def build_maxkcut_hamiltonian_qutip(
    weight_matrix: np.ndarray,
    n_vertices: int,
    k: int,
) -> tuple[qt.Qobj, qt.Qobj]:
    """
    Build the Ising Hamiltonian of Max-k-cut.

    Qubit layout
    ------------
    qubit q = i * k + c  (vertex i, color c)
    z_{i,c} = +1 → color c is not selected (|0⟩)
    z_{i,c} = -1 → color c is selected     (|1⟩)

    Cost Hamiltonian (minimization = cut maximization)
    --------------------------------------------------
    H_C = Σ_{i<j} W[i,j] Σ_c (I + Z_{ic} Z_{jc}) / 4

    Penalty Hamiltonian (one-hot constraint: Σ_c z_{i,c} = k-2, without coefficient)
    --------------------------------------------------------------------------------
    H_P = Σ_i (Σ_c Z_{ic} - (k-2))²

    The full Hamiltonian is H_C + A * H_P (A is the penalty coefficient).

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (n_vertices, n_vertices)
    n_vertices : int
    k : int — number of colors (number of partitions)

    Returns
    -------
    H_cost : qt.Qobj
    H_penalty : qt.Qobj
    """
    n_qubits = n_vertices * k

    def qubit(i: int, c: int) -> int:
        return i * k + c

    I = qt.tensor([qt.qeye(2)] * n_qubits)
    Z = [_single_qubit_z(q, n_qubits) for q in range(n_qubits)]

    H_cost = 0.0 * I
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            w = weight_matrix[i, j]
            if w == 0.0:
                continue
            for c in range(k):
                H_cost += (w / 4.0) * (I + Z[qubit(i, c)] * Z[qubit(j, c)])

    # H_P = Σ_i (S_i - (k-2))²,  S_i = Σ_c Z_{ic}
    # S_i² = k I + 2 Σ_{c<c'} Z_{ic} Z_{ic'}  (∵ Z² = I)
    H_penalty = 0.0 * I
    km2 = k - 2
    for i in range(n_vertices):
        S_sq = k * I + 2.0 * sum(
            Z[qubit(i, c)] * Z[qubit(i, cp)]
            for c, cp in combinations(range(k), 2)
        )
        S_lin = -2.0 * km2 * sum(Z[qubit(i, c)] for c in range(k))
        H_penalty += S_sq + S_lin + km2**2 * I

    return H_cost, H_penalty


def _single_qubit_z(qubit_idx: int, n_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2)] * n_qubits
    ops[qubit_idx] = qt.sigmaz()
    return qt.tensor(ops)


def maxkcut_hamiltonian_scale(weight_matrix: np.ndarray, k: int, penalty: float) -> float:
    """
    Return the maximum absolute Z_iZ_j / Z_ic coefficient of H_cost + penalty * H_penalty.

    Using the same expansion as build_maxkcut_hamiltonian_qutip, dividing the operator
    used for the time evolution by this value puts all coefficients within [-1, 1] (same
    idea as the scale computed by build_ising in sa-maxkcut.py).

    Derivation (see build_maxkcut_hamiltonian_qutip)
    ------------------------------------------------
    H_cost    → inter-vertex, same-color ZZ coefficient      : w_ij / 4
    H_penalty → intra-vertex, different-color ZZ coefficient : 2
                intra-vertex linear Z coefficient            : -2(k-2)

    When the penalty is not used in the dynamics, as in CQA, pass penalty=0.0
    (the scale is then computed from the H_cost terms only).

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (n, n)
    k : int — number of colors
    penalty : float — coefficient in H_problem = H_cost + penalty * H_penalty (0.0 if unused)

    Returns
    -------
    float — scale (never less than 1.0)
    """
    inter_vertex_zz = float(np.max(np.abs(weight_matrix))) / 4.0 if weight_matrix.size else 0.0
    intra_vertex_zz = abs(2.0 * penalty)
    linear = abs(2.0 * penalty * (k - 2))
    return max(inter_vertex_zz, intra_vertex_zz, linear, 1.0)
