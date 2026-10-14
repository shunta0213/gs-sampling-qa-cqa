from itertools import combinations
from typing import Literal

import numpy as np
import qutip as qt


XYTopology = Literal["complete", "chain", "ring"]


def build_gbp_xy_driver_qutip(n_vertices: int, topology: XYTopology = "complete") -> qt.Qobj:
    """
    Build the XY driver for Graph Bipartitioning that preserves the balance constraint (Σ_i Z_i = 0).

        H_D = -1/2 Σ_{(i,j) ∈ edges} (X_i X_j + Y_i Y_j)

    topology="complete" couples all vertex pairs, "chain" couples vertices as an
    open-boundary 1D chain, and "ring" as a 1D chain with periodic boundary.

    An XY coupling always conserves the sum of the magnetizations of the coupled pair,
    so the total magnetization Σ_i Z_i is conserved regardless of topology. Starting
    from a balanced initial state, the time evolution stays within the balanced subspace.
    """
    n_qubits = n_vertices
    identity = qt.tensor([qt.qeye(2)] * n_qubits)
    x_ops = [_single_qubit_op_qutip(qt.sigmax(), q, n_qubits) for q in range(n_qubits)]
    y_ops = [_single_qubit_op_qutip(qt.sigmay(), q, n_qubits) for q in range(n_qubits)]

    h_driver = 0.0 * identity
    for i, j in _xy_topology_edges(n_qubits, topology):
        h_driver += -0.5 * (x_ops[i] * x_ops[j] + y_ops[i] * y_ops[j])

    return h_driver


def _xy_topology_edges(n_vertices: int, topology: XYTopology) -> tuple[tuple[int, int], ...]:
    if topology == "complete":
        return tuple(combinations(range(n_vertices), 2))
    if topology == "chain":
        return tuple((i, i + 1) for i in range(n_vertices - 1))
    if topology == "ring":
        # For n_vertices=2, the closing edge coincides with the only chain edge.
        return tuple((i, (i + 1) % n_vertices) for i in range(n_vertices)) if n_vertices > 2 else tuple(
            (i, i + 1) for i in range(n_vertices - 1)
        )
    raise ValueError(
        f"Unknown XY topology: {topology!r}. "
        "Choose one of: 'complete', 'chain', 'ring'."
    )


def build_gbp_feasible_uniform_state_qutip(n_vertices: int) -> qt.Qobj:
    """
    Build the uniform superposition of all feasible states satisfying the balance
    constraint (n_vertices/2 vertices in each group).

    This coincides with the ground state of the XY driver on the complete graph
    (topology="complete"), i.e. the symmetric Dicke state.
    """
    if n_vertices % 2 != 0:
        raise ValueError("n_vertices must be even for balanced bipartitioning")

    half = n_vertices // 2
    terms = []
    for ones in combinations(range(n_vertices), half):
        bits = [0] * n_vertices
        for p in ones:
            bits[p] = 1
        terms.append(qt.tensor([qt.basis(2, b) for b in bits]))

    return sum(terms).unit()


def build_gbp_feasible_driver_ground_state_qutip(
    n_vertices: int, topology: XYTopology = "complete"
) -> qt.Qobj:
    """
    Build the ground state of the given XY driver within the balanced subspace.

    For topology="complete" it coincides with the uniform superposition by symmetry and
    has a closed form; "chain" and "ring" have no closed form, so the driver is
    restricted to the balanced subspace and diagonalized numerically.
    """
    if topology == "complete":
        return build_gbp_feasible_uniform_state_qutip(n_vertices)

    h_driver = build_gbp_xy_driver_qutip(n_vertices, topology)
    balanced_indices = _balanced_basis_indices(n_vertices)

    sub_matrix = h_driver.full()[np.ix_(balanced_indices, balanced_indices)]
    eigvals, eigvecs = np.linalg.eigh(sub_matrix)
    ground_vec = eigvecs[:, np.argmin(eigvals)]

    full_vec = np.zeros(2**n_vertices, dtype=complex)
    full_vec[balanced_indices] = ground_vec
    return qt.Qobj(full_vec, dims=[[2] * n_vertices, [1] * n_vertices]).unit()


def _balanced_basis_indices(n_vertices: int) -> list[int]:
    if n_vertices % 2 != 0:
        raise ValueError("n_vertices must be even for balanced bipartitioning")

    half = n_vertices // 2
    return [idx for idx in range(2**n_vertices) if bin(idx).count("1") == half]


def _single_qubit_op_qutip(op: qt.Qobj, qubit_idx: int, n_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2)] * n_qubits
    ops[qubit_idx] = op
    return qt.tensor(ops)
