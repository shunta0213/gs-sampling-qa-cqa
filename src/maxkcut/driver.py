from itertools import combinations
from typing import Literal

import numpy as np
import qutip as qt

XYTopology = Literal["complete", "chain", "ring"]


def build_maxkcut_xy_driver_qutip(n_vertices: int, k: int, topology: XYTopology = "complete") -> qt.Qobj:
    """
    Build the XY driver for Max-k-cut that preserves the one-hot feasible subspace.

    The qubit layout is the same as build_maxkcut_hamiltonian_qutip:
    qubit q = i * k + c  (vertex i, color c)

    For topology="complete":
        H_D = -1/2 Σ_i Σ_{c<c'} (X_{i,c} X_{i,c'} + Y_{i,c} Y_{i,c'})

    topology="chain" couples the color qubits as an open-boundary 1D chain,
    and topology="ring" as a 1D chain with periodic boundary.

    The number of excitations within each vertex is conserved, so starting from a
    one-hot initial state, the time evolution stays within the feasible subspace.
    """
    n_qubits = n_vertices * k
    identity = qt.tensor([qt.qeye(2)] * n_qubits)
    x_ops = [_single_qubit_op_qutip(qt.sigmax(), q, n_qubits) for q in range(n_qubits)]
    y_ops = [_single_qubit_op_qutip(qt.sigmay(), q, n_qubits) for q in range(n_qubits)]

    h_driver = 0.0 * identity
    color_edges = _xy_color_edges(k, topology)
    for i in range(n_vertices):
        for c, cp in color_edges:
            q = i * k + c
            qp = i * k + cp
            h_driver += -0.5 * (x_ops[q] * x_ops[qp] + y_ops[q] * y_ops[qp])

    return h_driver


def _xy_color_edges(k: int, topology: XYTopology) -> tuple[tuple[int, int], ...]:
    if topology == "complete":
        return tuple(combinations(range(k), 2))
    if topology == "chain":
        return tuple((c, c + 1) for c in range(k - 1))
    if topology == "ring":
        # For k=2, the closing edge coincides with the only chain edge.
        return tuple((c, (c + 1) % k) for c in range(k)) if k > 2 else tuple((c, c + 1) for c in range(k - 1))
    raise ValueError(f"Unknown XY topology: {topology!r}. Choose one of: 'complete', 'chain', 'ring'.")


def build_maxkcut_feasible_uniform_state_qutip(n_vertices: int, k: int) -> qt.Qobj:
    """
    Build the uniform superposition of all feasible states where every vertex is one-hot.

    Builds (|100...0> + |010...0> + ... + |000...1>) / sqrt(k) per vertex
    and takes the tensor product of n_vertices copies.
    """
    vertex_terms = []
    for c in range(k):
        bits = [0] * k
        bits[c] = 1
        vertex_terms.append(qt.tensor([qt.basis(2, bit) for bit in bits]))

    vertex_state = sum(vertex_terms).unit()
    return qt.tensor([vertex_state] * n_vertices)


def build_maxkcut_feasible_driver_ground_state_qutip(n_vertices: int, k: int, topology: XYTopology = "complete") -> qt.Qobj:
    """Build the ground state of the given XY driver within the one-hot feasible subspace."""
    _xy_color_edges(k, topology)  # Validate topology.

    if topology in ("complete", "ring"):
        return build_maxkcut_feasible_uniform_state_qutip(n_vertices, k)

    vertex_state = sum(np.sin((c + 1) * np.pi / (k + 1)) * qt.tensor([qt.basis(2, int(bit)) for bit in _one_hot_bits(k, c)]) for c in range(k)).unit()
    return qt.tensor([vertex_state] * n_vertices)


def _one_hot_bits(k: int, selected_color: int) -> list[int]:
    bits = [0] * k
    bits[selected_color] = 1
    return bits


def _single_qubit_op_qutip(op: qt.Qobj, qubit_idx: int, n_qubits: int) -> qt.Qobj:
    ops = [qt.qeye(2)] * n_qubits
    ops[qubit_idx] = op
    return qt.tensor(ops)
