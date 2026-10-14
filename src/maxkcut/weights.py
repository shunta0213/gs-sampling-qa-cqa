from __future__ import annotations

import numpy as np


def make_random_edges(
    n_vertices: int,
    edge_prob: float = 0.5,
    weight_range: tuple[float, float] | None = None,
    seed: int | np.random.Generator | None = None,
) -> list:
    """
    Generate a random edge list using the Erdős–Rényi model.

    Parameters
    ----------
    n_vertices : int
    edge_prob : float
        Probability that each edge exists (0 < edge_prob <= 1)
    weight_range : (min, max) or None
        If None, weights are 1 (unweighted graph).
        If given, weights are drawn uniformly from [min, max].
    seed : int, np.random.Generator, or None
        Random seed for reproducibility. Passing an existing Generator continues
        consuming from where it left off (np.random.default_rng returns a Generator
        as-is, so passing the same object repeatedly continues the stream across calls).

    Returns
    -------
    list of (i, j) or (i, j, weight)
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < edge_prob:
                if weight_range is None:
                    edges.append((i, j))
                else:
                    w = float(rng.uniform(weight_range[0], weight_range[1]))
                    edges.append((i, j, w))
    return edges


def make_weight_matrix(edges: list, n_vertices: int) -> np.ndarray:
    """
    Build a weight matrix from an edge list.

    Parameters
    ----------
    edges : list of (i, j) or (i, j, weight)
    n_vertices : int

    Returns
    -------
    W : np.ndarray, shape (n_vertices, n_vertices), symmetric matrix
    """
    W = np.zeros((n_vertices, n_vertices))
    for edge in edges:
        i, j = edge[0], edge[1]
        w = float(edge[2]) if len(edge) > 2 else 1.0
        W[i, j] = w
        W[j, i] = w
    return W


def min_penalty(weight_matrix: np.ndarray) -> float:
    """
    Return a minimum penalty coefficient that reliably enforces the one-hot constraint.

    A > max(w_ij) is a necessary condition to avoid competing with the cost term.
    """
    return float(weight_matrix.max()) + 1.0


def analytical_penalty_coeff(weight_matrix: np.ndarray) -> float:
    """
    Return the analytical tight lower bound A* of the penalty coefficient for the Max-k-cut QUBO.

    A* is computed without enumerating all 2^(nk) states, so it can be used even when
    n or k is large.

    Derivation
    ----------
    The most dangerous infeasible state is one where vertex i has 2 colors (one-hot
    violation, E_P = 4) and none of its neighbors uses the extra color. The cost
    decrease in that case is -d_i / 2 (d_i = weighted degree of vertex i).

        A* = max_i d_i / 8

    Maximizing (E_f* - E_C(u)) / E_P(u) over all infeasible states, the worst case is a
    single-vertex violation whose extra color is not used by any neighbor, which gives
    this formula (common to all k ≥ 2).

    Parameters
    ----------
    weight_matrix : np.ndarray, shape (n, n)

    Returns
    -------
    float  — A* (use a value larger than this as the penalty)
    """
    weighted_degrees = weight_matrix.sum(axis=1)
    return float(weighted_degrees.max()) / 8.0
