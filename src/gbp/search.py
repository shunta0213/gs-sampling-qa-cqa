"""Problem-instance search logic for Graph Bipartitioning.

Shared by qa-gbp.py / cqa-xy-gbp.py (and their respective add-tasks scripts).
"""

import functools

import numpy as np

from ..maxkcut import exact_min_penalty_from_diag, make_random_edges, make_weight_matrix
from .hamiltonian import build_gbp_hamiltonian_qutip, gbp_diag_numpy
from .solver import solve_gbp_from_diag


def search_problem(
    n_vertices: int,
    edge_prob: float,
    weight_range: tuple[float, float] | None,
    seed: int,
    penalty_offset: float,
    min_degenerate: int,
    max_tries: int,
    build_qutip: bool = True,
    diag_chunk_size: int | None = None,
):
    """Regenerate graphs from the seed's RNG stream until there are at least
    min_degenerate degenerate optimal solutions.

    However, a graph whose weight matrix W exactly matches one adopted for a smaller seed
    (0, 1, ..., seed - 1) is rejected and redrawn from the same stream. Since each seed
    uses an independent stream, for small N (e.g. N=6) different seeds can yield the same
    W, which would also share the same output location (the hash of W). The check depends
    only on the results for smaller seeds, so adding more seeds does not change the W of
    existing seeds. Isomorphic graphs (differing only by vertex relabeling) are not
    excluded (the difference from v2's src/gbp/instances.py).

    The search loop itself uses direct numpy computation (hamiltonian.gbp_diag_numpy)
    without going through qt.Qobj. The cost of building the tensor-product Hamiltonian
    with qt.Qobj dominates for large n_vertices (about 10 s per call at n=20), and
    building it on every trial in the loop adds up over max_tries and becomes impractical
    (at n=24 even a single call may not finish). Numerically it matches the diagonal of
    qt.Qobj exactly, so it alone suffices for exact_min_penalty and enumerating degenerate
    solutions.

    The qt.Qobj objects themselves (H_cost, H_penalty) are unnecessary for algorithms
    that only need the classical Ising J matrix, such as SA and SQA. Only algorithms that
    actually run quantum dynamics, such as QA and CQA, keep build_qutip=True, and they are
    built for the found graph only (once, outside the loop).

    Parameters
    ----------
    build_qutip : bool
        If True, call build_gbp_hamiltonian_qutip once after a graph satisfying the
        degeneracy condition is found, and return H_cost / H_penalty.
        If False, return None for H_cost / H_penalty (no qt.Qobj is built at all,
        so it is fast even for large n_vertices).
    diag_chunk_size : int | None
        Chunk size passed to gbp_diag_numpy (use it to reduce peak memory).
    """
    key = (n_vertices, edge_prob, weight_range, penalty_offset, min_degenerate, max_tries, diag_chunk_size)
    # Solve in order from smaller seeds (each seed's result is cached, so recursion stays shallow)
    for earlier_seed in range(seed):
        _search_unique(*key, earlier_seed)
    found = _search_unique(*key, seed)
    if found is None:
        raise RuntimeError(
            f"No problem with degeneracy >= {min_degenerate} that does not duplicate a smaller seed "
            f"was found after {max_tries} tries from the RNG stream of seed {seed}"
        )

    edges, W, penalty_min, penalty, assignments = found
    H_cost, H_penalty = build_gbp_hamiltonian_qutip(W, n_vertices) if build_qutip else (None, None)
    return seed, list(edges), W.copy(), H_cost, H_penalty, penalty_min, penalty, list(assignments)


@functools.cache
def _search_unique(
    n_vertices: int,
    edge_prob: float,
    weight_range: tuple[float, float] | None,
    penalty_offset: float,
    min_degenerate: int,
    max_tries: int,
    diag_chunk_size: int | None,
    seed: int,
):
    """Core of search_problem. Returns None if not found (None is cached too)."""
    earlier_weights = set()
    for earlier_seed in range(seed):
        earlier = _search_unique(
            n_vertices, edge_prob, weight_range, penalty_offset, min_degenerate, max_tries, diag_chunk_size, earlier_seed
        )
        if earlier is not None:
            earlier_weights.add(earlier[1].tobytes())

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        edges = make_random_edges(n_vertices, edge_prob, weight_range, rng)
        W = make_weight_matrix(edges, n_vertices)
        diag_c, diag_p = gbp_diag_numpy(W, n_vertices, chunk_size=diag_chunk_size)
        penalty_min = exact_min_penalty_from_diag(diag_c, diag_p)
        penalty = penalty_min + penalty_offset
        assignments = solve_gbp_from_diag(diag_c, diag_p, n_vertices, penalty)

        if len(assignments) >= min_degenerate and W.tobytes() not in earlier_weights:
            return edges, W, penalty_min, penalty, assignments
    return None
