"""Graph Bipartitioning CQA with XY driver"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from cqa_xy_gbp_taskio import build_params, params_hash, run_dir
from qutip import Qobj, sesolve

from src.gbp import (
    build_gbp_feasible_driver_ground_state_qutip,
    build_gbp_xy_driver_qutip,
    compute_cut_value,
    gbp_hamiltonian_scale,
    search_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph Bipartitioning CQA with XY driver")
    parser.add_argument("--n-vertices", type=int, default=12, help="Number of vertices (even)")
    parser.add_argument("--edge-prob", type=float, default=0.8, help="Edge probability")
    parser.add_argument(
        "--weight-min",
        type=float,
        default=None,
        help="Lower bound of edge weights (unweighted graph if omitted)",
    )
    parser.add_argument("--weight-max", type=float, default=None, help="Upper bound of edge weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for graph generation")
    parser.add_argument(
        "--topology",
        choices=["complete", "chain", "ring"],
        default="complete",
        help="Topology of the XY driver",
    )
    parser.add_argument("--time", type=float, default=100.0, help="Annealing time T")
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of time-evolution steps (number of output time points; barely affects sesolve's integration accuracy)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance of sesolve (qutip default is 1e-6). Not saved to input.npz since it is not a computation parameter",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance of sesolve (qutip default is 1e-8). Not saved to input.npz since it is not a computation parameter",
    )
    parser.add_argument("--penalty-offset", type=float, default=1e-3, help="penalty = A* + offset")
    parser.add_argument(
        "--min-degenerate",
        type=int,
        default=4,
        help="Search by incrementing the seed from --seed until the degeneracy of the optimal solutions is at least this value",
    )
    parser.add_argument(
        "--max-seed-tries",
        type=int,
        default=1000,
        help="Maximum number of tries until a seed satisfying --min-degenerate is found",
    )
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite even if output already exists")

    args = parser.parse_args()
    if (args.weight_min is None) != (args.weight_max is None):
        parser.error("Specify both --weight-min and --weight-max, or omit both")
    if args.n_vertices % 2 != 0:
        parser.error("--n-vertices must be even for a balanced bipartition")
    return args


def main() -> None:
    args = parse_args()
    weight_range = (args.weight_min, args.weight_max) if args.weight_min is not None else None

    seed, edges, W, H_cost, H_penalty, penalty_min, penalty, assignments = search_problem(
        args.n_vertices,
        args.edge_prob,
        weight_range,
        args.seed,
        args.penalty_offset,
        args.min_degenerate,
        args.max_seed_tries,
    )

    params = build_params(W=W, n_vertices=args.n_vertices, topology=args.topology, T=args.time)
    out_dir = run_dir(params)
    output_path = out_dir / "output.npz"
    if output_path.exists() and not args.force:
        print(f"Skip: {output_path} already exists (use --force to overwrite)")
        return

    # CQA does not use the penalty in its dynamics (H_problem = H_cost only), so
    # the scale is also computed from the H_cost terms only (penalty=0.0).
    scale = gbp_hamiltonian_scale(W, penalty=0.0)
    _save_input(
        out_dir,
        params,
        args.steps,
        penalty,
        seed,
        args.edge_prob,
        scale,
    )
    print(f"Run directory: {out_dir}")
    _print_problem(edges, W, H_cost, H_penalty, penalty_min, assignments)
    print(f"Normalization scale  : {scale:.6f}")

    H_driver = build_gbp_xy_driver_qutip(args.n_vertices, args.topology)
    psi0 = build_gbp_feasible_driver_ground_state_qutip(args.n_vertices, args.topology)
    # The time evolution uses H_cost divided by scale. Normalizing so that the absolute
    # values of all Z_iZ_j coefficients fall within [0, 1] makes the meaning of T less
    # dependent on the graph's weight scale (classification and the exact-solution search
    # use the original, unnormalized H_cost / H_penalty).
    H_cost_dynamics = H_cost / scale
    start_time = time.time()
    probs = _run_annealing(H_cost_dynamics, H_driver, psi0, args.time, args.steps, args.rtol, args.atol)
    elapsed_time = time.time() - start_time

    feasible_mask, optimal_mask = _classify_states(H_cost, H_penalty)
    feasible_probability = probs[feasible_mask].sum()
    print(f"\nFeasible state probability: {feasible_probability:.4f}")
    print(f"Optimal state probability: {probs[optimal_mask].sum():.4f}")

    partition_probs = _aggregate_partitions(probs, args.n_vertices)
    opt_partitions = sorted({_canonical(a) for a in assignments})
    total_optimal_prob = sum(partition_probs.get(p, 0.0) for p in opt_partitions)
    print(f"Total optimal probability: {total_optimal_prob:.4f}")

    _save_output(
        out_dir,
        probs,
        partition_probs,
        opt_partitions,
        total_optimal_prob,
        feasible_probability,
        elapsed_time,
    )


def _print_problem(edges, W, H_cost, H_penalty, penalty_min, assignments) -> None:
    print(f"Edges: {edges}")
    print(f"Weight matrix:\n{W}")
    print(f"\nExact penalty coeff A*: {penalty_min:.6f}")
    print(f"H_cost shape         : {H_cost.shape}")
    print(f"H_penalty shape      : {H_penalty.shape}")
    print(f"Number of optimal solutions (degeneracy): {len(assignments)}")
    print()
    for i, a in enumerate(assignments):
        cut = compute_cut_value(a, W)
        print(f"  [{i:2d}] Partition: {a}  Cut value: {cut:.4f}")


def _run_annealing(
    H_cost: Qobj, H_driver: Qobj, psi0: Qobj, T: float, n_steps: int, rtol: float, atol: float
) -> np.ndarray:
    tlist = np.linspace(0, T, n_steps + 1)
    H_td = [
        [H_driver, lambda t: 1.0 - t / T],
        [H_cost, lambda t: t / T],
    ]
    # Tight rtol/atol can make sesolve's internal integration steps exceed the default
    # limit (2500), so the limit is raised accordingly (the integration accuracy itself is
    # governed by rtol/atol; a larger nsteps does not by itself change the result).
    # Only the final state is used, so intermediate states are not stored (keeping state
    # vectors for every time point would strain memory for large N as tlist grows).
    options = {
        "rtol": rtol,
        "atol": atol,
        "nsteps": 100_000,
        "store_states": False,
        "store_final_state": True,
    }
    result = sesolve(H_td, psi0, tlist, options=options)
    return np.abs(result.final_state.full().flatten()) ** 2


def _classify_states(H_cost: Qobj, H_penalty: Qobj) -> tuple[np.ndarray, np.ndarray]:
    diag_c = np.real(H_cost.diag())
    diag_p = np.real(H_penalty.diag())
    feasible_mask = np.abs(diag_p) < 1e-10
    optimal_mask = feasible_mask & (np.abs(diag_c - diag_c[feasible_mask].min()) < 1e-10)
    return feasible_mask, optimal_mask


def _aggregate_partitions(probs: np.ndarray, n_vertices: int) -> dict[tuple[int, ...], float]:
    """Aggregate all basis states by partition, normalized for the group 0/1 label-swap symmetry."""
    partition_probs: dict[tuple[int, ...], float] = defaultdict(float)
    for idx, p in enumerate(probs):
        if p < 1e-6:
            continue
        partition_probs[_canonical(_decode(idx, n_vertices))] += p
    return partition_probs


def _save_input(
    out_dir: Path,
    params: dict,
    n_steps: int,
    penalty: float,
    seed: int,
    edge_prob: float,
    scale: float,
) -> None:
    """CQA does not use the penalty in its dynamics; it is only used to search for and
    classify the degenerate exact solutions (an implementation detail of search_problem),
    so penalty_offset is not saved.
    The penalty itself does not contribute to the dynamics either, so it is not part of
    the run_dir hash, but it is saved to input.npz for the record. Likewise, scale is a
    recorded value used to normalize the dynamics (H_cost) and is not hashed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "input.npz",
        params_hash=params_hash(params),
        n_steps=np.int64(n_steps),
        penalty=np.float64(penalty),
        seed=np.int64(seed),
        edge_prob=np.float64(edge_prob),
        scale=np.float64(scale),
        **{key: np.asarray(value) for key, value in params.items()},
    )


def _save_output(
    run_dir: Path,
    probs: np.ndarray,
    partition_probs: dict[tuple[int, ...], float],
    opt_partitions: list[tuple[int, ...]],
    total_optimal_prob: float,
    feasible_probability: float,
    elapsed_time: float,
) -> None:
    """Save the probabilities probs of all computational basis states, plus the
    (state, probability) pairs aggregated per partition, and the list of theoretically
    optimal partitions with their probabilities.

    optimal_partitions is the analytically obtained list of exact solutions (from
    assignments); it does not depend on measured probabilities, so its length always
    equals the degeneracy. The corresponding optimal_partition_probs are looked up in
    partition_probs by state vector (0 if not observed); they are matched by value, not
    position, so they do not depend on ordering.
    """
    partitions = sorted(partition_probs)
    np.savez(
        run_dir / "output.npz",
        probs=probs,
        partitions=np.array([list(p) for p in partitions], dtype=np.int64),
        partition_probs=np.array([partition_probs[p] for p in partitions]),
        optimal_partitions=np.array([list(p) for p in opt_partitions], dtype=np.int64),
        optimal_partition_probs=np.array([partition_probs.get(p, 0.0) for p in opt_partitions]),
        optimal_probability=np.float64(total_optimal_prob),
        feasible_probability=np.float64(feasible_probability),
        elapsed_time=np.float64(elapsed_time),
    )


def _decode(idx: int, n_vertices: int) -> list[int]:
    return [(idx >> (n_vertices - 1 - i)) & 1 for i in range(n_vertices)]


def _canonical(assignment: list[int]) -> tuple[int, ...]:
    """Normalize the group 0/1 label-swap symmetry so that the first vertex is always 0."""
    if assignment[0] == 1:
        return tuple(1 - bit for bit in assignment)
    return tuple(assignment)


if __name__ == "__main__":
    main()
