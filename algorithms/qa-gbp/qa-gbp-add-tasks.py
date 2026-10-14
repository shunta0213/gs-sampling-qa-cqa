"""Script that enqueues the qa-gbp.py parameter grid into pueue.

PARALLEL sets the number of parallel tasks in the pueue group (default 1 = serial).
When it is 2 or more, each task gets environment variables limiting BLAS/OpenMP threads
to (number of cores / PARALLEL), so that qutip's internal operations (BLAS/OpenMP)
do not compete for cores.

To split the work across multiple servers, set NUM_SHARDS to the number of servers and
SHARD to each server's index (0-based) (only seeds with seed % NUM_SHARDS == SHARD are
enqueued; if data/ is shared, the results end up in the same place).

Computation time grows with N and T, so tasks are enqueued in descending N, then descending T
(this keeps the parallel slots filled with heavy tasks while enqueuing, and avoids ending up
with only long tasks left at the end).

When SKIP_EXISTING = True, --force is not passed to each task, so if another task or server
writes the same result first after enqueuing, qa-gbp.py also skips it at run time.
"""

import functools
import itertools
import os
import subprocess
from pathlib import Path

import numpy as np
from qa_gbp_taskio import run_dir_for

from src.gbp import search_problem


def log_grid(lo: float, hi: float) -> list[float]:
    """Return m × 10^k (m ∈ LOG_GRID_MANTISSAS) with lo <= value <= hi, in ascending order."""
    values = {round(m * 10.0**k, 10) for k in range(-6, 7) for m in LOG_GRID_MANTISSAS}
    return sorted(v for v in values if lo <= v <= hi)


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR / "qa-gbp.py"

DEFAULT_MIN_DEGENERATE = 4
DEFAULT_MAX_SEED_TRIES = 1000

# --- Parameter grid (values to sweep) ---
# In QA the penalty enters the dynamics itself (H_cost + penalty * H_penalty), so unlike
# cqa-xy-gbp-add-tasks.py, PENALTY_OFFSET is also a list of values to sweep.
N_VERTICES = [6, 8, 10, 12]
SEED = [i for i in range(100)]

LOG_GRID_MANTISSAS = [1, 1.5, 2, 3, 5, 7]

T_MIN = 1e0
T_MAX = 1e5
TIME = log_grid(T_MIN, T_MAX)

PENALTY_OFFSET = log_grid(0.01, 10)

STEPS = int(1e5)
RTOL = 1e-10
ATOL = 1e-12

EDGE_PROB = [0.8]
MIN_DEGENERATE = 4
WEIGHT_MIN = None
WEIGHT_MAX = None
MAX_SEED_TRIES = None

# --- pueue settings ---
USE_PUEUE = True  # If False, run serially in place without pueue
GROUP = "qa-gbp"  # None means the default group (only used when USE_PUEUE = True)
PARALLEL = 48  # Adjust to the number of cores on the machine

DRY_RUN = False  # If True, only print the commands without enqueuing (running) them
SKIP_EXISTING = False  # Skip existing data (if False, pass --force so results are recomputed and overwritten)

# --- Splitting work across multiple servers ---
NUM_SHARDS = 1  # Number of servers sharing the work (split by seed % NUM_SHARDS)
SHARD = 0  # This server's index (0 <= SHARD < NUM_SHARDS)


def main() -> None:
    if not 0 <= SHARD < NUM_SHARDS:
        raise ValueError(f"SHARD must satisfy 0 <= SHARD < NUM_SHARDS (SHARD={SHARD}, NUM_SHARDS={NUM_SHARDS})")
    if USE_PUEUE and not DRY_RUN:
        _ensure_group(GROUP, PARALLEL)
    n_threads = thread_budget(PARALLEL)

    weight_range = (WEIGHT_MIN, WEIGHT_MAX) if WEIGHT_MIN is not None else None
    min_degenerate = MIN_DEGENERATE if MIN_DEGENERATE is not None else DEFAULT_MIN_DEGENERATE
    max_seed_tries = MAX_SEED_TRIES if MAX_SEED_TRIES is not None else DEFAULT_MAX_SEED_TRIES

    seeds = [seed for seed in SEED if seed % NUM_SHARDS == SHARD]
    # Descending N, then descending T (enqueue heavy tasks first)
    combos = itertools.product(
        sorted(N_VERTICES, reverse=False), sorted(TIME, reverse=False), seeds, EDGE_PROB, PENALTY_OFFSET
    )
    n_enqueued = 0
    n_skipped = 0
    n_no_problem = 0
    for n_vertices, time, seed, edge_prob, penalty_offset in combos:
        if (
            _find_problem(n_vertices, seed, edge_prob, weight_range, penalty_offset, min_degenerate, max_seed_tries)
            is None
        ):
            # qa-gbp.py would fail with the same RuntimeError, so do not enqueue (some seeds for N=6)
            n_no_problem += 1
            continue
        if SKIP_EXISTING and _already_computed(
            n_vertices,
            seed,
            time,
            edge_prob,
            weight_range,
            penalty_offset,
            min_degenerate,
            max_seed_tries,
        ):
            print(
                f"skip (exists): n_vertices={n_vertices} seed={seed} time={time} "
                f"edge_prob={edge_prob} penalty_offset={penalty_offset}"
            )
            n_skipped += 1
            continue

        command = thread_env_prefix(n_threads) + _build_command(n_vertices, seed, time, edge_prob, penalty_offset)
        if DRY_RUN:
            print(" ".join(command))
        elif USE_PUEUE:
            _enqueue(command)
        else:
            _run_serially(command)
        n_enqueued += 1

    action = "Printed" if DRY_RUN else ("Enqueued" if USE_PUEUE else "Ran")
    skip_note = f" ({n_skipped} skipped as already computed)" if SKIP_EXISTING else ""
    no_problem_note = f" ({n_no_problem} excluded because no problem was found for the seed)" if n_no_problem else ""
    print(f"\n{action} {n_enqueued} job(s){skip_note}{no_problem_note}")


@functools.cache
def _find_problem(
    n_vertices: int,
    seed: int,
    edge_prob: float,
    weight_range: tuple[float, float] | None,
    penalty_offset: float,
    min_degenerate: int,
    max_seed_tries: int,
):
    """Reproduce the (W, penalty) that qa-gbp.py actually uses (cached since it does not depend on T).
    Returns None for seeds where no graph satisfying the degeneracy condition is found.
    """
    try:
        _, _, W, _, _, _, penalty, _ = search_problem(
            n_vertices,
            edge_prob,
            weight_range,
            seed,
            penalty_offset,
            min_degenerate,
            max_seed_tries,
            build_qutip=False,
        )
    except RuntimeError:
        return None
    return W, penalty


def _already_computed(
    n_vertices: int,
    seed: int,
    time: float,
    edge_prob: float,
    weight_range: tuple[float, float] | None,
    penalty_offset: float,
    min_degenerate: int,
    max_seed_tries: int,
) -> bool:
    """Compute the run_dir qa-gbp.py would use (reproducing its problem search) and check whether output.npz exists."""
    W, penalty = _find_problem(
        n_vertices, seed, edge_prob, weight_range, penalty_offset, min_degenerate, max_seed_tries
    )
    run_dir = run_dir_for(W, n_vertices, time, penalty)
    return (run_dir / "output.npz").exists()


def _build_command(n_vertices: int, seed: int, time: float, edge_prob: float, penalty_offset: float) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        str(TARGET_SCRIPT),
        "--n-vertices",
        str(n_vertices),
        "--seed",
        str(seed),
        "--time",
        str(time),
        "--edge-prob",
        str(edge_prob),
        "--steps",
        str(STEPS),
        "--rtol",
        str(RTOL),
        "--atol",
        str(ATOL),
        "--penalty-offset",
        str(penalty_offset),
    ]
    if not SKIP_EXISTING:
        command.append("--force")
    if WEIGHT_MIN is not None:
        command += ["--weight-min", str(WEIGHT_MIN), "--weight-max", str(WEIGHT_MAX)]
    if MIN_DEGENERATE is not None:
        command += ["--min-degenerate", str(MIN_DEGENERATE)]
    if MAX_SEED_TRIES is not None:
        command += ["--max-seed-tries", str(MAX_SEED_TRIES)]
    return command


def available_cpus() -> int:
    """Return the number of available CPUs. os.cpu_count() may ignore the container's cgroup
    CPU quota and return the host's logical core count, so prefer `nproc` (which respects the
    quota) and fall back to os.cpu_count() only on failure.
    """
    try:
        out = subprocess.run(["nproc"], capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return os.cpu_count() or 1


def thread_budget(parallel: int) -> int:
    """Compute the per-task thread limit from the pueue parallelism (number of cores / parallel)."""
    return max(1, available_cpus() // parallel)


def thread_env_prefix(n_threads: int) -> list[str]:
    """Explicitly limit per-process thread counts via environment variables so that
    BLAS/OpenMP libraries do not compete for cores during parallel execution.
    """
    return [
        "env",
        f"OMP_NUM_THREADS={n_threads}",
        f"OPENBLAS_NUM_THREADS={n_threads}",
        f"MKL_NUM_THREADS={n_threads}",
        f"NUMEXPR_NUM_THREADS={n_threads}",
        f"VECLIB_MAXIMUM_THREADS={n_threads}",
    ]


def _ensure_group(group: str | None, parallel: int) -> None:
    if group is not None:
        subprocess.run(["pueue", "group", "add", group], capture_output=True)
    parallel_cmd = ["pueue", "parallel", str(parallel)]
    if group is not None:
        parallel_cmd += ["--group", group]
    subprocess.run(parallel_cmd, check=True)


def _enqueue(command: list[str]) -> None:
    pueue_command = ["pueue", "add", "--working-directory", str(SCRIPT_DIR)]
    if GROUP is not None:
        pueue_command += ["--group", GROUP]
    pueue_command += ["--", *command]
    subprocess.run(pueue_command, check=True)


def _run_serially(command: list[str]) -> None:
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
