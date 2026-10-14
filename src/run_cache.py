"""Helpers for the visualize notebooks that cache a DataFrame aggregated from run
directories (input.npz / output.npz) into a single .npz file.

The cache is not updated automatically. Use the `refresh` argument to control when it is updated.

- ``refresh=False``: use the cache if it exists (otherwise load all runs and build it)
- ``refresh="new"``: load only run directories not in the cache and append them
- ``refresh=True``: reload all runs and rebuild the cache
"""

import os
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

_COLUMN_PREFIX = "col:"


def _scan_run_dirs(data_dir: Path) -> list[Path]:
    return sorted(Path(e.path) for e in os.scandir(data_dir) if e.is_dir())


def _load_rows(run_dirs: list[Path], loader: Callable[[Path], dict | None]) -> tuple[list[dict], list[str]]:
    """Apply loader in parallel and return (loaded rows, names of the loaded run directories)."""
    if not run_dirs:
        return [], []
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
        rows = list(executor.map(loader, run_dirs, chunksize=64))
    loaded = [(row, run_dir.name) for row, run_dir in zip(rows, run_dirs) if row is not None]
    return [row for row, _ in loaded], [name for _, name in loaded]


def _column_to_array(values: pd.Series) -> np.ndarray:
    """Convert a DataFrame column into an array that can be saved to npz. Columns whose elements are arrays/tuples become object arrays."""
    items = values.tolist()
    if any(isinstance(v, (np.ndarray, tuple, list)) for v in items):
        arr = np.empty(len(items), dtype=object)
        arr[:] = items
        return arr
    if items and all(isinstance(v, str) for v in items):
        return np.array(items, dtype=str)
    return values.to_numpy()


def _save_cache(cache_path: Path, df: pd.DataFrame, run_dirs: list[str], data_dir: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{_COLUMN_PREFIX}{col}": _column_to_array(df[col]) for col in df.columns}
    arrays["__columns__"] = np.array(list(df.columns), dtype=str)
    arrays["__run_dirs__"] = np.array(run_dirs, dtype=str)
    arrays["__data_dir__"] = np.array(str(data_dir.resolve()))
    arrays["__created_at__"] = np.array(time.strftime("%Y-%m-%d %H:%M:%S"))
    # Replace via a temporary file so the existing cache is not corrupted if writing is interrupted
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(cache_path)


def _read_cache(cache_path: Path) -> tuple[pd.DataFrame, list[str], str]:
    with np.load(cache_path, allow_pickle=True) as data:
        columns = data["__columns__"].tolist()
        df = pd.DataFrame({col: data[f"{_COLUMN_PREFIX}{col}"] for col in columns}, columns=columns)
        run_dirs = data["__run_dirs__"].tolist()
        created_at = str(data["__created_at__"])
    return df, run_dirs, created_at


def load_runs_cached(
    data_dir: str | Path,
    loader: Callable[[Path], dict | None],
    cache_path: str | Path,
    refresh: bool | Literal["new"] = False,
) -> pd.DataFrame:
    """Return, with caching, a DataFrame built by applying loader to each run directory under data_dir.

    loader is a function that takes a run directory and returns a dict for one row (None if it cannot be loaded).
    It must be picklable for parallel execution (e.g. a function defined at the top level of a notebook).
    """
    data_dir = Path(data_dir)
    cache_path = Path(cache_path)
    all_run_dirs = _scan_run_dirs(data_dir)

    if refresh is True or not cache_path.exists():
        rows, run_dirs = _load_rows(all_run_dirs, loader)
        df = pd.DataFrame(rows)
        _save_cache(cache_path, df, run_dirs, data_dir)
        print(f"[cache] built {cache_path} ({len(run_dirs)} runs)")
        return df

    df, cached_run_dirs, created_at = _read_cache(cache_path)
    cached = set(cached_run_dirs)
    uncached_run_dirs = [d for d in all_run_dirs if d.name not in cached]

    if refresh == "new":
        rows, run_dirs = _load_rows(uncached_run_dirs, loader)
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            # Reorder by run directory name, as in a full load
            names = cached_run_dirs + run_dirs
            order = np.argsort(names, kind="stable")
            df = df.iloc[order].reset_index(drop=True)
            _save_cache(cache_path, df, [names[i] for i in order], data_dir)
        print(f"[cache] added {len(run_dirs)} runs to {cache_path} (total {len(cached_run_dirs) + len(run_dirs)} runs)")
        return df

    print(f"[cache] loaded {cache_path} ({len(cached_run_dirs)} runs, created at {created_at})")
    if uncached_run_dirs:
        print(
            f"[cache] {len(uncached_run_dirs)} run dirs in {data_dir} are not cached (new or incomplete); "
            'set refresh="new" to add them'
        )
    return df


def _file_signature(paths: list[Path]) -> list[tuple[str, int, int]]:
    return [(str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size) for p in paths]


def load_derived_cached(
    cache_path: str | Path,
    build: Callable[[], Any],
    depends_on: list[str | Path],
    refresh: bool = False,
) -> Any:
    """Cache the result of build() (any picklable object) at cache_path and return it.

    Intended for derived data that is slow to build, such as joins/aggregations computed from load_runs_cached caches.
    Rebuilt automatically if the mtime or size of any file in depends_on (e.g. the source caches) differs from when it was saved.
    ``refresh=True`` always rebuilds (e.g. after changing what build does).
    """
    cache_path = Path(cache_path)
    signature = _file_signature([Path(p) for p in depends_on])

    if not refresh and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached["signature"] == signature:
            print(f"[cache] loaded {cache_path} (created at {cached['created_at']})")
            return cached["value"]
        print(f"[cache] {cache_path} is stale (dependencies changed); rebuilding")

    value = build()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {"signature": signature, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "value": value},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)
    print(f"[cache] built {cache_path}")
    return value
