"""Shared dispatch helper for experiment scripts.

`dispatch(jobs, backend, n_jobs)` runs a list of `(callable, args, kwargs)`
units in serial, joblib (multi-process), or emits Slurm-array shell lines.

`run_with_cache(cache_dir, key, fn)` checks for a cached JSON result keyed by
`key`, computes via `fn` if missing, and writes the result back. This is the
restart-on-failure substrate every experiment relies on.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def dispatch(
    jobs: Sequence[tuple[Callable, tuple, dict]],
    *,
    backend: str = "serial",
    n_jobs: int = -1,
) -> list[Any]:
    """Run `jobs` and return their results in order.

    backend:
      "serial" : run sequentially (deterministic, for tests)
      "joblib" : multi-process via joblib.Parallel (local cluster fan-out)
      "slurm"  : do NOT run; print one shell line per job (for sbatch arrays)
    """
    if backend == "serial":
        return [fn(*args, **kwargs) for (fn, args, kwargs) in jobs]
    if backend == "joblib":
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fn)(*args, **kwargs) for (fn, args, kwargs) in jobs
        )
    if backend == "slurm":
        for i, (fn, args, kwargs) in enumerate(jobs):
            print(f"# job {i}: {fn.__module__}.{fn.__name__}  args={args}  kwargs={kwargs}")
        return []
    raise ValueError(f"unknown backend {backend!r}")


def run_with_cache(
    cache_dir: str | Path,
    key: str,
    fn: Callable[[], dict],
    *,
    force: bool = False,
) -> dict:
    """Return cached JSON result if present, else compute and persist."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    if not force and cache_file.exists():
        with cache_file.open("r") as fp:
            return json.load(fp)
    result = fn()
    # Convert numpy types to plain JSON-friendly forms
    serialisable = _to_serialisable(result)
    with cache_file.open("w") as fp:
        json.dump(serialisable, fp, indent=2)
    return serialisable


def _to_serialisable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_csv(rows: Iterable[dict], path: str | Path,
              columns: list[str] | None = None) -> None:
    """Minimal CSV writer (no pandas dependency)."""
    import csv

    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})
