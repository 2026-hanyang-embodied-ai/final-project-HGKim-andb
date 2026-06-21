#!/usr/bin/env python3
"""Stage 2 — Change detectors over the per-frame OOD signal.

Three detectors produce a per-frame "detection statistic". An alarm fires when
the statistic crosses a threshold; the threshold is what eval sweeps for the ROC,
so it is NOT fixed here.

  - threshold  (baseline): statistic = standardized signal z_t            (memoryless)
  - cusum                : S_t = max(0, S_{t-1} + (z_t - k))              (accumulates)
  - ewma                 : g_t = lam*z_t + (1-lam)*g_{t-1}                (accumulates)

All detectors RESET at scene boundaries (each nuScenes scene is an independent
episode). The signal is standardized with global mean/std so k and lam are
scale-free.

Importable as a library (used by eval_detectors.py); also runnable as a CLI to
write results/curiosity/signals_stats.csv for inspection.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SIG_CSV = PROJECT / "results/curiosity/signals.csv"
OUT_CSV = PROJECT / "results/curiosity/signals_stats.csv"

# Defaults; eval may sweep these.
CUSUM_K = 0.5
EWMA_LAM = 0.3


def standardize(x: np.ndarray) -> np.ndarray:
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / (sd if sd > 0 else 1.0)


def cusum(z: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    """One-sided upper CUSUM, reset to 0 at the start of the sequence."""
    s = np.empty_like(z, dtype=float)
    acc = 0.0
    for i, v in enumerate(z):
        acc = max(0.0, acc + (v - k))
        s[i] = acc
    return s


def ewma(z: np.ndarray, lam: float = EWMA_LAM) -> np.ndarray:
    """EWMA of the signal, initialised at the sequence's first value scaled to 0
    baseline (z is already zero-mean globally)."""
    g = np.empty_like(z, dtype=float)
    prev = 0.0  # expected value of standardized signal
    for i, v in enumerate(z):
        prev = lam * v + (1.0 - lam) * prev
        g[i] = prev
    return g


def per_scene(values: np.ndarray, scenes: np.ndarray, fn) -> np.ndarray:
    """Apply a per-sequence detector fn independently within each scene,
    preserving global row order."""
    out = np.empty_like(values, dtype=float)
    for sc in OrderedDict.fromkeys(scenes.tolist()):
        idx = np.where(scenes == sc)[0]
        out[idx] = fn(values[idx])
    return out


def compute_statistics(scenes: np.ndarray, signal: np.ndarray,
                       k: float = CUSUM_K, lam: float = EWMA_LAM) -> dict:
    """Return {detector_name: per-frame statistic array}. Signal is standardized
    globally first so detectors are scale-free."""
    z = standardize(signal)
    return {
        "threshold": z,
        "cusum": per_scene(z, scenes, lambda a: cusum(a, k)),
        "ewma": per_scene(z, scenes, lambda a: ewma(a, lam)),
    }


def load_signals(path=SIG_CSV, col: str = "signal_std") -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = list(csv.DictReader(open(path)))
    scenes = np.array([r["scene"] for r in rows])
    signal = np.array([float(r[col]) for r in rows])
    return scenes, signal, rows


def main() -> None:
    scenes, signal, rows = load_signals()
    stats = compute_statistics(scenes, signal)
    fields = list(rows[0].keys()) + [f"stat_{n}" for n in stats]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            for n, arr in stats.items():
                r[f"stat_{n}"] = float(arr[i])
            w.writerow(r)
    print(f"wrote {OUT_CSV}  (k={CUSUM_K}, lam={EWMA_LAM})")
    for n, arr in stats.items():
        print(f"  stat_{n}: range [{arr.min():.2f}, {arr.max():.2f}]")


if __name__ == "__main__":
    main()
