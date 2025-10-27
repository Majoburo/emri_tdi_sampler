#!/usr/bin/env python3
"""
Analyze UltraNest results midway through a run (or completed)

This script restores an UltraNest sampler from its log directory and plots
current posterior samples, even if the run is incomplete.

Usage:
  python analyze_ultra_midway.py --log-dir ultra_results --names m1 m2 p0 e0
  python analyze_ultra_midway.py --log-dir ultra_results_mpi_12345 --names m1 m2 p0 e0 --out midway_plots
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

try:
    from ultranest import ReactiveNestedSampler
    HAS_ULTRANEST = True
except ImportError:
    HAS_ULTRANEST = False
    print("ERROR: ultranest not available. Install with: pip install ultranest")
    sys.exit(1)

try:
    from sklearn.cluster import DBSCAN
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def wquantile(x: np.ndarray, w: np.ndarray, qs: List[float]):
    """Weighted quantiles"""
    i = np.argsort(x)
    x = x[i]
    w = w[i]
    c = np.cumsum(w)
    if c[-1] == 0:
        return [np.nan] * len(qs)
    c = c / c[-1]
    return [np.interp(q, c, x) for q in qs]


def corner_plot(path: Path, names: List[str], samples: np.ndarray, w: np.ndarray, bins: int = 50):
    """Create corner plot"""
    D = samples.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(2.6 * D, 2.2 * D))

    if D == 1:
        axes = np.array([[axes]])

    for i in range(D):
        for j in range(D):
            ax = axes[i, j] if D > 1 else axes[0, 0]
            if i == j:
                # 1D histogram
                hist, edges = np.histogram(samples[:, i], bins=bins, weights=w, density=True)
                centers = 0.5 * (edges[1:] + edges[:-1])
                ax.plot(centers, hist, 'k-', linewidth=1.5)
                ax.fill_between(centers, hist, alpha=0.3)
                ax.set_yticks([])
            elif i > j:
                # 2D scatter (downsample if needed)
                nmax = 10000
                if len(w) > nmax:
                    idx = np.random.choice(len(w), nmax, replace=False, p=w / w.sum())
                    x, y = samples[idx, j], samples[idx, i]
                else:
                    x, y = samples[:, j], samples[:, i]
                ax.scatter(x, y, s=1, alpha=0.3, c='k')
            else:
                ax.axis("off")

            if i == D - 1:
                ax.set_xlabel(names[j])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])

    fig.suptitle(f"Posterior Samples (N={len(samples)})", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {path}")


def trace_plot(path: Path, names: List[str], samples: np.ndarray, logls: np.ndarray):
    """Plot parameter traces vs likelihood"""
    D = samples.shape[1]
    fig, axes = plt.subplots(D + 1, 1, figsize=(10, 2 * (D + 1)), sharex=True)

    idx = np.arange(len(samples))

    # Plot each parameter
    for i in range(D):
        axes[i].plot(idx, samples[:, i], 'k.', markersize=1, alpha=0.5)
        axes[i].set_ylabel(names[i])
        axes[i].grid(True, alpha=0.3)

    # Plot log-likelihood
    axes[D].plot(idx, logls, 'r.', markersize=1, alpha=0.5)
    axes[D].set_ylabel('log L')
    axes[D].set_xlabel('Sample index')
    axes[D].grid(True, alpha=0.3)

    fig.suptitle('Parameter Traces', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Analyze UltraNest results midway through run")
    ap.add_argument("--log-dir", default="ultra_results",
                    help="UltraNest log directory (default: ultra_results)")
    ap.add_argument("--names", nargs="+", default=["m1", "m2", "p0", "e0"],
                    help="Parameter names (default: m1 m2 p0 e0)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: {log-dir}_midway_analysis)")
    ap.add_argument("--bins", type=int, default=50,
                    help="Number of histogram bins (default: 50)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        sys.exit(1)

    # Set output directory
    if args.out is None:
        outdir = Path(str(log_dir) + "_midway_analysis")
    else:
        outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading UltraNest sampler from: {log_dir.resolve()}")

    # Try to restore sampler
    try:
        sampler = ReactiveNestedSampler.restore(log_dir=str(log_dir))
        print("  Successfully restored sampler")
    except Exception as e:
        print(f"ERROR: Failed to restore sampler: {e}")
        print("\nTip: Make sure the sampler has written at least some output.")
        sys.exit(1)

    # Get results
    try:
        results = sampler.results
    except Exception as e:
        print(f"ERROR: Failed to get results: {e}")
        sys.exit(1)

    # Extract data
    samples = np.asarray(results['samples'], dtype=float)
    weights = np.asarray(results['weights'], dtype=float)
    logz = float(results.get('logz', np.nan))
    logzerr = float(results.get('logzerr', np.nan))

    # Get log-likelihoods if available
    logls = np.asarray(results.get('weighted_samples', {}).get('logl', []), dtype=float)
    if len(logls) == 0:
        logls = np.zeros(len(samples))
        has_logls = False
    else:
        has_logls = True

    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        print("WARNING: All weights are zero, using uniform weights")
        weights = np.ones(len(samples)) / len(samples)

    # Check if we have enough samples
    if len(samples) == 0:
        print("ERROR: No samples available yet. Run is too early.")
        sys.exit(1)

    print(f"\nCurrent Status:")
    print(f"  Samples: {len(samples)}")
    print(f"  Parameters: {samples.shape[1]}")
    print(f"  log(Z) = {logz:.6g} ± {logzerr:.3g}")

    # Validate parameter names
    if len(args.names) != samples.shape[1]:
        print(f"WARNING: {len(args.names)} names provided but {samples.shape[1]} parameters found")
        print(f"         Using provided names for first {min(len(args.names), samples.shape[1])} params")
        names = args.names[:samples.shape[1]]
        # Pad with generic names if needed
        while len(names) < samples.shape[1]:
            names.append(f"p{len(names)}")
    else:
        names = args.names

    # Print summary statistics
    print("\nWeighted Statistics:")
    print(f"{'Param':<12} {'Median':<12} {'Mean':<12} {'Std':<12} {'16%-ile':<12} {'84%-ile':<12}")
    print("-" * 72)

    mu = np.sum(weights[:, None] * samples, axis=0)
    var = np.sum(weights[:, None] * (samples - mu) ** 2, axis=0)
    std = np.sqrt(var)

    for i, name in enumerate(names):
        p16, p50, p84 = wquantile(samples[:, i], weights, [0.16, 0.5, 0.84])
        print(f"{name:<12} {p50:<12.6g} {mu[i]:<12.6g} {std[i]:<12.6g} {p16:<12.6g} {p84:<12.6g}")

    # Generate plots
    print("\nGenerating plots...")

    # Corner plot
    corner_plot(outdir / "corner_midway.png", names, samples, weights, bins=args.bins)

    # Trace plot (if we have log-likelihoods)
    if has_logls and len(logls) == len(samples):
        trace_plot(outdir / "traces_midway.png", names, samples, logls)

    # Evidence evolution plot (if available)
    if 'logz_sequence' in results:
        logz_seq = np.asarray(results['logz_sequence'])
        if len(logz_seq) > 1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(logz_seq, 'k-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('log(Z)')
            ax.set_title('Evidence Evolution')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "logz_evolution.png", dpi=160)
            plt.close(fig)
            print(f"  Saved: {outdir / 'logz_evolution.png'}")

    print(f"\nAll outputs saved to: {outdir.resolve()}")
    print("\nYou can re-run this script anytime to see updated results!")


if __name__ == "__main__":
    main()
