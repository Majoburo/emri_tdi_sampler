#!/usr/bin/env python3
"""
Analyze Eryn MCMC results

Usage:
  python analyze_eryn.py --samples eryn_results/eryn_samples.npz
  python analyze_eryn.py --chains eryn_results/eryn_chains.pkl --burn 1000
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: corner not available. Install with: pip install corner")


def plot_chains(chain, log_prob, param_names, output_path):
    """
    Plot MCMC chains to check convergence

    Args:
        chain: (nsteps, nwalkers, ndim) array
        log_prob: (nsteps, nwalkers) array
        param_names: list of parameter names
        output_path: where to save plot
    """
    nsteps, nwalkers, ndim = chain.shape

    fig, axes = plt.subplots(ndim + 1, 1, figsize=(10, 2 * (ndim + 1)), sharex=True)

    # Plot parameter chains
    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(chain[:, j, i], 'k-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel(param_names[i])
        ax.grid(True, alpha=0.3)

    # Plot log probability
    ax = axes[ndim]
    for j in range(nwalkers):
        ax.plot(log_prob[:, j], 'r-', alpha=0.3, linewidth=0.5)
    ax.set_ylabel('log L')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    fig.suptitle('MCMC Chains', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_corner_plot(samples, param_names, output_path, truths=None):
    """Create corner plot"""
    if not HAS_CORNER:
        print("  Skipping corner plot (corner not installed)")
        return

    fig = corner.corner(
        samples,
        labels=param_names,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.4g',
        smooth=1.0,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def compute_autocorr_time(chain):
    """
    Estimate autocorrelation time (simple method)

    Args:
        chain: (nsteps, nwalkers, ndim) array

    Returns:
        autocorr_time per parameter
    """
    nsteps, nwalkers, ndim = chain.shape

    # Flatten walkers
    flat_chain = chain.reshape(nsteps, -1)

    autocorr_times = []
    for i in range(ndim):
        # Simple autocorrelation estimate
        x = flat_chain[:, i]
        x_mean = np.mean(x)
        var = np.var(x)

        if var == 0:
            autocorr_times.append(np.nan)
            continue

        # Compute autocorrelation
        autocorr = np.correlate(x - x_mean, x - x_mean, mode='full') / (len(x) * var)
        autocorr = autocorr[len(autocorr) // 2:]

        # Find where autocorr drops below 1/e
        threshold = 1.0 / np.e
        below_threshold = np.where(autocorr < threshold)[0]

        if len(below_threshold) > 0:
            tau = below_threshold[0]
        else:
            tau = len(autocorr)

        autocorr_times.append(tau)

    return np.array(autocorr_times)


def main():
    ap = argparse.ArgumentParser(description="Analyze Eryn MCMC results")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--samples", help="Path to eryn_samples.npz (flattened samples)")
    g.add_argument("--chains", help="Path to eryn_chains.pkl (full chains)")
    ap.add_argument("--burn", type=int, default=None,
                    help="Burn-in steps to discard (only for --chains)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: same as input)")
    args = ap.parse_args()

    # Load data
    if args.samples:
        # Load flattened samples
        data = np.load(args.samples, allow_pickle=True)
        samples = data['samples']
        log_prob = data['log_prob']
        param_names = list(data['param_names'])

        if args.out is None:
            outdir = Path(args.samples).parent / "eryn_analysis"
        else:
            outdir = Path(args.out)

        print(f"Loaded flattened samples: {samples.shape}")
        chain = None
        truths = None

    else:
        # Load full chains
        with open(args.chains, 'rb') as f:
            results = pickle.load(f)

        chain = results['chain']
        log_prob_full = results['log_prob']
        param_names = results['param_names']
        truths = [results['theta0'][i] for i in results['sample_idx']]

        nsteps, nwalkers, ndim = chain.shape
        print(f"Loaded chains: {nsteps} steps × {nwalkers} walkers × {ndim} dims")

        # Apply burn-in
        burn = args.burn if args.burn is not None else results.get('burn', 0)
        if burn > 0:
            print(f"Discarding first {burn} steps as burn-in")
            chain_burned = chain[burn:, :, :]
            log_prob_burned = log_prob_full[burn:, :]
        else:
            chain_burned = chain
            log_prob_burned = log_prob_full

        # Flatten
        samples = chain_burned.reshape(-1, ndim)
        log_prob = log_prob_burned.reshape(-1)

        if args.out is None:
            outdir = Path(args.chains).parent / "eryn_analysis"
        else:
            outdir = Path(args.out)

    outdir.mkdir(parents=True, exist_ok=True)

    nsamples, ndim = samples.shape
    print(f"Analyzing {nsamples} samples across {ndim} parameters")

    # Summary statistics
    print("\nPosterior Summary:")
    print(f"{'Param':<12} {'Mean':<12} {'Std':<12} {'16%':<12} {'50%':<12} {'84%':<12}")
    print("-" * 72)

    for i, name in enumerate(param_names):
        mean_val = np.mean(samples[:, i])
        std_val = np.std(samples[:, i])
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name:<12} {mean_val:<12.6g} {std_val:<12.6g} {p16:<12.6g} {p50:<12.6g} {p84:<12.6g}")

    # Log-likelihood statistics
    print(f"\nLog-Likelihood:")
    print(f"  Max: {np.max(log_prob):.6g}")
    print(f"  Mean: {np.mean(log_prob):.6g}")
    print(f"  Median: {np.median(log_prob):.6g}")

    # Best-fit point
    best_idx = np.argmax(log_prob)
    print(f"\nBest-fit point (log L = {log_prob[best_idx]:.6g}):")
    for i, name in enumerate(param_names):
        print(f"  {name:<12} = {samples[best_idx, i]:.6g}")

    # Plots
    print("\nGenerating plots...")

    # Chain plot (if we have full chains)
    if chain is not None:
        plot_chains(chain, log_prob_full, param_names, outdir / "chains.png")

    # Corner plot
    plot_corner_plot(samples, param_names, outdir / "corner.png", truths=truths)

    # Autocorrelation (if we have chains)
    if chain is not None:
        print("\nEstimating autocorrelation times...")
        autocorr = compute_autocorr_time(chain)
        print("Autocorrelation time (steps):")
        for i, name in enumerate(param_names):
            if np.isfinite(autocorr[i]):
                print(f"  {name:<12}: {autocorr[i]:.1f}")
            else:
                print(f"  {name:<12}: N/A")

        effective_samples = nsamples / np.max(autocorr[np.isfinite(autocorr)])
        print(f"\nEffective number of samples: ~{effective_samples:.0f}")

    print(f"\nAll outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
