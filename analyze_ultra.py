#!/usr/bin/env python3
"""
Analyze UltraNest output produced by main.py
- Loads ultra_results/ultranest_result.pkl (or a path you pass)
- Prints evidence, information, call stats
- Computes weighted medians & 68/95% credible intervals
- Saves CSV summary
- Makes quick corner plot and pairwise scatter density (PNG)
- Optional cluster check for multimodality (DBSCAN if scikit-learn installed)

Usage:
  python analyze_ultra.py --result ultra_results/ultranest_result.pkl \
                          --names m1 m2 p0 e0 \
                          --out out_ultra --cluster
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pickle
import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import DBSCAN  # optional
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def load_ultra(path: Path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    # Expected keys: samples, weights, logz, logzerr, information, ncall
    samples = np.asarray(res["samples"], float)
    weights = np.asarray(res["weights"], float)
    weights = weights / np.sum(weights)
    info = {
        "logz": float(res.get("logz", np.nan)),
        "logzerr": float(res.get("logzerr", np.nan)),
        "information": float(res.get("information", np.nan)),
        "ncall": int(res.get("ncall", -1)),
    }
    return samples, weights, info


def wquantile(x: np.ndarray, w: np.ndarray, qs: List[float]):
    i = np.argsort(x)
    x = x[i]; w = w[i]
    c = np.cumsum(w)
    return [np.interp(q, c, x) for q in qs]


def weighted_corr(samples: np.ndarray, w: np.ndarray):
    mu = np.sum(w[:, None] * samples, axis=0)
    X = samples - mu
    C = (w[:, None] * X).T @ X
    sd = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    corr = C / (sd[:, None] * sd[None, :])
    return corr, C, mu


def save_summary_csv(path: Path, names: List[str], samples: np.ndarray, w: np.ndarray):
    import csv
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["param", "p16", "p50", "p84", "p2.5", "p97.5", "mean", "std"])
        mu = np.sum(w[:, None] * samples, axis=0)
        var = np.sum(w[:, None] * (samples - mu)**2, axis=0)
        std = np.sqrt(var)
        for j, name in enumerate(names):
            p16, p50, p84 = wquantile(samples[:, j], w, [0.16, 0.5, 0.84])
            p025, p975 = wquantile(samples[:, j], w, [0.025, 0.975])
            cw.writerow([name, p16, p50, p84, p025, p975, mu[j], std[j]])


def corner_plot(path: Path, names: List[str], samples: np.ndarray, w: np.ndarray, bins: int = 50):
    D = samples.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(2.6*D, 2.2*D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                hist, edges = np.histogram(samples[:, i], bins=bins, weights=w, density=True)
                centers = 0.5 * (edges[1:] + edges[:-1])
                ax.plot(centers, hist)
                ax.set_yticks([])
            elif i > j:
                # downsample for speed if huge
                nmax = 40000
                if len(w) > nmax:
                    idx = np.random.choice(len(w), nmax, replace=False, p=w/w.sum())
                    x, y = samples[idx, j], samples[idx, i]
                    ax.scatter(x, y, s=1, alpha=0.25)
                else:
                    ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.25)
            else:
                ax.axis("off")
            if i == D-1:
                ax.set_xlabel(names[j])
            if j == 0 and i>0:
                ax.set_ylabel(names[i])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cluster_check(names: List[str], samples: np.ndarray, w: np.ndarray, outdir: Path, eps: float = 0.5, min_samples: int = 50):
    if not HAVE_SK:
        print("[cluster] scikit-learn not available; skipping DBSCAN.")
        return
    # Angle-safe embedding: try to detect common angle names
    def embed_angles(col, name):
        if any(k in name.lower() for k in ("phi", "theta", "psi", "qk", "qs")):
            return np.column_stack([np.sin(col), np.cos(col)])
        return col[:, None]

    # Build embedded feature matrix
    cols = []
    for j, n in enumerate(names):
        cols.append(embed_angles(samples[:, j], n))
    X = np.hstack(cols)
    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # DBSCAN clustering
    algo = DBSCAN(eps=eps, min_samples=min_samples)
    labels = algo.fit_predict(X)
    uniq, counts = np.unique(labels, return_counts=True)

    # Weighted cluster masses
    masses = {k: float(w[labels==k].sum()) for k in uniq}
    print("[cluster] labels -> counts -> weights")
    for k, c in zip(uniq, counts):
        print(f"  {k:>3}: {c:>7}  weight={masses[k]:.4f}")

    # Save a simple 2D projection plot on first two params
    fig, ax = plt.subplots(figsize=(5,4))
    cmap = plt.cm.get_cmap('tab10', len(uniq))
    for k in uniq:
        m = labels == k
        ax.scatter(samples[m, 0], samples[m, 1], s=2, alpha=0.4, label=str(k))
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
    ax.legend(title="cluster", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(outdir/"clusters_proj.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze UltraNest results")
    ap.add_argument("--result", default="ultra_results/ultranest_result.pkl",
                    help="Path to UltraNest result pickle")
    ap.add_argument("--names", nargs="+", default=["m1","m2","p0","e0"],
                    help="Ordered parameter names (matching the sampler order)")
    ap.add_argument("--out", default="ultra_analysis", help="Output directory")
    ap.add_argument("--cluster", action="store_true", help="Run a DBSCAN cluster check if sklearn is available")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    samples, w, info = load_ultra(Path(args.result))

    print(f"logZ = {info['logz']:.6g} +/- {info['logzerr']:.3g}")
    print(f"information (bits) = {info['information']}")
    print(f"ncall = {info['ncall']}")

    # Summary
    print("\nWeighted medians and 68% CI:")
    for j, name in enumerate(args.names):
        p16, p50, p84 = wquantile(samples[:, j], w, [0.16, 0.5, 0.84])
        print(f"  {name:>10s}: {p50:.6g}  [{p16:.6g}, {p84:.6g}]")

    # Correlation matrix
    corr, cov, mu = weighted_corr(samples, w)
    np.savetxt(outdir/"corr_matrix.txt", corr, fmt="%.6g")

    # CSV summary
    save_summary_csv(outdir/"summary.csv", args.names, samples, w)

    # Corner plot
    corner_plot(outdir/"corner.png", args.names, samples, w, bins=60)

    # Optional clustering
    if args.cluster:
        cluster_check(args.names, samples, w, outdir)

    print(f"\nWrote: {outdir.resolve()}/summary.csv, corr_matrix.txt, corner.png")
    if args.cluster and HAVE_SK:
        print(f"Cluster projection: {outdir.resolve()}/clusters_proj.png")

if __name__ == "__main__":
    main()

