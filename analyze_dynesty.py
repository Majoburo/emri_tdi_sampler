#!/usr/bin/env python3
"""
Simple dynesty analyzer

Usage examples:
  # preferred (fast, no pickle issues)
  python analyze_dynesty.py --npz results/posterior.npz

  # or from a dynesty Results pickle
  python analyze_dynesty.py --pickle results/run_results.pkl --names m1 m2 p0 e0

(Checkpoint restore is supported but fragile; prefer NPZ or pickle.)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import corner

# ---------- utils ----------

def wquantile(x: np.ndarray, w: np.ndarray, qs):
    i = np.argsort(x)
    x = x[i]; w = w[i]
    c = np.cumsum(w)
    return [np.interp(q, c, x) for q in qs]

# ---------- loaders ----------

def load_from_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    samples = z["samples"]
    w = z["weights"]
    names = list(z["names"]) if "names" in z else None
    meta = {
        "logz": float(z["logz"]) if "logz" in z else np.nan,
        "logzerr": float(z["logzerr"]) if "logzerr" in z else np.nan,
        "information": float(z["information"]) if "information" in z else np.nan,
        "ncall": int(z["ncall"]) if "ncall" in z else -1,
    }
    w = w / w.sum()
    return samples, w, names, meta


def load_from_pickle(path: Path, names_arg=None):
    with open(path, "rb") as f:
        res = pickle.load(f)
    logwt = np.asarray(res.logwt, float)
    logz = np.asarray(res.logz, float)
    samples = np.asarray(res.samples, float)
    w = np.exp(logwt - logz[-1]); w /= np.sum(w)
    breakpoint()
    # dynesty Results does not store names; require them or leave None
    names = names_arg
    meta = {
        "logz": float(logz[-1]),
        "logzerr": float(getattr(res, 'logzerr', [np.nan])[-1]) if hasattr(res,'logzerr') else np.nan,
        "information": float(getattr(res, 'information', np.nan)),
        "ncall": int(getattr(res, 'ncall', -1)),
    }
    return samples, w, names, meta


def load_from_checkpoint(path: Path, names_arg=None):
    """Minimal checkpoint restore (works if main_dynesty.py is importable)."""
    try:
        import sys, importlib, dynesty  # type: ignore
        # make project root importable and alias main_dynesty as __main__
        breakpoint()
        root = Path.cwd()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            m = importlib.import_module('main_dynesty')
            sys.modules['__main__'] = m
        except Exception:
            pass
        sampler = dynesty.DynamicNestedSampler.restore(str(path))
        res = sampler.results
        return load_from_pickle(None, names_arg)  # reuse code path
    except Exception as e:
        raise SystemExit(f"Failed to restore checkpoint: {e}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Simple dynesty analyzer")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument('--npz', default='results/posterior.npz', help='Path to posterior.npz (preferred)')
    g.add_argument('--pickle', help='Path to dynesty Results pickle (run_results.pkl)')
    g.add_argument('--checkpoint', help='Path to checkpoint file (run.dynesty)')
    ap.add_argument('--names', nargs='+', help='Parameter names (required if not in NPZ)')
    ap.add_argument('--out', default='dynesty_simple_analysis', help='Output directory')
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    if args.pickle:
        if not args.names:
            print('[info] No --names provided; results will use p0..p{D-1}')
        samples, w, names, meta = load_from_pickle(Path(args.pickle), names_arg=args.names)
    elif args.checkpoint:
        samples, w, names, meta = load_from_checkpoint(Path(args.checkpoint), names_arg=args.names)
    else:
        samples, w, names, meta = load_from_npz(Path(args.npz))
        if names is None:
            names = args.names

    if names is None:
        names = [f"p{i}" for i in range(samples.shape[1])]

    print(f"logZ = {meta['logz']:.6g} +/- {meta['logzerr']:.3g}")
    print(f"information (bits) = {meta['information']}")
    print(f"ncall = {meta['ncall']}")

    # Print simple table
    print("\nWeighted medians [16,50,84]%:")
    for j, n in enumerate(names):
        p16, p50, p84 = wquantile(samples[:, j], w, [0.16, 0.5, 0.84])
        print(f"  {n:>10s}: {p50:.6g}  [{p16:.6g}, {p84:.6g}]")

    fig = corner.corner(samples, labels=names, weights=w, bins=60,
                        plot_datapoints=False, smooth=1.0)
    fig.savefig(outdir / 'corner.png', dpi=160)
    plt.close(fig)
    print(f"\nSaved: {outdir.resolve()}/corner.png")

if __name__ == '__main__':
    main()
