#!/usr/bin/env python3
"""
main_dynesty.py — EMRI TDI sampler (dynesty) with analysis-friendly outputs

- CLI flags: --prior {broad,zoom}, --samples ..., --results DIR, --maxiter, --dlogz, --resume
- Saves not only the dynesty Results pickle and checkpoint, but also:
  * results/posterior.npz  (samples, weights, names, logz, logzerr, info, ncall)
  * results/paramnames.txt (one name per line, in sample order)
  * results/summary.csv    (weighted medians + 68/95% CIs)
- Prints a compact table at the end for quick viewing.
- All callables are top-level (pickle-safe for checkpoint restore).
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import dynesty

# FEW + LISA Tools
from few.waveform import GenerateEMRIWaveform
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.sensitivity import AE1SensitivityMatrix
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

# ------------------------
# Defaults / injected theta0
# ------------------------
DEFAULT_TOBS = 0.1   # years
DEFAULT_DT = 100.0   # seconds

PARAM_NAMES = [
    "m1","m2","a","p0","e0","x0","dist","qS","phiS","qK","phiK","Phi_phi0","Phi_theta0","Phi_r0"
]
NAME_TO_IDX = {n: i for i, n in enumerate(PARAM_NAMES)}

THETA0 = np.array([
    0.5e6,      # m1 (Msun)
    10.0,       # m2 (Msun)
    0.9,        # a
    12.0,       # p0
    0.01,       # e0
    1.0,        # x0 = cos(incl)
    1.0,        # dist (Gpc)
    0.01,       # qS
    np.pi/3,    # phiS
    np.pi/3,    # qK
    np.pi/3,    # phiK
    np.pi/3,    # Phi_phi0
    0.0,        # Phi_theta0
    np.pi/3,    # Phi_r0
], dtype=float)

# Module globals used by loglike (top-level for pickle safety)
resp = None
freqs = None
analysis = None
sample_idx = None
sample_names = None
theta0 = None

# ------------------------
# Prior (pickle-safe class; no closures)
# ------------------------
class PriorTransform:
    """Callable that maps unit cube u -> parameter subset.
    Modes:
      - 'broad' global ranges
      - 'zoom' windows around injected theta0
    """
    _ordered_param_names = PARAM_NAMES

    def __init__(self, sample_names, theta0, mode="broad"):
        self.sample_names = list(sample_names)
        self.mode = str(mode)
        self.theta0 = np.asarray(theta0, dtype=float)
        self.name_to_idx = {n: i for i, n in enumerate(self._ordered_param_names)}
        self.t0 = {n: self.theta0[self.name_to_idx[n]] for n in self._ordered_param_names}

    @staticmethod
    def _wrap_angle(val):
        twopi = 2*np.pi
        return (val + twopi) % twopi

    @staticmethod
    def _scale(center, frac_width, u, lo=None, hi=None, positive=False):
        val = center * (1.0 - 0.5*frac_width + frac_width * u)
        if positive:
            val = np.maximum(val, np.finfo(float).tiny)
        if lo is not None: val = np.maximum(val, lo)
        if hi is not None: val = np.minimum(val, hi)
        return val

    def _broad_value(self, name, u):
        if name == "m1":        return 10 ** (5.0 + 3.0 * u)         # 1e5..1e8
        if name == "m2":        return 10 ** (1.0 + 1.0 * u)         # 10..100
        if name == "a":         return 0.999 * u                     # [0,0.999]
        if name == "p0":        return 7.5 + (30.0 - 7.5) * u        # [7.5,30]
        if name == "e0":        return 0.75 * u                      # [0,0.75]
        if name == "x0":        return -1.0 + 2.0 * u                # [-1,1]
        if name == "dist":      return 10 ** (-3.0 + 6.0 * u)        # 1e-3..1e3 Gpc
        if name == "qS":        return np.arccos(1 - 2*u)            # [0,π]
        if name == "phiS":      return 2*np.pi*u                     # [0,2π)
        if name == "qK":        return np.arccos(1 - 2*u)            # [0,π]
        if name == "phiK":      return 2*np.pi*u                     # [0,2π)
        if name == "Phi_phi0":  return 2*np.pi*u
        if name == "Phi_theta0":return 2*np.pi*u
        if name == "Phi_r0":    return 2*np.pi*u
        raise KeyError(f"Unknown parameter name: {name}")

    def _zoom_value(self, name, u):
        t0 = self.t0
        if name == "m1":        return self._scale(t0["m1"], 0.20, u, lo=1e5, hi=1e8, positive=True)
        if name == "m2":        return self._scale(t0["m2"], 0.20, u, lo=1.0, hi=1e3, positive=True)
        if name == "a":         return np.clip(t0["a"] + (u-0.5)*0.2, 0.0, 0.999)
        if name == "p0":        return self._scale(t0["p0"], 0.20, u, lo=7.5, hi=30.0, positive=True)
        if name == "e0":
            if t0["e0"] > 0:    return np.clip(t0["e0"] + (u-0.5)*0.4*t0["e0"], 0.0, 0.75)
            else:               return np.clip(0.2*u, 0.0, 0.75)
        if name == "x0":        return np.clip(t0["x0"] + (u-0.5)*0.4, -1.0, 1.0)
        if name == "dist":      return self._scale(t0["dist"], 1.0, u, lo=1e-3, hi=1e3, positive=True)
        if name == "qS":        return np.clip(t0["qS"] + (u-0.5)*(np.pi/2), 0.0, np.pi)
        if name == "phiS":      return self._wrap_angle(t0["phiS"] + (u-0.5)*np.pi)
        if name == "qK":        return np.clip(t0["qK"] + (u-0.5)*(np.pi/2), 0.0, np.pi)
        if name == "phiK":      return self._wrap_angle(t0["phiK"] + (u-0.5)*np.pi)
        if name == "Phi_phi0":  return self._wrap_angle(t0["Phi_phi0"] + (u-0.5)*np.pi)
        if name == "Phi_theta0":return self._wrap_angle(t0["Phi_theta0"] + (u-0.5)*np.pi)
        if name == "Phi_r0":    return self._wrap_angle(t0["Phi_r0"] + (u-0.5)*np.pi)
        raise KeyError(f"Unknown parameter name: {name}")

    def __call__(self, u):
        u = np.asarray(u)
        out = np.empty(len(self.sample_names), dtype=float)
        for i, name in enumerate(self.sample_names):
            if self.mode == "broad":   out[i] = self._broad_value(name, u[i])
            elif self.mode == "zoom":  out[i] = self._zoom_value(name, u[i])
            else: raise ValueError(f"Unknown prior mode: {self.mode}")
        return out

    def desc(self):
        if self.mode == "broad":
            return {
                "m1": "log-uniform [1e5,1e8] Msun",
                "m2": "log-uniform [10,100] Msun",
                "a":  "[0,0.999]",
                "p0": "[7.5,30]",
                "e0": "[0,0.75]",
                "x0": "[-1,1]",
                "dist":"log-uniform [1e-3,1e3] Gpc",
                "qS":  "[0,π]",
                "phiS":"[0,2π)",
                "qK":  "[0,π]",
                "phiK":"[0,2π)",
                "Phi_phi0":"[0,2π)", "Phi_theta0":"[0,2π)", "Phi_r0":"[0,2π)",
            }
        t0 = self.t0
        return {
            "m1": f"±10% around {t0['m1']:.3g}",
            "m2": f"±10% around {t0['m2']:.3g}",
            "a":  f"±0.1 around {t0['a']:.3g}",
            "p0": f"±10% around {t0['p0']:.3g}",
            "e0": f"~±20%*e0 around {t0['e0']:.3g} (→[0,0.75])",
            "x0": f"±0.2 around {t0['x0']:.3g} (→[-1,1])",
            "dist": f"~0.5x–1.5x around {t0['dist']:.3g} Gpc (→[1e-3,1e3])",
            "qS": "±π/4 (clamped [0,π])",
            "phiS": "±π/2 (wrapped)",
            "qK": "±π/4 (clamped [0,π])",
            "phiK": "±π/2 (wrapped)",
            "Phi_phi0": "±π/2 (wrapped)",
            "Phi_theta0": "±π/2 (wrapped)",
            "Phi_r0": "±π/2 (wrapped)",
        }

# ------------------------
# Likelihood (top-level function; uses module globals)
# ------------------------

def loglike(theta_sub):
    global theta0, sample_idx, resp, freqs, analysis
    theta_full = theta0.copy()
    theta_full[sample_idx] = np.asarray(theta_sub, dtype=float)
    A_, E_, _ = resp(*theta_full)
    A_, E_ = A_[1:], E_[1:]
    template = DataResidualArray(np.vstack([A_, E_]), f_arr=freqs)
    return analysis.template_likelihood(template)

# ------------------------
# Setup model & data
# ------------------------

def setup(tobs: float, dt: float):
    global resp, freqs, analysis, theta0
    theta0 = THETA0.copy()

    gen_wave = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    )
    resp_local = ResponseWrapper(
        gen_wave, tobs, dt,
        index_lambda=8, index_beta=7,
        flip_hx=True,
        remove_sky_coords=False,
        is_ecliptic_latitude=False,
        remove_garbage=True,
        t0=100000., order=25,
        tdi="1st generation", tdi_chan="AET",
        orbits=EqualArmlengthOrbits(),
    )
    A, E, _ = resp_local(*theta0)
    f = np.fft.rfftfreq(2*(len(A)-1), d=dt)[1:]
    A, E = A[1:], E[1:]
    data_fd = np.vstack([np.asarray(A, complex), np.asarray(E, complex)])

    data = DataResidualArray(data_fd, f_arr=f)
    sens_mat = AE1SensitivityMatrix(f)
    analysis_local = AnalysisContainer(data, sens_mat)

    # publish to module globals
    globals().update(resp=resp_local, freqs=f, analysis=analysis_local)

# ------------------------
# Utilities: summaries / saving
# ------------------------

def wquantile(x, w, qs):
    i = np.argsort(x); x = x[i]; w = w[i]
    c = np.cumsum(w)
    return [np.interp(q, c, x) for q in qs]


def as_weighted_samples(res):
    logwt = np.asarray(res.logwt, float)
    logz = np.asarray(res.logz, float)
    samples = np.asarray(res.samples, float)
    w = np.exp(logwt - logz[-1]); w /= np.sum(w)
    return samples, w


def save_summary_csv(path: Path, names, samples, w):
    mu = np.sum(w[:, None] * samples, axis=0)
    var = np.sum(w[:, None] * (samples - mu)**2, axis=0)
    std = np.sqrt(var)
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["param", "p16", "p50", "p84", "p2.5", "p97.5", "mean", "std"])
        for j, n in enumerate(names):
            p16, p50, p84 = wquantile(samples[:, j], w, [0.16, 0.5, 0.84])
            p025, p975 = wquantile(samples[:, j], w, [0.025, 0.975])
            cw.writerow([n, p16, p50, p84, p025, p975, mu[j], std[j]])

# ------------------------
# CLI and main
# ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EMRI TDI dynesty sampler")
    p.add_argument("--prior", choices=["broad","zoom"], default="broad",
                   help="Prior mode: broad=global, zoom=around injected theta0")
    p.add_argument("--samples", nargs="+", default=["m1","m2","p0","e0"],
                   help=f"Parameters to sample (subset/order of: {', '.join(PARAM_NAMES)})")
    p.add_argument("--tobs", type=float, default=DEFAULT_TOBS, help="Observation time [yr]")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="Sample spacing [s]")
    p.add_argument("--results", default="results", help="Output directory")
    p.add_argument("--maxiter", type=int, default=20000, help="Maximum iterations")
    p.add_argument("--dlogz", type=float, default=0.3, help="Initial evidence tolerance")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")
    return p.parse_args()


def main():
    global sample_idx, sample_names
    args = parse_args()

    # Validate sample names
    for n in args.samples:
        if n not in NAME_TO_IDX:
            raise SystemExit(f"Unknown parameter in --samples: {n}")
    sample_names = list(args.samples)
    sample_idx = np.array([NAME_TO_IDX[n] for n in sample_names], dtype=int)
    print("Sampling:", sample_names, "-> indices:", sample_idx.tolist())

    # Build model
    setup(args.tobs, args.dt)

    # Prior
    prior = PriorTransform(sample_names, THETA0, mode=args.prior)
    print(f"[prior] mode={args.prior}")
    for n in sample_names:
        print(f"  {n:>10s}: {prior.desc()[n]}")

    # Sampler + checkpoint
    outdir = Path(args.results); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "run.dynesty"

    ndim = len(sample_idx)
    if ckpt_path.exists() and args.resume:
        sampler = dynesty.DynamicNestedSampler.restore(str(ckpt_path))
        sampler.run_nested(resume=True, checkpoint_file=str(ckpt_path))
    else:
        sampler = dynesty.DynamicNestedSampler(
            loglike, prior, ndim, bound="multi", sample="rwalk"
        )
        sampler.run_nested(dlogz_init=args.dlogz, maxiter=args.maxiter,
                           checkpoint_file=str(ckpt_path))

    res = sampler.results

    # Save dynesty Results pickle
    pkl_path = outdir / "run_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(res, f)

    # Lightweight analysis artifacts
    samples, w = as_weighted_samples(res)
    np.savez(outdir / "posterior.npz",
             samples=samples, weights=w, names=np.array(sample_names),
             logz=float(res.logz[-1]), logzerr=float(getattr(res, 'logzerr', [np.nan])[-1]),
             information=float(getattr(res, 'information', np.nan)),
             ncall=int(getattr(res, 'ncall', -1)))

    with open(outdir / "paramnames.txt", "w") as f:
        for n in sample_names:
            f.write(n + "\n")

    save_summary_csv(outdir / "summary.csv", sample_names, samples, w)

    # Console summary
    logz = float(res.logz[-1])
    logzerr = float(getattr(res, 'logzerr', [np.nan])[-1])
    print(f"\nlogZ = {logz:.6g} +/- {logzerr:.3g}")
    print(f"Saved: {pkl_path.resolve()}\n       {str(outdir / 'posterior.npz')}\n       {str(outdir / 'summary.csv')}\n       {str(ckpt_path)} (checkpoint)")

if __name__ == "__main__":
    main()
