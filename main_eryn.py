#!/usr/bin/env python3
"""
main_eryn.py — EMRI TDI sampler using Eryn (MCMC)

Eryn is an ensemble MCMC sampler useful for:
- Starting from a ball around known good values
- Problems where nested sampling struggles to converge
- Faster exploration when you're near the solution

Usage:
  python main_eryn.py --prior zoom --samples m1 m2 p0 e0 --nwalkers 32 --nsteps 5000
  python main_eryn.py --prior broad --samples m1 m2 a p0 e0 x0 --nwalkers 64 --nsteps 10000
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np

# Eryn MCMC
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State

# FEW + LISA Tools
from few.waveform import GenerateEMRIWaveform
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.sensitivity import AE1SensitivityMatrix
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

# ------------------------
# Configuration
# ------------------------
gen_wave = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
)

theta0 = np.array([
    10**5.770401,    # m1: central object mass (solar masses)
    10**1.422654,    # m2: secondary object mass (solar masses)
    0.843247,        # a: spin parameter
    10.954791,       # p0: initial semi-latus rectum
    0.01203,         # e0: eccentricity
    1.0,             # x0: cos(inclination)
    1.0,             # dist: luminosity distance (Gpc)
    0.01,            # qS: polar sky angle
    np.pi / 3,       # phiS: azimuthal viewing angle
    np.pi / 3,       # qK: polar spin angle
    np.pi / 3,       # phiK: azimuthal spin angle
    np.pi / 3,       # Phi_phi0: initial phase (azimuthal)
    0.0,             # Phi_theta0: initial phase (polar)
    np.pi / 3,       # Phi_r0: initial phase (radial)
], dtype=float)

param_names = ["m1", "m2", "a", "p0", "e0", "x0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
name_to_idx = {n: i for i, n in enumerate(param_names)}

# Module globals (for likelihood)
resp = None
freqs = None
analysis = None
sample_idx = None

# ------------------------
# Prior Definitions
# ------------------------
def create_prior(sample_names, theta0_vals, mode="broad"):
    """
    Create Eryn ProbDistContainer prior

    Args:
        sample_names: List of parameter names to sample
        theta0_vals: Full theta0 array
        mode: 'broad' or 'zoom'

    Returns:
        ProbDistContainer with priors for each parameter
    """
    priors = {}
    t0 = {n: theta0_vals[name_to_idx[n]] for n in param_names}

    for i, name in enumerate(sample_names):
        if mode == "broad":
            # Broad priors - use integer index as key
            if name == "m1":
                priors[i] = uniform_dist(1e5, 1e8)  # log-space handled in transform
            elif name == "m2":
                priors[i] = uniform_dist(10.0, 100.0)
            elif name == "a":
                priors[i] = uniform_dist(0.0, 0.999)
            elif name == "p0":
                priors[i] = uniform_dist(7.5, 30.0)
            elif name == "e0":
                priors[i] = uniform_dist(0.0, 0.75)
            elif name == "x0":
                priors[i] = uniform_dist(-1.0, 1.0)
            elif name == "dist":
                priors[i] = uniform_dist(1e-3, 1e3)
            elif name in ["qS", "qK"]:
                priors[i] = uniform_dist(0.0, np.pi)
            elif name in ["phiS", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]:
                priors[i] = uniform_dist(0.0, 2 * np.pi)
            else:
                raise ValueError(f"Unknown parameter: {name}")

        elif mode == "zoom":
            # Narrow priors around injection - use integer index as key
            if name == "m1":
                width = 0.2 * t0[name]
                priors[i] = uniform_dist(max(1e5, t0[name] - width), min(1e8, t0[name] + width))
            elif name == "m2":
                width = 0.2 * t0[name]
                priors[i] = uniform_dist(max(1.0, t0[name] - width), min(1e3, t0[name] + width))
            elif name == "a":
                priors[i] = uniform_dist(max(0.0, t0[name] - 0.1), min(0.999, t0[name] + 0.1))
            elif name == "p0":
                width = 0.2 * t0[name]
                priors[i] = uniform_dist(max(7.5, t0[name] - width), min(30.0, t0[name] + width))
            elif name == "e0":
                width = 0.4 * t0[name] if t0[name] > 0 else 0.1
                priors[i] = uniform_dist(max(0.0, t0[name] - width), min(0.75, t0[name] + width))
            elif name == "x0":
                priors[i] = uniform_dist(max(-1.0, t0[name] - 0.2), min(1.0, t0[name] + 0.2))
            elif name == "dist":
                width = 0.5 * t0[name]
                priors[i] = uniform_dist(max(1e-3, t0[name] - width), min(1e3, t0[name] + width))
            elif name in ["qS", "qK"]:
                priors[i] = uniform_dist(max(0.0, t0[name] - np.pi/4), min(np.pi, t0[name] + np.pi/4))
            elif name in ["phiS", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]:
                # For angles, wrap around
                center = t0[name]
                width = np.pi / 2
                priors[i] = uniform_dist(0.0, 2 * np.pi)  # Keep full range for angles
            else:
                raise ValueError(f"Unknown parameter: {name}")

    return ProbDistContainer(priors)

# ------------------------
# Likelihood
# ------------------------
def log_like_fn(params):
    """
    Log-likelihood function for Eryn

    Args:
        params: Dictionary with integer indices as keys (Eryn format)

    Returns:
        log-likelihood value
    """
    global theta0, sample_idx, resp, freqs, analysis

    # Build full parameter vector
    theta_full = theta0.copy()
    for i, idx in enumerate(sample_idx):
        theta_full[idx] = params[i]  # Use integer index

    # Generate waveform
    try:
        A_, E_, _ = resp(*theta_full)
        A_, E_ = A_[1:], E_[1:]
        template = DataResidualArray(np.vstack([A_, E_]), f_arr=freqs)
        logl = analysis.template_likelihood(template)

        # Check for valid likelihood
        if not np.isfinite(logl):
            return -np.inf
        return logl
    except Exception as e:
        # Return -inf for failed waveform generation
        return -np.inf

# ------------------------
# Initialize walkers
# ------------------------
def initialize_walkers(sample_names, prior_container, nwalkers, mode="broad", theta0_vals=None, scatter_frac=1e-7):
    """
    Initialize walker positions

    Args:
        sample_names: List of parameter names
        prior_container: Eryn ProbDistContainer
        nwalkers: Number of walkers
        mode: 'broad' (sample from prior) or 'zoom' (ball around theta0)
        theta0_vals: Injection values (used if mode='zoom')
        scatter_frac: Fractional scatter for initialization (default: 1e-7)

    Returns:
        coords: Dictionary of initial positions with integer keys (nwalkers, ndim)
    """
    ndim = len(sample_names)

    if mode == "zoom" and theta0_vals is not None:
        # Start in a tight ball around injection values: inj * (1 + scatter_frac)
        coords = {}
        t0 = {n: theta0_vals[name_to_idx[n]] for n in param_names}

        for i, name in enumerate(sample_names):
            center = t0[name]

            # Use fractional scatter: center * scatter_frac
            if center != 0:
                scatter = abs(center) * scatter_frac
            else:
                scatter = scatter_frac * 1e-3  # For zero values, use small absolute scatter

            # Generate walker positions
            coords[i] = np.random.normal(center, scatter, nwalkers)

            # Apply bounds for bounded parameters
            if name in ["a", "e0"]:
                coords[i] = np.clip(coords[i], 0.0, 0.999 if name == "a" else 0.75)
            elif name == "x0":
                coords[i] = np.clip(coords[i], -1.0, 1.0)
            elif name in ["qS", "qK"]:
                coords[i] = np.clip(coords[i], 0.0, np.pi)
            elif name in ["phiS", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]:
                coords[i] = coords[i] % (2 * np.pi)  # Wrap angles
            elif name in ["m1", "m2", "dist"]:
                coords[i] = np.maximum(coords[i], 1e-10)  # Keep positive
    else:
        # Sample from prior
        coords = prior_container.rvs(size=nwalkers)

    return coords

# ------------------------
# Setup model & data
# ------------------------
def setup(tobs: float, dt: float):
    """Setup response wrapper and data"""
    global resp, freqs, analysis

    # Use CPU backend (matching main_ultranest.py)
    force_backend = "cpu"
    print(f"Using backend: {force_backend}")

    resp_local = ResponseWrapper(
        gen_wave, tobs, dt,
        index_lambda=8, index_beta=7,
        flip_hx=True,
        remove_sky_coords=False,
        is_ecliptic_latitude=False,
        remove_garbage=True,
        t0=100000., order=25,
        tdi="1st generation", tdi_chan="AET",
        orbits=EqualArmlengthOrbits(force_backend=force_backend),
        force_backend=force_backend,
    )

    # Generate data
    A, E, _ = resp_local(*theta0)
    f = np.fft.rfftfreq(2*(len(A)-1), d=dt)[1:]
    A, E = A[1:], E[1:]
    data_fd = np.vstack([np.asarray(A, complex), np.asarray(E, complex)])

    data = DataResidualArray(data_fd, f_arr=f)
    sens_mat = AE1SensitivityMatrix(f)
    analysis_local = AnalysisContainer(data, sens_mat)

    # Update globals
    globals().update(resp=resp_local, freqs=f, analysis=analysis_local)

# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EMRI TDI Eryn (MCMC) sampler")
    p.add_argument("--prior", choices=["broad", "zoom"], default="zoom",
                   help="Prior mode: broad=global, zoom=around injected theta0 (default: zoom)")
    p.add_argument("--samples", nargs="+", default=["m1", "m2", "p0", "e0"],
                   help=f"Parameters to sample (default: m1 m2 p0 e0)")
    p.add_argument("--tobs", type=float, default=0.1, help="Observation time [yr]")
    p.add_argument("--dt", type=float, default=100.0, help="Sample spacing [s]")
    p.add_argument("--results", default="eryn_results", help="Output directory")
    p.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers (default: 32)")
    p.add_argument("--nsteps", type=int, default=5000, help="Number of MCMC steps (default: 5000)")
    p.add_argument("--burn", type=int, default=1000, help="Burn-in steps to discard (default: 1000)")
    p.add_argument("--thin", type=int, default=1, help="Thinning factor (default: 1)")
    p.add_argument("--init-scatter", type=float, default=1e-7,
                   help="Fractional scatter for walker initialization: inj*(1 + scatter) (default: 1e-7)")
    return p.parse_args()

# ------------------------
# Main
# ------------------------
def main():
    global sample_idx
    args = parse_args()

    # Validate sample names
    for n in args.samples:
        if n not in name_to_idx:
            raise SystemExit(f"Unknown parameter in --samples: {n}")

    sample_names = list(args.samples)
    sample_idx = np.array([name_to_idx[n] for n in sample_names], dtype=int)
    ndim = len(sample_idx)

    print("=" * 60)
    print("EMRI TDI Sampler - Eryn (MCMC)")
    print("=" * 60)
    print(f"Sampling: {sample_names}")
    print(f"Dimensionality: {ndim}")
    print(f"Prior mode: {args.prior}")
    print(f"Walkers: {args.nwalkers}")
    print(f"Steps: {args.nsteps}")
    print(f"Burn-in: {args.burn}")
    print(f"Tobs: {args.tobs} yr")
    print(f"dt: {args.dt} s")
    print("=" * 60)

    # Setup model
    print("\nSetting up model...")
    setup(args.tobs, args.dt)

    # Create prior
    print(f"Creating prior (mode={args.prior})...")
    prior = create_prior(sample_names, theta0, mode=args.prior)

    # Initialize walkers
    print(f"Initializing {args.nwalkers} walkers (scatter={args.init_scatter})...")
    coords = initialize_walkers(sample_names, prior, args.nwalkers,
                                 mode=args.prior, theta0_vals=theta0,
                                 scatter_frac=args.init_scatter)

    # Print initial positions summary
    print("\nInitial walker positions (mean ± std):")
    for i, name in enumerate(sample_names):
        mean_val = np.mean(coords[i])  # Use integer key
        std_val = np.std(coords[i])
        true_val = theta0[name_to_idx[name]]
        print(f"  {name:>10s}: {mean_val:12.6g} ± {std_val:10.4g}  (true: {true_val:12.6g})")

    # Create sampler
    print("\nCreating Eryn sampler...")
    # Note: branch_names should be a single-element list for single-model sampling
    # The actual parameter names are tracked separately
    sampler = EnsembleSampler(
        args.nwalkers,
        ndim,
        log_like_fn,
        prior,
    )

    # Create initial state
    print("Creating initial state...")
    # Convert coords dict to numpy array: (nwalkers, ndim)
    coords_array = np.column_stack([coords[i] for i in range(ndim)])
    state = State(coords_array)

    # Run MCMC
    print(f"\nRunning MCMC for {args.nsteps} steps...")
    print("(This may take a while...)\n")

    sampler.run_mcmc(state, args.nsteps, progress=True, thin_by=args.thin)

    # Save results
    outdir = Path(args.results)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {outdir.resolve()}")

    # Get chains
    chain = sampler.get_chain()  # shape: (nsteps, nwalkers, ndim)
    log_prob = sampler.get_log_prob()  # shape: (nsteps, nwalkers)

    # Save raw chains
    results = {
        'chain': chain,
        'log_prob': log_prob,
        'param_names': sample_names,
        'theta0': theta0,
        'sample_idx': sample_idx,
        'nwalkers': args.nwalkers,
        'nsteps': args.nsteps,
        'burn': args.burn,
        'thin': args.thin,
        'prior_mode': args.prior,
    }

    with open(outdir / "eryn_chains.pkl", "wb") as f:
        pickle.dump(results, f)

    # Compute flattened samples (after burn-in)
    if args.burn > 0:
        chain_burned = chain[args.burn:, :, :]
        log_prob_burned = log_prob[args.burn:, :]
    else:
        chain_burned = chain
        log_prob_burned = log_prob

    flat_samples = chain_burned.reshape(-1, ndim)
    flat_log_prob = log_prob_burned.reshape(-1)

    # Save flattened samples
    np.savez(
        outdir / "eryn_samples.npz",
        samples=flat_samples,
        log_prob=flat_log_prob,
        param_names=sample_names,
    )

    # Print acceptance fraction
    try:
        acceptance = sampler.acceptance_fraction
        print(f"\nMean acceptance fraction: {np.mean(acceptance):.3f}")
        print(f"Acceptance range: [{np.min(acceptance):.3f}, {np.max(acceptance):.3f}]")
    except:
        pass

    # Print summary statistics
    print("\nPosterior summary (after burn-in):")
    print(f"{'Param':<12} {'Mean':<12} {'Std':<12} {'16%':<12} {'50%':<12} {'84%':<12} {'True':<12}")
    print("-" * 84)

    for i, name in enumerate(sample_names):
        samples_i = flat_samples[:, i]
        mean_val = np.mean(samples_i)
        std_val = np.std(samples_i)
        p16, p50, p84 = np.percentile(samples_i, [16, 50, 84])
        true_val = theta0[name_to_idx[name]]
        print(f"{name:<12} {mean_val:<12.6g} {std_val:<12.6g} {p16:<12.6g} {p50:<12.6g} {p84:<12.6g} {true_val:<12.6g}")

    print(f"\nSaved:")
    print(f"  - {outdir / 'eryn_chains.pkl'} (full chains)")
    print(f"  - {outdir / 'eryn_samples.npz'} (flattened samples)")
    print("\nDone!")

if __name__ == "__main__":
    main()
