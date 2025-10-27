import numpy as np
import pickle
from pathlib import Path
from ultranest import ReactiveNestedSampler
import argparse
import os

# Parallelization support
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    HAS_MPI = True
except ImportError:
    comm = None
    rank = 0
    size = 1
    HAS_MPI = False

# GPU support
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

# FEW + LISA Tools
from few.waveform import GenerateEMRIWaveform
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.sensitivity import AE1SensitivityMatrix
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
# ------------------------
# Configuration & CLI
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EMRI TDI UltraNest sampler with parallelization")
    parser.add_argument("--prior", choices=["broad", "zoom"], default="zoom",
                        help="Prior mode: broad=global, zoom=around injected theta0")
    parser.add_argument("--samples", nargs="+", default=["m1", "m2", "p0", "e0"],
                        help="Parameters to sample")
    parser.add_argument("--tobs", type=float, default=0.1, help="Observation time [yr]")
    parser.add_argument("--dt", type=float, default=100.0, help="Sample spacing [s]")
    parser.add_argument("--results", default="ultra_results", help="Output directory")
    parser.add_argument("--min-live", type=int, default=None,
                        help="Min live points (default: max(400, 50*ndim))")
    parser.add_argument("--dlogz", type=float, default=0.3, help="Evidence tolerance")
    parser.add_argument("--use-mpi", action="store_true",
                        help="Use MPI parallelization (requires mpi4py)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU acceleration")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU device ID (for multi-GPU, default: rank %% n_gpus)")
    parser.add_argument("--vectorized", action="store_true",
                        help="Use vectorized likelihood (GPU batch mode)")
    return parser.parse_args()

# ------------------------
# 1) Configure generator (global for all ranks)
# ------------------------
gen_wave = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
)

theta0 = [
    10**5.770401,         # m1: central object mass (solar masses)
    10**1.422654,          # m2: secondary object mass (solar masses)
    0.843247,           # a: spin parameter
    10.954791,          # p0: initial semi-latus rectum
    0.01203,           # e0: eccentricity
    1.0,           # x0: cos(inclination)
    1.0,           # dist: luminosity distance (Gpc)
    0.01,     # qS: polar sky angle
    np.pi / 3,     # phiS: azimuthal viewing angle
    np.pi / 3,     # qK: polar spin angle
    np.pi / 3,     # phiK: azimuthal spin angle
    np.pi / 3,     # Phi_phi0: initial phase (azimuthal)
    0.0,           # Phi_theta0: initial phase (polar)
    np.pi / 3,     # Phi_r0: initial phase (radial)
]


theta0 = np.array(theta0, dtype=float)

param_names = ["m1","m2","a","p0","e0","x0","dist","qS","phiS","qK","phiK","Phi_phi0","Phi_theta0","Phi_r0"]
name_to_idx = {n:i for i,n in enumerate(param_names)}

# ------------------------
# 4) Priors: cube -> physical (automatic from sample_names)
#    NOTE: Implemented as a top-level callable class so dynesty can pickle it.
# ------------------------
class PriorTransform:
    """
    Pickle-safe callable used by dynesty to map unit-cube u -> parameter subset.
    Supports two modes:
      - 'broad' global ranges
      - 'zoom' windows centered on the injected theta0
    """
    _ordered_param_names = ["m1","m2","a","p0","e0","x0","dist","qS","phiS","qK","phiK","Phi_phi0","Phi_theta0","Phi_r0"]

    def __init__(self, sample_names, theta0, mode="broad"):
        self.sample_names = list(sample_names)
        self.mode = str(mode)
        self.theta0 = np.asarray(theta0, dtype=float)
        self.name_to_idx = {n: i for i, n in enumerate(self._ordered_param_names)}
        # Cache injected values by name
        self.t0 = {n: self.theta0[self.name_to_idx[n]] for n in self._ordered_param_names}

    # ---- helpers (static, pickle-safe) ----
    @staticmethod
    def _wrap_angle(val):
        twopi = 2*np.pi
        return (val + twopi) % twopi

    @staticmethod
    def _scale(center, frac_width, u, lo=None, hi=None, positive=False):
        """Symmetric fractional window around center: center*(1 ± frac_width/2)."""
        val = center * (1.0 - 0.5*frac_width + frac_width * u)
        if positive:
            val = np.maximum(val, np.finfo(float).tiny)
        if lo is not None:
            val = np.maximum(val, lo)
        if hi is not None:
            val = np.minimum(val, hi)
        return val

    # ---- value rules ----
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
        if name == "Phi_phi0":  return 2*np.pi*u                     # [0,2π)
        if name == "Phi_theta0":return 2*np.pi*u                     # [0,2π)
        if name == "Phi_r0":    return 2*np.pi*u                     # [0,2π)
        raise KeyError(f"Unknown parameter name: {name}")

    def _zoom_value(self, name, u):
        t0 = self.t0
        if name == "m1":        return self._scale(t0["m1"], 0.20, u, lo=1e5, hi=1e8, positive=True)
        if name == "m2":        return self._scale(t0["m2"], 0.20, u, lo=1.0, hi=1e3, positive=True)
        if name == "a":         return np.clip(t0["a"] + (u-0.5)*0.2, 0.0, 0.999)   # ±0.1
        if name == "p0":        return self._scale(t0["p0"], 0.20, u, lo=7.5, hi=30.0, positive=True)
        if name == "e0":
            if t0["e0"] > 0:
                return np.clip(t0["e0"] + (u-0.5)*0.4*t0["e0"], 0.0, 0.75)
            else:
                return np.clip(0.2*u, 0.0, 0.75)
        if name == "x0":        return np.clip(t0["x0"] + (u-0.5)*0.4, -1.0, 1.0)   # ±0.2
        if name == "dist":      return self._scale(t0["dist"], 1.0, u, lo=1e-3, hi=1e3, positive=True)  # ~0.5x..1.5x
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
            if self.mode == "broad":
                out[i] = self._broad_value(name, u[i])
            elif self.mode == "zoom":
                out[i] = self._zoom_value(name, u[i])
            else:
                raise ValueError(f"Unknown prior mode: {self.mode}")
        return out

    # Human-readable description of the active prior per sampled parameter
    def desc(self):
        if self.mode == "broad":
            return {
                "m1": "log-uniform [1e5, 1e8] Msun",
                "m2": "log-uniform [10, 100] Msun",
                "a": "[0, 0.999]",
                "p0": "[7.5, 30]",
                "e0": "[0, 0.75]",
                "x0": "[-1, 1]",
                "dist": "log-uniform [1e-3, 1e3] Gpc",
                "qS": "[0, π]",
                "phiS": "[0, 2π)",
                "qK": "[0, π]",
                "phiK": "[0, 2π)",
                "Phi_phi0": "[0, 2π)",
                "Phi_theta0": "[0, 2π)",
                "Phi_r0": "[0, 2π)",
            }
        else:
            t0 = self.t0
            return {
                "m1": f"±10% around {t0['m1']:.3g}",
                "m2": f"±10% around {t0['m2']:.3g}",
                "a":  f"±0.1 around {t0['a']:.3g}",
                "p0": f"±10% around {t0['p0']:.3g}",
                "e0": f"~±20%*e0 around {t0['e0']:.3g} (clamped [0,0.75])",
                "x0": f"±0.2 around {t0['x0']:.3g} (clamped [-1,1])",
                "dist": f"~0.5x–1.5x around {t0['dist']:.3g} Gpc (clamped [1e-3,1e3])",
                "qS": "±π/4 around injected (clamped [0,π])",
                "phiS": "±π/2 around injected (wrapped)",
                "qK": "±π/4 around injected (clamped [0,π])",
                "phiK": "±π/2 around injected (wrapped)",
                "Phi_phi0": "±π/2 around injected (wrapped)",
                "Phi_theta0": "±π/2 around injected (wrapped)",
                "Phi_r0": "±π/2 around injected (wrapped)",
            }

# ------------------------
# GPU Device Selection
# ------------------------
def setup_gpu(gpu_id=None):
    """Setup GPU device if available"""
    if not HAS_GPU:
        if rank == 0:
            print("[warning] --use-gpu specified but cupy not available. Using CPU.")
        return False

    if gpu_id is None:
        # Auto-select based on MPI rank
        n_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_id = rank % n_gpus

    cp.cuda.Device(gpu_id).use()
    if rank == 0 or True:  # Let each rank report
        print(f"[rank {rank}] Using GPU {gpu_id} / {cp.cuda.runtime.getDeviceCount()}")
    return True

# ------------------------
# Vectorized Likelihood (for GPU batching)
# ------------------------
def make_vectorized_loglike(resp, analysis, theta0, sample_idx, freqs):
    """
    Create a vectorized likelihood that can evaluate multiple parameter sets at once.
    This is beneficial for GPU batching.

    Returns:
        Callable that accepts (N, ndim) array and returns (N,) array of log-likelihoods
    """
    def loglike_vec(theta_batch):
        """
        theta_batch: (N, ndim) array where N is batch size
        returns: (N,) array of log-likelihoods
        """
        N = theta_batch.shape[0]
        logls = np.zeros(N, dtype=float)

        for i in range(N):
            theta_full = theta0.copy()
            theta_full[sample_idx] = theta_batch[i]

            out = resp(*theta_full)
            A_, E_, _ = out
            A_, E_ = A_[1:], E_[1:]

            template = DataResidualArray(np.vstack([A_, E_]), f_arr=freqs)
            logls[i] = analysis.template_likelihood(template)

        return logls

    return loglike_vec

# ------------------------
# Main execution
# ------------------------
def main():
    args = parse_args()

    # MPI setup
    if args.use_mpi and not HAS_MPI:
        if rank == 0:
            print("[warning] --use-mpi specified but mpi4py not available")

    # GPU setup
    using_gpu = False
    if args.use_gpu:
        using_gpu = setup_gpu(args.gpu_id)

    # Only rank 0 prints configuration
    if rank == 0:
        print("="*60)
        print("EMRI TDI Sampler - UltraNest")
        print("="*60)
        print(f"MPI: {'Yes' if HAS_MPI and size > 1 else 'No'} (size={size})")
        print(f"GPU: {'Yes' if using_gpu else 'No'}")
        print(f"Vectorized: {args.vectorized}")
        print(f"Tobs: {args.tobs} yr")
        print(f"dt: {args.dt} s")
        print(f"Sampling: {args.samples}")
        print(f"Prior mode: {args.prior}")
        print("="*60)

    # Build sample indices
    sample_names_list = list(args.samples)
    sample_idx_arr = np.array([name_to_idx[n] for n in sample_names_list], dtype=int)

    if rank == 0:
        print("Sampling:", sample_names_list, " -> indices:", sample_idx_arr.tolist())

    # Determine backend for ResponseWrapper
    force_backend = "cpu"
    if using_gpu:
        # Auto-detect CUDA version
        try:
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_version // 1000
            if major >= 12:
                force_backend = "cuda12x"
            else:
                force_backend = "cuda11x"
            if rank == 0:
                print(f"[backend] Using {force_backend} (CUDA {major}.x)")
        except Exception:
            force_backend = "cpu"
    print(f"Using backend: {force_backend}")

    # Build response model (each rank gets its own)
    resp_local = ResponseWrapper(
        gen_wave, args.tobs, args.dt,
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

    # Generate data (each rank does this - could optimize to broadcast)
    out = resp_local(*theta0)
    A, E, _ = out
    freqs_arr = np.fft.rfftfreq(2*(len(A)-1), d=args.dt)[1:]
    A, E = A[1:], E[1:]
    A = np.asarray(A, dtype=complex)
    E = np.asarray(E, dtype=complex)
    freqs_arr = np.asarray(freqs_arr)
    data_fd = np.vstack([A, E])

    # Build likelihood components
    data = DataResidualArray(data_fd, f_arr=freqs_arr)
    sens_mat = AE1SensitivityMatrix(freqs_arr)
    analysis_local = AnalysisContainer(data, sens_mat)

    # Create likelihood function
    def loglike_single(theta_sub):
        theta_full = theta0.copy()
        theta_full[sample_idx_arr] = np.asarray(theta_sub, dtype=float)
        out = resp_local(*theta_full)
        A_, E_, _ = out
        A_, E_ = A_[1:], E_[1:]
        template = DataResidualArray(np.vstack([A_, E_]), f_arr=freqs_arr)
        return analysis_local.template_likelihood(template)

    # Choose likelihood function
    if args.vectorized:
        loglike_fn = make_vectorized_loglike(resp_local, analysis_local, theta0, sample_idx_arr, freqs_arr)
        if rank == 0:
            print("[likelihood] Using vectorized mode")
    else:
        loglike_fn = loglike_single
        if rank == 0:
            print("[likelihood] Using single-evaluation mode")

    # Build prior
    prior_transform_obj = PriorTransform(sample_names_list, theta0, mode=args.prior)
    if rank == 0:
        print(f"[prior] mode={args.prior}")
        for n in sample_names_list:
            print(f"  {n:>10s}: {prior_transform_obj.desc()[n]}")

    # Setup output directory
    results_dir = Path(args.results)
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)

    # Wait for directory creation
    if HAS_MPI and size > 1:
        comm.Barrier()

    # Determine min live points
    ndim = len(sample_idx_arr)
    min_live = args.min_live if args.min_live is not None else max(400, 50 * ndim)

    if rank == 0:
        print(f"[sampler] ndim={ndim}, min_live_points={min_live}, dlogz={args.dlogz}")

    # Create sampler
    sampler = ReactiveNestedSampler(
        sample_names_list,
        loglike_fn,
        transform=prior_transform_obj,
        log_dir=str(results_dir),
        resume=True,
        vectorized=args.vectorized,  # Tell UltraNest if likelihood is vectorized
    )

    # Run sampling
    if rank == 0:
        print("[sampler] Starting nested sampling...")

    result = sampler.run(
        min_num_live_points=min_live,
        dlogz=args.dlogz,
    )

    # Only rank 0 saves and prints results
    if rank == 0:
        try:
            sampler.print_results()
        except Exception:
            pass

        pkl_path = results_dir / "ultranest_result.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        print(f"\nSaved UltraNest result pickle to: {pkl_path.resolve()}")
        print(f"UltraNest run directory: {results_dir.resolve()}")

if __name__ == "__main__":
    main()
