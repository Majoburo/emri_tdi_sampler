import numpy as np
import pickle
from pathlib import Path
from ultranest import ReactiveNestedSampler

# FEW + LISA Tools
from few.waveform import GenerateEMRIWaveform
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.sensitivity import AE1SensitivityMatrix
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
# ------------------------
# 1) Configure generator
# ------------------------
Tobs = 1.  # 3 months
dt   = 100.0              # seconds


gen_wave = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
)

resp = ResponseWrapper(
    gen_wave, Tobs, dt,
    index_lambda=8, index_beta=7,  # EMRITDIWaveform uses these
    flip_hx=True,
    remove_sky_coords=False,
    is_ecliptic_latitude=False,
    remove_garbage=True,
    t0=100000., order=25,
    tdi="1st generation", tdi_chan="AET",
    orbits=EqualArmlengthOrbits(),
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

#5.770401  1.422654  0.843247  10.954791  0.01203  0.027183  1.580719

theta0 = np.array(theta0, dtype=float)

param_names = ["m1","m2","a","p0","e0","x0","dist","qS","phiS","qK","phiK","Phi_phi0","Phi_theta0","Phi_r0"]
name_to_idx = {n:i for i,n in enumerate(param_names)}

# sample just these:
sample_names = ["m1", "m2", "p0", "e0"]
sample_idx = np.array([name_to_idx[n] for n in sample_names], dtype=int)
print("Sampling:", sample_names, " -> indices:", sample_idx.tolist())

# Choose prior mode: "broad" (global, generic) or "zoom" (around injected theta0)
PRIOR_MODE = "zoom"   # "broad" is global; "zoom" focuses around theta0 values

# ------------------------
# 2) Generate A/E and freqs
# ------------------------
out = resp(*theta0)

A, E, _ = out
freqs = np.fft.rfftfreq(2*(len(A)-1), d=dt)[1:]
A, E = A[1:], E[1:]

# Ensure arrays are compatible complex FD on same f-grid
A = np.asarray(A, dtype=complex)
E = np.asarray(E, dtype=complex)
freqs = np.asarray(freqs)

data_fd = np.vstack([A, E])
# Keep copies of the "observed" A/E used as data for later plotting/storage

# ------------------------
# 3) Build LISATools likelihood
# ------------------------
data = DataResidualArray(data_fd, f_arr=freqs)
sens_mat = AE1SensitivityMatrix(freqs)
analysis = AnalysisContainer(data, sens_mat)

def loglike(theta_sub):
    theta_full = theta0.copy()
    theta_full[sample_idx] = np.asarray(theta_sub, dtype=float)

    out = resp(*theta_full)
    A_, E_, _ = out
    A_, E_ = A_[1:], E_[1:]

    template = DataResidualArray(np.vstack([A_, E_]), f_arr=freqs)
    return analysis.template_likelihood(template)

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

# Build prior from user-defined sample_names
prior_transform = PriorTransform(sample_names, theta0, mode=PRIOR_MODE)
print(f"[prior] mode={PRIOR_MODE}")
for n in sample_names:
    print(f"  {n:>10s}: {prior_transform.desc()[n]}")

# ------------------------
# 5) Run UltraNest
# ------------------------
ndim = len(sample_idx)
results_dir = Path("./ultra_results")
results_dir.mkdir(parents=True, exist_ok=True)

# UltraNest sampler with resume + logs under results_dir
sampler = ReactiveNestedSampler(
    sample_names,
    loglike,
    transform=prior_transform,
    log_dir=str(results_dir),
    resume=True,
)

# A sensible heuristic for live points: scale with dimension
min_live = max(400, 50 * ndim)
result = sampler.run(
    min_num_live_points=min_live,
    dlogz=0.3,
)

# Optional: print a short summary to stdout
try:
    sampler.print_results()
except Exception:
    pass

# Save a compact result pickle for later plotting
pkl_path = results_dir / "ultranest_result.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(result, f)

print(f"\nSaved UltraNest result pickle to: {pkl_path.resolve()}")
print(f"UltraNest run directory: {results_dir.resolve()} (contains logs, checkpoints, and plots)")
