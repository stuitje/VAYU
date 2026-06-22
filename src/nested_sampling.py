"""
Bayesian model comparison for AGNI emission spectra based on Zhang et al 2024.

For a given planet, loops over (surface, atmosphere) combinations, computes
the planet-star contrast spectrum, and runs nested sampling to obtain the
log-evidence ln Z. Bayes factors are reported relative to a chosen
reference model (best-fitting surface by default).

The single free parameter is a Gaussian-prior scale factor on the model
contrast. The prior width is planet-specific based on planet parameters and a 3% flux calibration uncertainty (Zhang et al. 2024)
"""

import os
import argparse
import toml
import numpy as np
import pandas as pd

from scipy.stats import norm
from src.utils import contrast_ppm, compute_dayside_brightness_temperature
from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.constants import r_earth, r_sun

# Must be imported after netCDF4 (pulled in via load_agni_output) to avoid
# a known load-order conflict between dynesty and netCDF4.
from dynesty import NestedSampler

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

# Per-planet Gaussian prior width on the contrast scaling parameter alpha,
# computed from the quadrature sum of stellar-flux, radius-ratio, orbital
# distance, and a 3% flux-calibration uncertainty (Zhang et al. 2024).
SIGMA_SCALE = {
    "gj367b":      0.077,
    "gj486b":      0.042,
    "trappist-1b": 0.040,
    "trappist-1c": 0.040,
}


# Stellar spectrum 

def get_star_spectrum_path(planet_name: str) -> str:
    """Return path to the SPHINX stellar spectrum if present, else the fallback."""
    star_name = planet_name[:-1]
    star_path        = os.path.join(ROOT, "..", "res", "stellar_spectra", f"{star_name}.txt")
    star_path_sphinx = os.path.join(ROOT, "..", "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    return star_path_sphinx if os.path.exists(star_path_sphinx) else star_path


# Likelihood and prior 

def log_likelihood(theta, obs_wavelength, obs_contrast, obs_error,
                   model_wavelength, model_contrast):
    """Gaussian log-likelihood with a single free scale parameter alpha."""
    alpha = theta[0]
    scaled_model = np.interp(obs_wavelength, model_wavelength, model_contrast * alpha)
    resid = (obs_contrast - scaled_model) / obs_error
    return -0.5 * np.sum(resid ** 2)


def make_prior_transform(sigma_scale: float):
    """Factory: returns a prior_transform that samples alpha ~ N(1, sigma_scale^2)."""
    def prior_transform(uu):
        return [norm.ppf(uu[0], loc=1.0, scale=sigma_scale)]
    return prior_transform


#  Evidence

def compute_log_evidence(model_wavelength, model_contrast, observed_df, sigma_scale: float):
    """Run nested sampling and return (logZ, logZ_err)."""
    obs_wavelength = observed_df["X"].values * 1000  # micron to nm
    obs_contrast   = observed_df["Y"].values
    obs_error      = observed_df["\u0394Y"].values

    sampler = NestedSampler(
        log_likelihood,
        make_prior_transform(sigma_scale),
        ndim=1,
        logl_args=(obs_wavelength, obs_contrast, obs_error, model_wavelength, model_contrast),
    )
    sampler.run_nested(dlogz=0.01, print_progress=False)
    return sampler.results.logz[-1], sampler.results.logzerr[-1]


#  Forward modelling

def model_contrast_from_agni(nc_path: str, T_star, R_planet, R_star, star_path):
    """Load AGNI output and return (wavelength_nm, contrast_ppm)."""
    data = load_agni_output(nc_path)
    contrast = contrast_ppm(
        wavelength_nm=data["bandcenter"],
        T_star=T_star,
        R_planet_m=R_planet,
        R_star_m=R_star,
        planet_flux=data["ba_U_total"],
        stellar_spectrum=star_path,
    )
    return data["bandcenter"], contrast


# Model comparison 

def is_valid_pair(surface: str, atmosphere: str) -> bool:
    """Only (surface + bare_rock) and (greybody + atmosphere) combinations are physical."""
    surface_only_run = (atmosphere == "bare_rock" and surface != "greybody")
    atmo_only_run    = (surface == "greybody" and atmosphere != "bare_rock")
    return surface_only_run or atmo_only_run


def compare_models(planet_name, surfaces, atmospheres,
                   reference_surface=None, write_to_csv=True):
    """Run nested sampling over all (surface, atmosphere) combinations and return a results DataFrame.

    Parameters
    ----------
    planet_name : str
        Planet identifier matching planet_csv and SIGMA_SCALE keys (e.g. 'gj486b').
    surfaces, atmospheres : list of str
        Surface/atmosphere identifiers to loop over.
    reference_surface : str, optional
        Name of the reference surface for Bayes factors. Special value 'greybody' computes
        a synthetic blackbody reference. If None, the best-fitting model is used as reference.
    """
    if planet_name not in SIGMA_SCALE:
        raise KeyError(
            f"No prior-width sigma defined for planet '{planet_name}'. "
            f"Add it to SIGMA_SCALE before running."
        )
    sigma_scale = SIGMA_SCALE[planet_name]

    contrast_data = load_contrast_data(os.path.join(CONFIG["obs_data_dir"], f"{planet_name}_data.csv"))
    pdata = get_planet_data(planet_name)

    T_star   = pdata["star_temp"]
    R_star   = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth

    star_path = get_star_spectrum_path(planet_name)

    results = []
    best_logZ = None

    for surface in surfaces:
        for atmo in atmospheres:
            if not is_valid_pair(surface, atmo):
                continue

            nc_path = os.path.join(CONFIG["output_dir"], planet_name, surface, atmo, "atm.nc")
            if not os.path.exists(nc_path):
                print(f"[SKIP] File missing: {nc_path}")
                continue

            wl, model = model_contrast_from_agni(nc_path, T_star, R_planet, R_star, star_path)
            logZ, logZ_err = compute_log_evidence(wl, model, contrast_data, sigma_scale)

            print(f"[INFO] {surface} + {atmo}: logZ = {logZ:.2f} +/- {logZ_err:.2f}")
            results.append({
                "surface": surface,
                "atmosphere": atmo,
                "logZ": logZ,
                "logZ_err": logZ_err,
            })

            if best_logZ is None or logZ > best_logZ:
                best_logZ = logZ

    # Determine reference logZ
    reference_logZ = _resolve_reference_logZ(
        reference_surface, results, best_logZ, planet_name,
        T_star, R_planet, R_star, star_path, sigma_scale, contrast_data, pdata,
    )

    for r in results:
        r["\u0394lnZ"]       = r["logZ"] - reference_logZ
        r["bayes_factor"]    = float(np.exp(r["\u0394lnZ"]))

    df = pd.DataFrame(results)

    if write_to_csv:
        output_path = os.path.join(CONFIG["output_dir"], planet_name, "bayes_model_comparison.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DONE] Results written to {output_path}")

    return df


def _resolve_reference_logZ(reference_surface, results, best_logZ, planet_name,
                            T_star, R_planet, R_star, star_path, sigma_scale,
                            contrast_data, pdata):
    """Return the logZ value to use as the Bayes-factor denominator."""

    if reference_surface == "greybody":
        print("[INFO] Computing synthetic greybody reference model (blackbody planet)")
        wavelengths = np.logspace(np.log10(4000), np.log10(20000), 300)
        T_planet = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2 / 3,
        )
        model_contrast = contrast_ppm(
            wavelength_nm=wavelengths,
            T_star=T_star,
            R_planet_m=R_planet,
            R_star_m=R_star,
            T_planet=T_planet,
            stellar_spectrum=star_path,
        )
        logZ, _ = compute_log_evidence(wavelengths, model_contrast, contrast_data, sigma_scale)
        print(f"[INFO] Synthetic greybody reference logZ = {logZ:.2f}")
        return logZ

    if reference_surface:
        # Look for the reference in the already-computed results
        ref = next((r for r in results
                    if r["surface"] == reference_surface and r["atmosphere"] == "bare_rock"),
                   None)
        if ref is not None:
            print(f"[INFO] Using {reference_surface}+bare_rock as reference, logZ = {ref['logZ']:.2f}")
            return ref["logZ"]

        # Otherwise try to load it on the fly
        nc_path = os.path.join(CONFIG["output_dir"], planet_name, reference_surface, "bare_rock", "atm.nc")
        if os.path.exists(nc_path):
            print(f"[INFO] Loading external reference model from {nc_path}")
            wl, model = model_contrast_from_agni(nc_path, T_star, R_planet, R_star, star_path)
            logZ, _ = compute_log_evidence(wl, model, contrast_data, sigma_scale)
            print(f"[INFO] Reference {reference_surface}+bare_rock logZ = {logZ:.2f}")
            return logZ

        print(f"[WARNING] Reference model {reference_surface}+bare_rock not found. Using best model.")
        return best_logZ

    return best_logZ


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian model comparison for AGNI outputs.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g. 'gj486b')")
    parser.add_argument("--ref", type=str, help="Reference surface for Bayes factor (e.g. 'greybody' or 'hematite')")
    args = parser.parse_args()

    planet = args.planet.lower()

    surface_list_path = os.path.join(ROOT, "..", "surface_list.toml")
    atmos_list_path   = os.path.join(ROOT, "..", "atmos_list.toml")
    surfaces    = toml.load(surface_list_path).get("surfaces", [])
    atmospheres = toml.load(atmos_list_path).get("atmospheres", [])

    df_results = compare_models(planet, surfaces, atmospheres, reference_surface=args.ref)
    print(df_results.sort_values("\u0394lnZ"))