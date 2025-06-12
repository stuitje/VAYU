import os
import toml
import numpy as np
import pandas as pd
import argparse

from scipy.stats import norm
from src.utils import contrast_ppm, compute_dayside_brightness_temperature
from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.constants import r_earth, r_sun

from dynesty import NestedSampler #This must be placed after importing load_agni_output to avoid a netCDF error

# Load paths
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

def log_likelihood(theta, obs_wavelength, obs_contrast, obs_error, model_wavelength, model_contrast):
    scale = theta[0]
    scaled_model = np.interp(obs_wavelength, model_wavelength, model_contrast * scale)
    resid = (obs_contrast - scaled_model) / obs_error
    return -0.5 * np.sum(resid**2)

def prior_transform(uu):
    scale = norm.ppf(uu[0], loc=1.0, scale=0.077)  # Gaussian prior N(1.0, 0.077) like Zhang et al (2024)
    return [scale]

def compute_log_evidence(model_wavelength, model_contrast, observed_df):
    obs_wavelength = observed_df["X"].values * 1000  # micron to nm
    obs_contrast = observed_df["Y"].values
    obs_error = observed_df["\u0394Y"].values

    sampler = NestedSampler(
        log_likelihood,
        prior_transform,
        ndim=1,
        logl_args=(obs_wavelength, obs_contrast, obs_error, model_wavelength, model_contrast)
    )
    sampler.run_nested(dlogz=0.01)
    return sampler.results.logz[-1]

def compare_models(planet_name, surfaces, atmospheres, use_greybody_reference=False, write_to_csv=True):
    results = []
    contrast_data = load_contrast_data(os.path.join(CONFIG["obs_data_dir"], f"{planet_name}_data.csv"))
    pdata = get_planet_data(planet_name)

    T_star = pdata["star_temp"]
    R_star = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth

    star_name = planet_name[:-1]
    star_path = os.path.join(ROOT, "..", "res", "stellar_spectra", f"{star_name}.txt")
    star_path_SPHINX = os.path.join(ROOT, "..", "res", "stellar_spectra", f"{star_name}_SPHINX.txt")

    if os.path.exists(star_path_SPHINX):
        star_path = star_path_SPHINX

    best_logZ = None
    reference_logZ = None

    for surface in surfaces:
        for atmo in atmospheres:
            if (surface != 'greybody' and atmo != 'bare_rock') or (surface == 'greybody' and atmo == 'bare_rock') :
                continue

            nc_path = os.path.join(CONFIG["output_dir"], planet_name, surface, atmo, "atm.nc")

            if not os.path.exists(nc_path):
                print(f"[SKIP] File missing: {nc_path}")
                continue

            data = load_agni_output(nc_path)
            model_contrast = contrast_ppm(
                wavelength_nm=data["bandcenter"],
                T_star=T_star,
                R_planet_m=R_planet,
                R_star_m=R_star,
                planet_flux=data["ba_U_total"],
                stellar_spectrum=star_path
            )

            logZ = compute_log_evidence(data["bandcenter"], model_contrast, contrast_data)
            print(f"[INFO] {surface} + {atmo}: logZ = {logZ:.2f}")
            results.append({
                "surface": surface,
                "atmosphere": atmo,
                "logZ": logZ
            })

            if best_logZ is None or logZ > best_logZ:
                best_logZ = logZ

    if use_greybody_reference:
        print("[INFO] Computing synthetic greybody reference model (blackbody planet)")
        wavelengths = np.logspace(np.log10(4000), np.log10(20000), 300)

        T_planet = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2/3
        )

        model_contrast = contrast_ppm(
            wavelength_nm=wavelengths,
            T_star=T_star,
            R_planet_m=R_planet,
            R_star_m=R_star,
            T_planet=T_planet,
            stellar_spectrum=star_path
        )

        reference_logZ = compute_log_evidence(wavelengths, model_contrast, contrast_data)
        print(f"[INFO] Synthetic greybody reference logZ = {reference_logZ:.2f}")
    else:
        reference_logZ = best_logZ

    for result in results:
        delta_lnZ = result["logZ"] - reference_logZ
        result["\u0394lnZ"] = delta_lnZ
        result["bayes_factor"] = np.exp(delta_lnZ)

    df = pd.DataFrame(results)

    if write_to_csv:
        df.to_csv(os.path.join(CONFIG["output_dir"], planet_name, "bayes_model_comparison.csv"), index=False)
        print(f"[DONE] Results written to bayes_model_comparison.csv")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian model comparison for AGNI outputs.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g., 'gj367b')")
    parser.add_argument("--use-greybody", action="store_true", help="Use greybody+bare_rock as Bayes factor reference")
    args = parser.parse_args()

    planet = args.planet.lower()

    # Load surface and atmosphere lists 
    surface_list_path = os.path.join(ROOT, "../", "surface_list.toml")
    atmos_list_path = os.path.join(ROOT, "../", "atmos_list.toml")

    surfaces = toml.load(surface_list_path).get("surfaces", [])
    atmospheres = toml.load(atmos_list_path).get("atmospheres", [])

    df_results = compare_models(planet, surfaces, atmospheres, use_greybody_reference=args.use_greybody)
    print(df_results.sort_values("\u0394lnZ"))
