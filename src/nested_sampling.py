import os
import toml
import numpy as np
import pandas as pd
import argparse

from scipy.stats import norm
from src.utils import contrast_ppm, compute_dayside_brightness_temperature
from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.constants import r_earth, r_sun
from src.stat import compute_chi_squared 

from dynesty import NestedSampler  # Must be placed after importing load_agni_output to avoid netCDF error

# Load paths
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

def get_star_spectrum_info(planet_name: str):
    star_name = planet_name[:-1]
    star_path = os.path.join(ROOT, "..", "res", "stellar_spectra", f"{star_name}.txt")
    star_path_sphinx = os.path.join(ROOT, "..", "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    if os.path.exists(star_path_sphinx):
        star_path = star_path_sphinx

    if star_name == 'gj486':
        return star_path, True, 3300
    elif star_name == 'gj367':
        return star_path, True, 3500
    elif star_name == 'trappist-1':
        return star_path, False, None
    else:
        return star_path, True, None

def log_likelihood(theta, obs_wavelength, obs_contrast, obs_error, model_wavelength, model_contrast):
    scale = theta[0]
    scaled_model = np.interp(obs_wavelength, model_wavelength, model_contrast * scale)
    resid = (obs_contrast - scaled_model) / obs_error
    return -0.5 * np.sum(resid**2)

def prior_transform(uu):
    scale = norm.ppf(uu[0], loc=1.0, scale=0.0402) #0.077
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

def compare_models(planet_name, surfaces, atmospheres, reference_surface=None, write_to_csv=True):
    results = []
    contrast_data = load_contrast_data(os.path.join(CONFIG["obs_data_dir"], f"{planet_name}_data.csv"))
    pdata = get_planet_data(planet_name)

    T_star = pdata["star_temp"]
    R_star = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth

    star_path, rescale, T_spectrum = get_star_spectrum_info(planet_name)

    best_logZ = None
    reference_logZ = None

    for surface in surfaces:
        for atmo in atmospheres:
            if (surface != 'greybody' and atmo != 'bare_rock') or (surface == 'greybody' and atmo == 'bare_rock'):
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
                stellar_spectrum=star_path,
                rescale=rescale,
                T_spectrum=T_spectrum
            )

            logZ = compute_log_evidence(data["bandcenter"], model_contrast, contrast_data)
            chi2_red = compute_chi_squared(contrast_data, data["bandcenter"], model_contrast)

            print(f"[INFO] {surface} + {atmo}: logZ = {logZ:.2f}")
            results.append({
                "surface": surface,
                "atmosphere": atmo,
                "logZ": logZ,
                "chi2_red": chi2_red
            })

            if best_logZ is None or logZ > best_logZ:
                best_logZ = logZ

    # Determine reference_logZ
    if reference_surface == "greybody":
        print("[INFO] Computing synthetic greybody reference model (blackbody planet)")
        wavelengths = np.logspace(np.log10(4000), np.log10(20000), 300)

        T_planet = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2 / 3
        )

        model_contrast = contrast_ppm(
            wavelength_nm=wavelengths,
            T_star=T_star,
            R_planet_m=R_planet,
            R_star_m=R_star,
            T_planet=T_planet,
            stellar_spectrum=star_path,
            rescale=rescale,
            T_spectrum=T_spectrum
        )

        reference_logZ = compute_log_evidence(wavelengths, model_contrast, contrast_data)
        print(f"[INFO] Synthetic greybody reference logZ = {reference_logZ:.2f}")

    elif reference_surface:
        ref_result = next((r for r in results if r["surface"] == reference_surface and r["atmosphere"] == "bare_rock"), None)

        if ref_result:
            reference_logZ = ref_result["logZ"]
            print(f"[INFO] Using {reference_surface}+bare_rock from results as reference with logZ = {reference_logZ:.2f}")
        else:
            nc_path = os.path.join(CONFIG["output_dir"], planet_name, reference_surface, "bare_rock", "atm.nc")
            if os.path.exists(nc_path):
                print(f"[INFO] Loading external reference model from {nc_path}")
                data = load_agni_output(nc_path)
                model_contrast = contrast_ppm(
                    wavelength_nm=data["bandcenter"],
                    T_star=T_star,
                    R_planet_m=R_planet,
                    R_star_m=R_star,
                    planet_flux=data["ba_U_total"],
                    stellar_spectrum=star_path,
                    rescale=rescale,
                    T_spectrum=T_spectrum
                )
                reference_logZ = compute_log_evidence(data["bandcenter"], model_contrast, contrast_data)
                print(f"[INFO] Loaded reference {reference_surface}+bare_rock logZ = {reference_logZ:.2f}")
            else:
                print(f"[WARNING] Reference model {reference_surface}+bare_rock not found. Using best model as fallback.")
                reference_logZ = best_logZ
    else:
        reference_logZ = best_logZ

    for result in results:
        delta_lnZ = result["logZ"] - reference_logZ
        result["\u0394lnZ"] = delta_lnZ
        result["bayes_factor"] = np.exp(delta_lnZ)

    df = pd.DataFrame(results)

    if write_to_csv:
        output_path = os.path.join(CONFIG["output_dir"], planet_name, "bayes_model_comparison.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DONE] Results written to bayes_model_comparison.csv")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian model comparison for AGNI outputs.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g., 'gj367b')")
    parser.add_argument("--ref", type=str, help="Reference surface for Bayes factor (e.g., 'greybody' or 'hematite')")
    args = parser.parse_args()

    planet = args.planet.lower()

    # Load surface and atmosphere lists
    surface_list_path = os.path.join(ROOT, "../", "surface_list.toml")
    atmos_list_path = os.path.join(ROOT, "../", "atmos_list.toml")

    surfaces = toml.load(surface_list_path).get("surfaces", [])
    atmospheres = toml.load(atmos_list_path).get("atmospheres", [])

    df_results = compare_models(planet, surfaces, atmospheres, reference_surface=args.ref)
    print(df_results.sort_values("\u0394lnZ"))
