import os
import argparse
import toml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.dataloader import load_agni_output, load_contrast_data, get_planet_data
from src.utils import compute_dayside_brightness_temperature, contrast_ppm
from src.constants import r_earth, r_sun

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

def get_atmospheres():
    atmos_list_path = os.path.join(ROOT, "..", "atmos_list.toml")
    atmospheres = toml.load(atmos_list_path).get("atmospheres", [])
    return atmospheres

def load_literary_model(filepath):
    df = pd.read_csv(
        filepath,
        delim_whitespace=True,
        header=None,
        comment='#',
        names=[
            "bin", "wavelength", "wavelength_low_int",
            "wavelength_delta", "flux_down", "flux_up", "contrast"
        ]
    )
    df["contrast"] = df["contrast"] * 1e6  # Convert to ppm
    return df

def plot_single_atmo_vs_literature(agni_flux, agni_wavelengths_nm, lit_wavelengths_um, lit_contrast_ppm, planet_name, agni_label, lit_label, T_star, R_planet_rearth, R_star_rsun, observed_df):
    agni_wavelengths_um = agni_wavelengths_nm / 1000  # Convert nm to µm

    contrast_agni = contrast_ppm(
        wavelength_nm=agni_wavelengths_nm,
        T_star=T_star,
        R_planet_m=R_planet_rearth * r_earth,
        R_star_m=R_star_rsun * r_sun,
        planet_flux=agni_flux
    )

    window_size = 40
    lit_contrast_smoothed = pd.Series(lit_contrast_ppm).rolling(window=window_size, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(agni_wavelengths_um, contrast_agni, label=agni_label, color='dodgerblue')
    ax.plot(lit_wavelengths_um, lit_contrast_smoothed, label=lit_label, color='crimson')

    ax.set_xlabel(f"Wavelength [$\mu$m]")
    ax.set_ylabel("Contrast [ppm]")
    ax.set_title(f"{planet_name} – AGNI vs. Mansfield")
    ax.set_xlim(4, 20)
    ax.set_ylim(0, 400)
    ax.legend()
    ax.grid(True, alpha = 0.3)

    if observed_df is not None:
        ax.errorbar(observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
                     fmt="o", color="blue",  ecolor='gray', capsize=2, label="Observed")

    fig.tight_layout()

    # Save to the proper output directory
    out_dir = os.path.join(CONFIG["output_dir"], planet_name)
    os.makedirs(out_dir, exist_ok=True)
    figpath = os.path.join(out_dir, "agni_vs_literature.png")

    fig.savefig(figpath)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot AGNI vs. literary atmosphere.")
    parser.add_argument("planet", help="Planet name")
    parser.add_argument("--literary", help="Filename of literary model (.dat file) in Mansfield_2024")
    parser.add_argument("--surface", required=False, help="Surface name")

    args = parser.parse_args()

    if args.literary and "bare" not in args.literary:
        surface = 'greybody'
    else:
        surface = args.surface

    pdata = get_planet_data(args.planet)
    T_star, R_star, R_planet, a_planet = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"], pdata["planet_a"]

    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{args.planet}_data.csv")
    contrast_data = load_contrast_data(contrast_path)

    fluxes = {}
    agni_wavelengths = None

    for atmo in get_atmospheres():
        nc_path = os.path.join(CONFIG["output_dir"], args.planet, surface, atmo, "atm.nc")
        if os.path.isfile(nc_path):
            data = load_agni_output(nc_path)
            if agni_wavelengths is None:
                agni_wavelengths = np.asarray(data["bandcenter"], dtype=float)
                agni_wavelengths = agni_wavelengths[~np.isnan(agni_wavelengths)]
            fluxes[atmo] = data["ba_U_total"]
        else:
            print(f"[SKIP] Missing AGNI output: {nc_path}")

    # Load and include literary model
    if args.literary:
        literary_path = os.path.join(CONFIG["obs_data_dir"], "Mansfield_2024", f"{args.planet}_{args.literary}_post_TOA_flux_eclipse.dat")
        if os.path.isfile(literary_path):
            literary_df = load_literary_model(literary_path)

            if fluxes:
                first_atmo = list(fluxes.keys())[0]
                agni_flux = fluxes[first_atmo]

                plot_single_atmo_vs_literature(
                    agni_flux=agni_flux,
                    agni_wavelengths_nm=agni_wavelengths,
                    lit_wavelengths_um=literary_df["wavelength"].values,
                    lit_contrast_ppm=literary_df["contrast"].values,
                    planet_name=args.planet,
                    agni_label=f"AGNI: {first_atmo}",
                    lit_label=F"Mansfield 2024: {args.literary}",
                    T_star=T_star,
                    R_planet_rearth=R_planet,
                    R_star_rsun=R_star,
                    observed_df=contrast_data
                )
                return
            else:
                print("[ERROR] No AGNI data loaded. Cannot compare.")
                return
        else:
            print(f"[ERROR] Literary model file not found: {literary_path}")
            return
    
    # might need it for later
    T_planet = compute_dayside_brightness_temperature(
        stellar_temperature=T_star,
        stellar_radius_rsun=R_star,
        distance_au=a_planet,
        bond_albedo=0,
        redistribution_factor=2/3
    )

if __name__ == "__main__":
    main()
