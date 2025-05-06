import os
import argparse
import subprocess
import pandas as pd
import toml

from src.config_gen import write_agni_config
from src.temperature_fit import fit_planet_temperature
from src.plots import (
    plot_bandflux_and_contrast,
    plot_contrasts_multi_atmosphere,
    load_contrast_data
)
from src.utils import load_agni_output, compute_equilibrium_temperature

# Load path config
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

def get_planet_data(name: str) -> dict:
    df = pd.read_csv(CONFIG["planet_csv"])
    row = df[df["planet"].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Planet '{name}' not found.")
    return row.iloc[0].to_dict()


def get_surfaces():
    return sorted(
        f.replace(".dat", "") for f in os.listdir(CONFIG["surface_dir"])
        if f.endswith(".dat")
    )


def get_atmospheres():
    return sorted(
        f.replace(".toml", "") for f in os.listdir(CONFIG["atmosphere_dir"])
        if f.endswith(".toml")
    )


def main():
    parser = argparse.ArgumentParser(description="Run AGNI + plot for planet/surface/atmo setup.")
    parser.add_argument("planet")
    parser.add_argument("-s", "--surface", required=True, help="'all' or surface name")
    parser.add_argument("-a", "--atmosphere", required=True, help="'all', 'list', or atmosphere name")
    parser.add_argument("--flux", default="false", choices=["true", "false"])
    args = parser.parse_args()

    flux_mode = args.flux.lower() == "true"
    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{args.planet}_data.csv")
    contrast_data = load_contrast_data(contrast_path)

    pdata = get_planet_data(args.planet)
    T_star, R_star, R_planet = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"]

    if contrast_data is not None:
        T_planet, _ = fit_planet_temperature(
            csv_path=contrast_path,
            T_star=T_star,
            R_star=R_star,
            R_planet=R_planet
        )
    else:
        T_planet = compute_equilibrium_temperature(
            stellar_luminosity_logL=pdata["star_lum"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=0.5  # dayside-only
        )


    # Handle surfaces
    if args.surface == "all":
        surfaces = get_surfaces()
    else:
        surfaces = [args.surface]

    # Handle atmospheres
    if args.atmosphere == "all":
        atmospheres = get_atmospheres()
    elif args.atmosphere == "list":
        atmospheres = toml.load(os.path.join(ROOT, "..", "atmos_list.toml")).get("atmospheres", [])
    else:
        atmospheres = [args.atmosphere]

    for surface in surfaces:
        if len(atmospheres) > 1 and not flux_mode:
            fluxes = {}
            wavelengths = None

            for atmo in atmospheres:
                # Write config
                write_agni_config(args.planet, atmo, surface, T_planet)
                config_dir = os.path.join(CONFIG["config_dir"], f"{args.planet}_{surface}_{atmo}".lower())
                config_file = os.path.join(config_dir, "config.toml")

                # Run AGNI
                print(f"Running AGNI for {surface}, {atmo}")
                subprocess.run(["julia", "AGNI/agni.jl", config_file])

                # Load output
                nc_path = os.path.join(CONFIG["output_dir"], args.planet, surface, atmo, "atm.nc")
                if os.path.isfile(nc_path):
                    data = load_agni_output(nc_path)
                    if wavelengths is None:
                        wavelengths = data["bandcenter"]
                    fluxes[atmo] = data["ba_U_total"]

            if fluxes:
                plot_contrasts_multi_atmosphere(
                    flux_dict=fluxes,
                    wavelength_nm=wavelengths,
                    observed_df=contrast_data,
                    T_planet=T_planet,
                    T_star=T_star,
                    R_planet_rearth=R_planet,
                    R_star_rsun=R_star,
                    planet_name=args.planet,
                    surface=surface
                )
        else:
            for atmo in atmospheres:
                write_agni_config(args.planet, atmo, surface, T_planet)
                config_dir = os.path.join(CONFIG["config_dir"], f"{args.planet}_{surface}_{atmo}".lower())
                config_file = os.path.join(config_dir, "config.toml")

                print(f"Running AGNI for {surface}, {atmo}")
                subprocess.run(["julia", "AGNI/agni.jl", config_file])

                nc_path = os.path.join(CONFIG["output_dir"], args.planet, surface, atmo,"atm.nc")
                if os.path.isfile(nc_path):
                    data = load_agni_output(nc_path)
                    plot_bandflux_and_contrast(
                        wavelength_nm=data["bandcenter"],
                        planet_flux_lw=data["ba_U_LW"],
                        planet_flux_sw=data["ba_U_SW"],
                        T_planet=T_planet,
                        T_star=T_star,
                        R_planet_rearth=R_planet,
                        R_star_rsun=R_star,
                        observed_df=contrast_data,
                        planet_name=args.planet,
                        surface=surface,
                        atmosphere_key=atmo
                    )
                else:
                    print(f"[SKIP] No output found: {nc_path}")


if __name__ == "__main__":
    main()
