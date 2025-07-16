import os
import argparse
import subprocess
import pandas as pd
import toml
import time

from src.config_gen import write_agni_config
from src.temperature_fit import fit_planet_temperature
from src.plots import (
    plot_bandflux_and_contrast,
    plot_contrasts_multi_atmosphere,
    plot_contrasts_multi_surface,  
    plot_surface_albedo,
    plot_multiple_surface_albedos
)
from src.utils import compute_equilibrium_temperature, compute_dayside_brightness_temperature
from src.dataloader import load_agni_output, load_contrast_data, get_planet_data
from src.chi2_table import generate_chi2_table, write_chi2_table

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

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
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run AGNI + plot for planet/surface/atmo setup.")
    parser.add_argument("planet")
    parser.add_argument("-s", "--surface", required=True, help="'all', 'list', or surface name")
    parser.add_argument("-a", "--atmosphere", required=True, help="'all', 'list', or atmosphere name")
    parser.add_argument("-T", choices=["dayside", "substellar", "full", "half", "more", "less"], default="dayside", help="Temperature model")
    parser.add_argument("--no-run", action="store_true", help="Skip config + AGNI run and just process existing output.")
    parser.add_argument("--flux-only", action="store_true", help="Only generate flux plots for individual configs. Skip contrast comparisons.")

    args = parser.parse_args()

    flux_mode = args.flux_only
    if flux_mode:
        print("[INFO] Running in flux-only mode: skipping multi-surface and multi-atmosphere comparison plots.")

    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{args.planet}_data.csv")
    contrast_data = load_contrast_data(contrast_path)

    pdata = get_planet_data(args.planet)
    T_star, R_star, R_planet = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"]

    redis_factors = {
        "substellar": 1.0,
        "full": 1/4,
        "half": 11/24,
        "more": 17/48,
        "less": 27/48,
        "dayside": 2/3
    }
    f_redis = redis_factors[args.T]

    T_planet = compute_dayside_brightness_temperature(T_star, R_star, pdata["planet_a"], 0, f_redis)
    T_planet_day = compute_dayside_brightness_temperature(T_star, R_star, pdata["planet_a"], 0, 2/3)
    T_planet_full = compute_dayside_brightness_temperature(T_star, R_star, pdata["planet_a"], 0, 1/4)

    if args.surface == "all":
        surfaces = get_surfaces()
    elif args.surface == "list":
        surfaces = toml.load(os.path.join(ROOT, "..", "surface_list.toml")).get("surfaces", [])
        plot_multiple_surface_albedos(surfaces, CONFIG["surface_dir"], args.planet)
    else:
        surfaces = [args.surface]

    if args.atmosphere == "all":
        atmospheres = get_atmospheres()
    elif args.atmosphere == "list":
        atmospheres = toml.load(os.path.join(ROOT, "..", "atmos_list.toml")).get("atmospheres", [])
    else:
        atmospheres = [args.atmosphere]

    for surface in surfaces:
        plot_surface_albedo(surface, CONFIG["surface_dir"], args.planet)
        fluxes = {}
        wavelengths = None

        for atmo in atmospheres:
            atmo_file = atmo
            atmo_mode = f"{atmo}_{args.T}" if args.T != "dayside" else atmo

            nc_path = os.path.join(CONFIG["output_dir"], args.planet, surface, atmo_mode, "atm.nc")

            if not args.no_run:
                if os.path.isfile(nc_path):
                    print(f"[SKIP] AGNI output already exists for {surface}, {atmo_mode}. Skipping run.")
                else:
                    write_agni_config(args.planet, atmo_file, surface, T_planet, args.T, atmo_mode=atmo_mode)
                    config_dir = os.path.join(CONFIG["config_dir"], f"{args.planet}_{surface}_{atmo_mode}".lower())
                    config_file = os.path.join(config_dir, "config.toml")
                    print(f"Running AGNI for {surface}, {atmo}, mode={args.T}")
                    subprocess.run(["julia", "AGNI/agni.jl", config_file])
            else:
                print(f"[SKIP] Skipping config and AGNI run for {surface}, {atmo_mode}")

            if os.path.isfile(nc_path):
                data = load_agni_output(nc_path)

                if not flux_mode:
                    if wavelengths is None:
                        wavelengths = data["bandcenter"]
                    fluxes[atmo_mode] = data["ba_U_total"]

                plot_bandflux_and_contrast(
                    wavelength_nm=data["bandcenter"],
                    planet_flux_lw=data["ba_U_LW"],
                    planet_flux_sw=data["ba_U_SW"],
                    T_planet=T_planet_day,
                    T_planet_full=T_planet_full,
                    T_surf=data["tmp_surf"],
                    T_star=T_star,
                    R_planet_rearth=R_planet,
                    R_star_rsun=R_star,
                    observed_df=contrast_data,
                    planet_name=args.planet,
                    surface=surface,
                    atmosphere_key=atmo_mode
                )
            else:
                print(f"[SKIP] No output found: {nc_path}")

        if args.atmosphere == "list" and len(fluxes) > 1 and not flux_mode:
            plot_contrasts_multi_atmosphere(
                flux_dict=fluxes,
                wavelength_nm=wavelengths,
                observed_df=contrast_data,
                T_planet=T_planet_day,
                T_planet_full=T_planet_full,
                T_star=T_star,
                R_planet_rearth=R_planet,
                R_star_rsun=R_star,
                planet_name=args.planet,
                surface=surface
            )

    if args.surface == "list" and len(surfaces) > 1 and len(atmospheres) == 1 and not flux_mode:
        surface_fluxes = {}
        wavelengths = None
        atmo = atmospheres[0]
        atmo_mode = f"{atmo}_{args.T}" if args.T != "dayside" else atmo

        for surface in surfaces:
            nc_path = os.path.join(CONFIG["output_dir"], args.planet, surface, atmo_mode, "atm.nc")
            if os.path.isfile(nc_path):
                data = load_agni_output(nc_path)
                if wavelengths is None:
                    wavelengths = data["bandcenter"]
                surface_fluxes[surface] = data["ba_U_total"]

        if len(surface_fluxes) > 1:
            plot_contrasts_multi_surface(
                surface_flux_dict=surface_fluxes,
                wavelength_nm=wavelengths,
                observed_df=contrast_data,
                T_planet=T_planet_day,
                T_planet_full=T_planet_full,
                T_star=T_star,
                R_planet_rearth=R_planet,
                R_star_rsun=R_star,
                planet_name=args.planet,
                atmosphere_key=atmo_mode
            )

    if contrast_data is not None:
        bare_results, atmo_results = generate_chi2_table(
            output_dir=CONFIG["output_dir"],
            planet=args.planet,
            contrast_data=contrast_data,
            T_star=T_star,
            R_star_rsun=R_star,
            R_planet_rearth=R_planet
        )
        write_chi2_table(
            planet=args.planet,
            output_dir=CONFIG["output_dir"],
            bare_results=bare_results,
            atmo_results=atmo_results
        )

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n[INFO] Total runtime: {elapsed_min:.2f} minutes")

if __name__ == "__main__":
    main()
