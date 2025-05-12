import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import planck, compute_equilibrium_temperature
from src.dataloader import load_agni_output
from src.throughput import get_throughput  
from src.dataloader import load_contrast_data, get_planet_data
from src.temperature_fit import fit_planet_temperature
import toml

# Define paths 
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

# Integrate over throughput 
def integrate_flux(wavelength_um, flux, throughput):
    """Integrate flux multiplied by throughput over wavelength."""
    return np.trapz(flux * throughput, wavelength_um)

# Find relative emission w.r.t. blackbody 
def compute_relative_emissions(nc_path, T_planet, wave_um, throughputs):
    """Compute relative emissions (model / blackbody) for multiple filters."""

    data = load_agni_output(nc_path)

    model_flux = data["ba_U_total"]
    wl_model = data["bandcenter"] / 1000  # to microns

    # Interpolate model flux to match throughput wavelength grid
    interp_model = np.interp(wave_um, wl_model, model_flux)

    wave_nm = wave_um * 1000  # to nm for planck
    bb_flux = planck(wave_nm, T_planet)

    results = {}
    for filt, throughput in throughputs.items():
        model_int = integrate_flux(wave_um, interp_model, throughput)
        bb_int = integrate_flux(wave_um, bb_flux, throughput)
        result = model_int / bb_int if bb_int > 0 else np.nan
        results[filt] = result

    return results

# main script 
def main():
    planet = "gj486b"
    surface = "greybody"  
    atmosphere = "bare_rock"
    output_dir = "out"
    surface_dir = "res/surfaces"
    atmosphere_dir = "res/atmospheres"

    pdata = get_planet_data(planet)
    T_star, R_star, R_planet = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"]

    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{planet}_data.csv")
    contrast_data = load_contrast_data(contrast_path)

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
            redistribution_factor=0.5
        )

    # Get surfaces 
    surfaces = [f.replace(".dat", "") for f in os.listdir(surface_dir) if f.endswith(".dat")]

    wave_um = np.linspace(10.0, 20.0, 1000)
    throughputs = {
        "F1280W": get_throughput(wave_um, "f1280w"),
        "F1500W": get_throughput(wave_um, "f1500w")
    }

    # Surface emission
    results_surface = []
    for srf in surfaces:
        nc_path = os.path.join(output_dir, planet, srf, atmosphere, "atm.nc")
        if os.path.exists(nc_path):
            emissions = compute_relative_emissions(nc_path, T_planet, wave_um, throughputs)
            results_surface.append({"Surface": srf, **emissions})
        else:
            print(f"[WARNING] Missing: {nc_path}")

    df_surface = pd.DataFrame(results_surface).sort_values("F1280W")

    # Atmosphere emission
    atmospheres = [f.replace(".toml", "") for f in os.listdir(atmosphere_dir) if f.endswith(".toml")]
    results_atmo = []
    for atmo in atmospheres:
        if "O2" in atmo and not ("SO2" in atmo or "CO2" in atmo):
            print(f"[SKIPPED] Skipping {atmo} due to O2 without SO2 or CO2")
            continue

        nc_path = os.path.join(output_dir, planet, surface, atmo, "atm.nc")
        if os.path.exists(nc_path):
            emissions = compute_relative_emissions(nc_path, T_planet, wave_um, throughputs)
            results_atmo.append({"Atmosphere": atmo, **emissions})
        else:
            print(f"[WARNING] Missing: {nc_path}")

    df_atmo = pd.DataFrame(results_atmo).sort_values("F1280W")

    # Plotting 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True, gridspec_kw={'wspace': 0.05})

    # Surface plot
    x1 = np.arange(len(df_surface))
    ax1.axhline(1.0, linestyle="--", color="black")
    ax1.plot(x1, df_surface["F1280W"], "x", label="F1280W", color="dodgerblue")
    ax1.plot(x1, df_surface["F1500W"], "o", label="F1500W", color="crimson")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(df_surface["Surface"], rotation=45, ha="right")
    ax1.set_title("Surface emission", fontsize=16)
    ax1.set_ylabel("Emission relative to blackbody", fontsize=12)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Atmosphere plot
    x2 = np.arange(len(df_atmo))
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.plot(x2, df_atmo["F1280W"], "x", label="F1280W", color="dodgerblue")
    ax2.plot(x2, df_atmo["F1500W"], "o", label="F1500W", color="crimson")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(df_atmo["Atmosphere"], rotation=45, ha="right")
    ax2.set_title("Atmosphere emission", fontsize=16)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(bottom=0.35, wspace=0.05)

    # Save
    out_dir = os.path.join("out", planet)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relative_emission_{atmosphere}_and_atmo.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()