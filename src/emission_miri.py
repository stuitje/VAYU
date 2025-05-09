import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_agni_output, planck, compute_equilibrium_temperature
from src.throughput import get_throughput  
from src.plots import load_contrast_data
from src.temperature_fit import fit_planet_temperature
import toml

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

def get_planet_data(name: str) -> dict:
    df = pd.read_csv(CONFIG["planet_csv"])
    row = df[df["planet"].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Planet '{name}' not found.")
    return row.iloc[0].to_dict()

def integrate_flux(wavelength_um, flux, throughput):
    """Integrate flux multiplied by throughput over wavelength."""
    return np.trapz(flux * throughput, wavelength_um)

def compute_relative_emission(nc_path, T_planet, wave_um, throughput):
    """Compute relative emission (model / blackbody) integrated through a filter."""
    data = load_agni_output(nc_path)
    model_flux = data["ba_U_total"]
    wl_model = data["bandcenter"] / 1000 # to microns

    # Interpolate model flux to match throughput wavelength grid (microns)
    interp_model = np.interp(wave_um, wl_model, model_flux)

    wave_nm = wave_um * 1000 # to nm for planck
    bb_flux = planck(wave_nm, T_planet)

    model_int = integrate_flux(wave_um, interp_model, throughput)
    bb_int = integrate_flux(wave_um, bb_flux, throughput)

    return model_int / bb_int if bb_int > 0 else np.nan

def main():
    planet = "gj486b"
    atmosphere = "bare_rock"
    output_dir = "out"
    surface_dir = "res/surfaces"

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

    # Get all surfaces from .dat files
    surfaces = [f.replace(".dat", "") for f in os.listdir(surface_dir) if f.endswith(".dat")]

    # JWST MIRI wavelength range in microns
    wave_um = np.linspace(10.0, 20.0, 1000)

    # Load real filter throughputs
    throughput_1280 = get_throughput(wave_um, "f1280w")
    throughput_1500 = get_throughput(wave_um, "f1500w")

    results = []
    for surface in surfaces:
        nc_path = os.path.join(output_dir, planet, surface, atmosphere, "atm.nc")
        if os.path.exists(nc_path):
            rel_1280 = compute_relative_emission(nc_path, T_planet, wave_um, throughput_1280)
            rel_1500 = compute_relative_emission(nc_path, T_planet, wave_um, throughput_1500)
            results.append({"Surface": surface, "F1280W": rel_1280, "F1500W": rel_1500})
        else:
            print(f"[WARNING] Missing: {nc_path}")

    # Convert to DF and sort
    df = pd.DataFrame(results).sort_values("F1280W")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.axhline(1.0, linestyle="--", color="black", label="Blackbody = 1.0")
    ax.plot(x, df["F1280W"], "x", markersize = 4, label="F1280W", color="dodgerblue")
    ax.plot(x, df["F1500W"], "o", markersize = 3, label="F1500W", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Surface"], rotation=45, ha="right")
    ax.set_ylabel("Emission Relative to Blackbody")
    ax.set_title(f"{planet.upper()} — Surface emission")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    out_dir = os.path.join("out", planet)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relative_emission_{atmosphere}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
