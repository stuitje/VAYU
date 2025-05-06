import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict
from scipy.interpolate import interp1d

from src.constants import r_earth, r_sun
from src.utils import planck, contrast_ppm
from src.atmosphere_labels import atmosphere_labels
import toml

# Load config paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]
OUTPUT_DIR = CONFIG["output_dir"]


def load_contrast_data(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def compute_chi_squared(
    observed_df: pd.DataFrame,
    model_wavelength_nm: np.ndarray,
    model_contrast: np.ndarray
) -> float:
    interp_model = interp1d(model_wavelength_nm, model_contrast, bounds_error=False, fill_value="extrapolate")
    model_vals = interp_model(observed_df["X"] * 1000)  # micron to nm
    chi2 = np.sum(((observed_df["Y"] - model_vals) / observed_df["ΔY"]) ** 2)
    dof = len(observed_df["Y"]) - 1
    return chi2 / dof


def plot_bandflux_and_contrast(
    wavelength_nm: np.ndarray,
    planet_flux_lw: np.ndarray,
    planet_flux_sw: np.ndarray,
    T_planet: float,
    T_star: float,
    R_planet_rearth: float,
    R_star_rsun: float,
    observed_df: Optional[pd.DataFrame],
    planet_name: str,
    surface: str,
    atmosphere_key: str,
    save_path: Optional[str] = "auto"
):
    R_planet_m = R_planet_rearth * r_earth
    R_star_m = R_star_rsun * r_sun

    flux_total = planet_flux_lw + planet_flux_sw
    wavelength_um = wavelength_nm / 1000

    contrast_model = contrast_ppm(wavelength_nm, T_star, R_planet_m, R_star_m, planet_flux=flux_total)
    contrast_bb = contrast_ppm(wavelength_nm, T_star, R_planet_m, R_star_m, planet_flux=planck(wavelength_nm, T_planet))

    label = atmosphere_labels.get(atmosphere_key, atmosphere_key)
    if observed_df is not None:
        chi2_red = compute_chi_squared(observed_df, wavelength_nm, contrast_model)
        label += f" ($\chi^2$= {chi2_red:.2f})"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(wavelength_um, flux_total, label="Total", color="black")
    ax1.plot(wavelength_um, planet_flux_lw, "--", label="LW", color="royalblue")
    ax1.plot(wavelength_um, planet_flux_sw, ":", label="SW", color="orangered")
    ax1.set_ylabel(r"Flux (W/m$^2$/nm)", fontsize = 13)
    ax1.set_title(f"{planet_name.upper()} — {label} — {surface}", fontsize = 15)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(wavelength_um, contrast_model, label=label, color="crimson")
    ax2.plot(wavelength_um, contrast_bb, "--", label=f"Blackbody ({T_planet:.0f} K)", color="black")

    if observed_df is not None:
        ax2.errorbar(observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
                     fmt="o", color="blue",  ecolor='gray', capsize=2, label="Observed")

    ax2.set_xlabel(r"Wavelength ($\mu$m)", fontsize = 13)
    ax2.set_ylabel("Contrast (ppm)", fontsize = 13)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 200)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if save_path == "auto":
        save_path = os.path.join(OUTPUT_DIR, planet_name, surface, atmosphere_key, "bandflux_and_contrast.png")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_contrasts_multi_atmosphere(
    flux_dict: Dict[str, np.ndarray],
    wavelength_nm: np.ndarray,
    observed_df: Optional[pd.DataFrame],
    T_planet: float,
    T_star: float,
    R_planet_rearth: float,
    R_star_rsun: float,
    planet_name: str,
    surface: str,
    save_path: Optional[str] = "auto"
):
    """
    Plot contrast curves for multiple atmospheres for a single surface.

    Args:
        flux_dict: Dictionary mapping atmosphere keys to total flux (LW + SW) [W/m²/nm]
        wavelength_nm: Wavelengths in nanometers.
        observed_df: DataFrame with observational contrast data (X, Y, ΔY).
        T_planet: Planet surface temperature [K].
        T_star: Stellar effective temperature [K].
        R_planet_rearth: Planet radius in Earth radii.
        R_star_rsun: Star radius in Solar radii.
        planet_name: Name of the planet.
        surface: Name of the surface.
        save_path: Optional file path to save the figure. If "auto", uses AGNI output path.
    """
    R_planet_m = R_planet_rearth * r_earth
    R_star_m = R_star_rsun * r_sun
    wavelength_um = wavelength_nm / 1000

    contrast_bb = contrast_ppm(
        wavelength_nm=wavelength_nm,
        T_star=T_star,
        R_planet_m=R_planet_m,
        R_star_m=R_star_m,
        T_planet=T_planet
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for atmo_key, planet_flux in flux_dict.items():
        contrast = contrast_ppm(
            wavelength_nm=wavelength_nm,
            T_star=T_star,
            R_planet_m=R_planet_m,
            R_star_m=R_star_m,
            planet_flux=planet_flux
        )
        label = atmosphere_labels.get(atmo_key, atmo_key)
        if observed_df is not None:
            chi2_red = compute_chi_squared(observed_df, wavelength_nm, contrast)
            label += f" ($\chi^2$ = {chi2_red:.2f})"
        ax.plot(wavelength_um, contrast, label=label)

    ax.plot(wavelength_um, contrast_bb, "--", color="black", label=f"Blackbody ({T_planet:.0f} K)")

    if observed_df is not None:
        ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
            fmt="o", color="blue", ecolor='gray', capsize=2, label="Observed"
        )

    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize = 13)
    ax.set_ylabel("Contrast (ppm)", fontsize = 13)
    ax.set_xlim(4, 14)
    ax.set_ylim(0, 400)
    ax.set_title(f"{planet_name.upper()} — multiple atmospheres ({surface})", fontsize = 15)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path == "auto":
        save_path = os.path.join(OUTPUT_DIR, planet_name, surface, f"contrast_{surface}.png")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

