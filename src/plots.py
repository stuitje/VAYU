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
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 400)
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
    ax.set_xlim(4, 20)
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


def plot_contrasts_multi_surface(
    surface_flux_dict: Dict[str, np.ndarray],
    wavelength_nm: np.ndarray,
    observed_df: Optional[pd.DataFrame],
    T_planet: float,
    T_star: float,
    R_planet_rearth: float,
    R_star_rsun: float,
    planet_name: str,
    atmosphere_key: str,
    save_path: Optional[str] = "auto"
):
    """
    Plot contrast curves for multiple surfaces for a single atmosphere.

    Args:
        surface_flux_dict: Dictionary mapping surface names to total flux (LW + SW) [W/m²/nm]
        wavelength_nm: Wavelengths in nanometers.
        observed_df: DataFrame with observational contrast data (X, Y, ΔY).
        T_planet: Planet surface temperature [K].
        T_star: Stellar effective temperature [K].
        R_planet_rearth: Planet radius in Earth radii.
        R_star_rsun: Star radius in Solar radii.
        planet_name: Name of the planet.
        atmosphere_key: Name of the atmosphere.
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

    for surface_name, planet_flux in surface_flux_dict.items():
        contrast = contrast_ppm(
            wavelength_nm=wavelength_nm,
            T_star=T_star,
            R_planet_m=R_planet_m,
            R_star_m=R_star_m,
            planet_flux=planet_flux
        )
        label = f"{surface_name}"
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

    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=13)
    ax.set_ylabel("Contrast (ppm)", fontsize=13)
    ax.set_xlim(4, 20)
    ax.set_ylim(0, 400)
    ax.set_title(f"{planet_name.upper()} — multiple bare-rock surfaces", fontsize=15)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path == "auto":
        save_path = os.path.join(OUTPUT_DIR, planet_name, "contrast_multi_surface.png")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_surface_albedo(surface_name: str, surface_dir: str, planet_name: str):
    """
    Plot the raw surface albedo.

    Args:
        surface_name: Name of the surface (without extension).
        surface_dir: Directory where .dat files are stored.
        planet_name: Name of the planet for output path.
    """

    save_path = os.path.join(CONFIG["output_dir"], planet_name, surface_name, "surface_albedo.png")
    filepath = os.path.join(surface_dir, f"{surface_name}.dat")

    if os.path.isfile(save_path):
        print(f"[SKIP] Albedo plot already exists: {save_path}")
        return
    
    if not os.path.isfile(filepath):
        print(f"[WARNING] Albedo file not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath, sep=r"\s+", names=["wavelength_nm", "albedo"], comment="#", engine="python")
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return

    wavelength_um = df["wavelength_nm"] / 1000

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelength_um, df["albedo"],  color="dodgerblue")
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=13)
    ax.set_ylabel("Albedo", fontsize=13)
    ax.set_xlim(wavelength_um.min(), wavelength_um.max())
    ax.set_ylim(0, 1)
    ax.set_title(f"{surface_name} — Surface Albedo", fontsize=15)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Albedo plot saved to {save_path}")

def plot_multiple_surface_albedos(surface_names, surface_dir: str, planet_name: str):
    """
    Plot multiple surface albedo curves on one figure.
    """

    save_path = os.path.join(CONFIG["output_dir"], planet_name, "multiple_surfaces_albedo.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    for surface in surface_names:
        path = os.path.join(surface_dir, f"{surface}.dat")
        if not os.path.isfile(path):
            print(f"[WARN] Missing albedo file: {surface}")
            continue
        try:
            df = pd.read_csv(path, sep=r"\s+", names=["wavelength_nm", "albedo"], comment="#", engine="python")
            ax.plot(df["wavelength_nm"] / 1000, df["albedo"], label=surface)
        except Exception as e:
            print(f"[ERROR] Could not read {surface}: {e}")

    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=13)
    ax.set_ylabel("Albedo", fontsize=13)
    ax.set_title(f"{planet_name.upper()} — Surface Albedos", fontsize=15)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, ncol=2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Multiple-surfaces albedo plot saved to {save_path}")

