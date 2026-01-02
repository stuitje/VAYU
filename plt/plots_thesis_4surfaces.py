import sys
import os
import toml
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from itertools import cycle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.utils import contrast_ppm, planck, compute_dayside_brightness_temperature
from src.constants import r_earth, r_sun
from src.stat import compute_chi_squared

# Paths and configuration
def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]
CONFIG = {k: resolve_path(v) for k, v in RAW_CONFIG.items()}
SURFACE_LIST_PATH = os.path.join(ROOT, "misc", "surfaces.toml")

# Matplotlib style
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
})

def get_surface_labels_and_highlights():
    if not os.path.isfile(SURFACE_LIST_PATH):
        raise FileNotFoundError(f"Could not find surface list at {SURFACE_LIST_PATH}")
    data = toml.load(SURFACE_LIST_PATH)
    surface_labels = data["surfaces"]
    highlight_surfaces = data.get("highlights", {}).get("surfaces", [])
    return surface_labels, highlight_surfaces

def safe_math_label(label):
    return label.replace('-', r'\text{-}').replace(' ', r'\ ')

def plot_surface_contrasts(
    planet,
    surface_keys,
    surface_labels,
    atmo_key="bare_rock",
    highlight_surfaces=None,
    contrast_color_ref=None
):
    pdata = get_planet_data(planet)
    T_star = pdata["star_temp"]
    R_star = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth

    star_name = planet[:-1]
    star_path = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}.txt")
    star_path_SPHINX = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    if os.path.exists(star_path_SPHINX):
        star_path = star_path_SPHINX

    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{planet}_data.csv")
    observed_df = load_contrast_data(contrast_path)

    model_contrasts, bandcenters = {}, {}
    for surface in surface_keys:
        nc_path = os.path.join(CONFIG["output_dir"], planet, surface, atmo_key, "atm.nc")
        if not os.path.isfile(nc_path):
            print(f"[SKIP] Missing file: {nc_path}")
            continue
        data = load_agni_output(nc_path)
        bandcenter = data["bandcenter"]
        flux_total = data["ba_U_total"]
        model_contrasts[surface] = contrast_ppm(
            wavelength_nm=bandcenter, T_star=T_star,
            R_planet_m=R_planet, R_star_m=R_star,
            planet_flux=flux_total, stellar_spectrum=star_path
        )
        bandcenters[surface] = bandcenter

    available_surfaces = list(model_contrasts.keys())
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(available_surfaces)))
    color_map = dict(zip(available_surfaces, colors))
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
    linestyle_cycle = cycle(LINESTYLES)
    highlight_surfaces = highlight_surfaces or []

    fig, ax = plt.subplots(figsize=(12, 6))
    linestyle_cycle = cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))])
    highlight_surfaces = highlight_surfaces or []

    for surface in available_surfaces:
        is_highlight = surface in highlight_surfaces
        lw = 3 if is_highlight else 2.5
        ls = next(linestyle_cycle)
        alpha = 0.9 if is_highlight else 0.7
        zorder = 10 if is_highlight else 3

        chi2_str = ""
        if observed_df is not None:
            chi2 = compute_chi_squared(
                observed_df,
                bandcenters[surface],
                model_contrasts[surface],
            )
            chi2_str = rf" ($\chi^2$={chi2:.2f})"

        label_base = surface_labels.get(surface, surface)
        label = f"$\\mathbf{{{safe_math_label(label_base)}}}$" + chi2_str if is_highlight else label_base + chi2_str

        ax.plot(
            bandcenters[surface] / 1000,
            model_contrasts[surface],
            linewidth=lw, linestyle=ls, label=label,
            alpha=alpha, zorder=zorder
        )


    T_bb = compute_dayside_brightness_temperature(
        stellar_temperature=T_star,
        stellar_radius_rsun=pdata["star_radius"],
        distance_au=pdata["planet_a"],
        bond_albedo=0.0,
        redistribution_factor=2/3
    )
    ref_surface = highlight_surfaces[0] if highlight_surfaces and highlight_surfaces[0] in bandcenters else next(iter(bandcenters))
    bandcenter_bb = bandcenters[ref_surface]
    bb_contrast = contrast_ppm(
        wavelength_nm=bandcenter_bb, T_star=T_star,
        R_planet_m=R_planet, R_star_m=R_star,
        planet_flux=planck(bandcenter_bb, T_bb), stellar_spectrum=star_path
    )
    ax.plot(
        bandcenter_bb / 1000, bb_contrast,
        color="black", linestyle="--", linewidth=2.0,
        label=f"Blackbody (f=2/3, $T$={T_bb:.0f} K)", zorder=12
    )

    if observed_df is not None:
        ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label="Zhang et al. (2024)", zorder=100
        )

    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(f"{planet.upper()}: surface models")
    ax.set_ylim(20, 160)
    ax.set_xlim(4, 12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, ncol=2)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, "surface_contrast_unique_colors.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot contrast of surfaces for a given planet using distinct colors per surface.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g., 'gj367b')")
    parser.add_argument("--atmo", default="bare_rock", help="Atmosphere key (default: bare_rock)")
    parser.add_argument("--ref", default="pyrite", help="Reference surface for bandcenter fallback")
    args = parser.parse_args()

    surface_labels, highlight_surfaces_toml = get_surface_labels_and_highlights()
    surface_keys = list(surface_labels.keys())
    highlight_surfaces = highlight_surfaces_toml or [args.ref]

    plot_surface_contrasts(
        planet=args.planet,
        surface_keys=surface_keys,
        surface_labels=surface_labels,
        atmo_key=args.atmo,
        highlight_surfaces=highlight_surfaces,
        contrast_color_ref=args.ref
    )
