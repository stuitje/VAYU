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

# paths
def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]
CONFIG = {k: resolve_path(v) for k, v in RAW_CONFIG.items()}
SURFACE_LIST_PATH = os.path.join(ROOT, "misc", "surfaces.toml")

# style
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
    """Load surface labels and highlight list from TOML."""
    if not os.path.isfile(SURFACE_LIST_PATH):
        raise FileNotFoundError(f"Could not find surface list at {SURFACE_LIST_PATH}")
    data = toml.load(SURFACE_LIST_PATH)
    surface_labels = data["surfaces"]
    highlight_surfaces = data.get("highlights", {}).get("surfaces", [])
    return surface_labels, highlight_surfaces

def safe_math_label(label):
    """Format label for LaTeX math bold."""
    return label.replace('-', r'\text{-}').replace(' ', r'\ ')

def plot_surface_contrasts(
    planet,
    surface_keys,
    surface_labels,
    atmo_key="bare_rock",
    highlight_surfaces=None,
    contrast_color_ref="basalt_large"
):
    
    pdata = get_planet_data(planet)
    T_star = pdata["star_temp"]
    R_star = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth

    # Stellar spectrum path 
    star_name = planet[:-1]
    star_path = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}.txt")
    star_path_SPHINX = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    if os.path.exists(star_path_SPHINX):
        star_path = star_path_SPHINX

    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{planet}_data.csv")
    observed_df = load_contrast_data(contrast_path)

    if contrast_color_ref not in surface_keys:
        raise ValueError(f"Reference surface '{contrast_color_ref}' not found in surface list.")

    bayes_csv = os.path.join(CONFIG["output_dir"], planet, "bayes_model_comparison.csv")
    bayes_df = pd.read_csv(bayes_csv)

    # ΔlnZ per surface
    delta_lnZ = {
        surface: float(row["ΔlnZ"].values[0])
        for surface in surface_keys
        if not bayes_df[(bayes_df["surface"] == surface) & (bayes_df["atmosphere"] == atmo_key)].empty
        for row in [bayes_df[(bayes_df["surface"] == surface) & (bayes_df["atmosphere"] == atmo_key)]]
    }

    model_contrasts, bandcenters = {}, {}
    for surface in delta_lnZ:
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

    # Restrict to available surfaces
    available_surfaces = set(model_contrasts)
    delta_lnZ = {s: delta_lnZ[s] for s in available_surfaces}

    # Classification by Bayesian evidence
    accepted = [s for s in delta_lnZ if delta_lnZ[s] >= -3]
    rejected = [s for s in delta_lnZ if delta_lnZ[s] < -3]

    # Setup coloring and linestyles
    n = len(delta_lnZ)
    blue_cmap = plt.get_cmap('Blues', n+2)
    red_cmap = plt.get_cmap('Reds', n+6)
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
    linestyle_cycle = cycle(LINESTYLES)
    highlight_surfaces = highlight_surfaces or [contrast_color_ref]

    fig, ax = plt.subplots(figsize=(12, 8))

    all_surfaces = accepted + rejected
    surface_types = {s: ('accepted' if s in accepted else 'rejected') for s in all_surfaces}

    for i, surface in enumerate(all_surfaces):

        is_highlight = surface in highlight_surfaces
        color = blue_cmap(i +4 )[:3] if surface_types[surface] == 'accepted' else red_cmap(i + 2)[:3]
        lw = 3 if is_highlight else 2
        ls = next(linestyle_cycle)
        alpha = 0.9 if is_highlight else 0.7
        zorder = 10 if is_highlight else 3

        chi2_str = ""
        if observed_df is not None:
            chi2 = compute_chi_squared(
                observed_df,
                bandcenters[surface],      # model_wavelength_nm
                model_contrasts[surface],  # model_contrast
            )
            chi2_str = rf" ($\chi^2$={chi2:.2f})"

        if is_highlight:
            label = f"$\\mathbf{{{safe_math_label(surface_labels.get(surface, surface))}}}$" + chi2_str
        else:
            label = surface_labels.get(surface, surface) + chi2_str

        ax.plot(
            bandcenters[surface]/1000,
            model_contrasts[surface],
            color=color, linewidth=lw, linestyle=ls, label=label,
            alpha=alpha, zorder=zorder
        )


    # Blackbody contrast (f = 2/3)
    T_bb = compute_dayside_brightness_temperature(
        stellar_temperature=T_star,
        stellar_radius_rsun=pdata["star_radius"],
        distance_au=pdata["planet_a"],
        bond_albedo=0.0,
        redistribution_factor=2/3
    )
    surface_ref = highlight_surfaces[0] if highlight_surfaces[0] in bandcenters else next(iter(available_surfaces))
    bandcenter_bb = bandcenters[surface_ref]
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

    # Observed data
    if observed_df is not None:
        ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
            fmt="o", color="black",  elinewidth=1.5, capsize=3,
            label="Zhang et al. (2024)", zorder=100
        )

    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(f"{planet.upper()}: surface models")
    ax.set_ylim(50, 1250)
    ax.set_xlim(4, 12)
    ax.grid(alpha=0.3)

    # Put all highlighted surfaces first in the legend
    handles, labels = ax.get_legend_handles_labels()
    highlighted_labels = [
        f"$\\mathbf{{{safe_math_label(surface_labels.get(s, s))}}}$" for s in highlight_surfaces
    ]

    highlighted_handles_labels = []
    other_handles_labels = []
    for h, l in zip(handles, labels):
        if any(l.startswith(hl) for hl in highlighted_labels):
            highlighted_handles_labels.append((h, l))
        else:
            other_handles_labels.append((h, l))
    # Combine so highlights are first
    sorted_handles_labels = highlighted_handles_labels + other_handles_labels
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, fontsize=12, ncol=2, loc = 2)


    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, "surface_contrast_bayes.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot contrast of surfaces for a given planet with Bayesian selection coloring and distinct linestyles.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g., 'gj367b')")
    parser.add_argument("--atmo", default="bare_rock", help="Atmosphere key (default: bare_rock)")
    parser.add_argument("--ref", default="pyrite", help="Reference surface for Delta lnZ coloring")
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
