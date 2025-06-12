import sys
import os
import toml
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from itertools import cycle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.utils import contrast_ppm, planck, compute_dayside_brightness_temperature
from src.constants import r_earth, r_sun
from src.stat import compute_chi_squared

def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]
CONFIG = {k: resolve_path(v) for k, v in RAW_CONFIG.items()}
ATMOS_LIST_PATH = os.path.join(ROOT, "misc", "atmospheres.toml")

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
})

def get_atmosphere_labels_and_highlights():
    """Load atmosphere labels and highlight list from TOML."""
    if not os.path.isfile(ATMOS_LIST_PATH):
        raise FileNotFoundError(f"Could not find atmosphere list at {ATMOS_LIST_PATH}")
    data = toml.load(ATMOS_LIST_PATH)
    atmo_labels = data["atmospheres"]
    highlight_atmospheres = data.get("highlights", {}).get("atmospheres", [])
    return atmo_labels, highlight_atmospheres

def plot_atmosphere_contrasts(
    planet,
    atmo_keys,
    atmo_labels,
    surface_key="greybody",
    highlight_atmospheres=None,
    contrast_color_ref="01bar_N2_1000ppm_H2O"
):
    pdata = get_planet_data(planet)
    T_star = pdata["star_temp"]
    R_star = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth

    star_name = planet[:-1]
    star_path = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}.txt")
    sphinx_path = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    if os.path.exists(sphinx_path):
        star_path = sphinx_path

    contrast_path = os.path.join(CONFIG["obs_data_dir"], f"{planet}_data.csv")
    observed_df = load_contrast_data(contrast_path)

    bayes_csv = os.path.join(CONFIG["output_dir"], planet, "bayes_model_comparison.csv")
    bayes_df = pd.read_csv(bayes_csv)

    delta_lnZ = {
        atmo: float(row["ΔlnZ"].values[0])
        for atmo in atmo_keys
        if not bayes_df[(bayes_df["surface"] == surface_key) & (bayes_df["atmosphere"] == atmo)].empty
        for row in [bayes_df[(bayes_df["surface"] == surface_key) & (bayes_df["atmosphere"] == atmo)]]
    }

    model_contrasts, bandcenters = {}, {}
    for atmo in delta_lnZ:
        nc_path = os.path.join(CONFIG["output_dir"], planet, surface_key, atmo, "atm.nc")
        if not os.path.isfile(nc_path):
            print(f"[SKIP] Missing file: {nc_path}")
            continue
        data = load_agni_output(nc_path)
        bandcenter = data["bandcenter"]
        flux_total = data["ba_U_total"]
        model_contrasts[atmo] = contrast_ppm(
            wavelength_nm=bandcenter, T_star=T_star,
            R_planet_m=R_planet, R_star_m=R_star,
            planet_flux=flux_total, stellar_spectrum=star_path
        )
        bandcenters[atmo] = bandcenter

    available_atmospheres = set(model_contrasts)
    delta_lnZ = {a: delta_lnZ[a] for a in available_atmospheres}

    accepted = [a for a in delta_lnZ if delta_lnZ[a] >= -4]
    rejected = [a for a in delta_lnZ if delta_lnZ[a] < -4]

    blue_cmap = plt.get_cmap('Blues', len(accepted)+2)
    red_cmap = plt.get_cmap('Reds', len(rejected)+10)
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyle_cycle = cycle(LINESTYLES)
    highlight_atmospheres = highlight_atmospheres or [contrast_color_ref]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Font styles
    bold_font = FontProperties(weight='bold', size=14)
    normal_font = FontProperties(weight='normal', size=14)

    legend_handles = []
    legend_labels = []
    legend_fonts = []

    all_atmospheres = accepted + rejected
    model_types = {a: ('accepted' if a in accepted else 'rejected') for a in all_atmospheres}

    for i, atmo in enumerate(all_atmospheres):
        is_highlight = atmo in highlight_atmospheres
        color = blue_cmap(i+1) if model_types[atmo] == 'accepted' else red_cmap(i+1)
        lw = 3 if is_highlight else 2
        ls = next(linestyle_cycle)
        alpha = 0.9 if is_highlight else 0.7
        zorder = 10 if is_highlight else 3

        chi2_str = ""
        if observed_df is not None:
            chi2 = compute_chi_squared(
                observed_df,
                bandcenters[atmo],
                model_contrasts[atmo]
            )
            chi2_str = rf" ($\chi^2$={chi2:.2f})"

        label_text = atmo_labels.get(atmo, atmo) + chi2_str

        line, = ax.plot(
            bandcenters[atmo]/1000,
            model_contrasts[atmo],
            color=color, linewidth=lw, linestyle=ls, label=label_text,
            alpha=alpha, zorder=zorder
        )

        legend_handles.append(line)
        legend_labels.append(label_text)
        legend_fonts.append(bold_font if is_highlight else normal_font)

    # Add observed data
    if observed_df is not None:
        obs = ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label="Weiner-Mansfield (2024)", zorder=100
        )
        legend_handles.append(obs)  # full container, not just obs[0]
        legend_labels.append("Weiner-Mansfield (2024)")
        legend_fonts.append(normal_font)


    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(f"{planet.upper()} atmospheric models")
    ax.set_ylim(10, 400)
    ax.set_xlim(4, 12)
    ax.grid(alpha=0.3)

    # Custom legend with bold highlights
    legend = Legend(ax, legend_handles, legend_labels, ncol=2, loc='best', frameon=False)
    for text_obj, font in zip(legend.get_texts(), legend_fonts):
        text_obj.set_fontproperties(font)
    ax.add_artist(legend)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, "atmo_contrast_bayes.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot contrast of atmospheric models for a given planet.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g., 'gj367b')")
    parser.add_argument("--surface", default="greybody", help="Surface key to fix (default: greybody)")
    parser.add_argument("--ref", default="01bar_N2_1000ppm_H2O", help="Reference atmosphere for coloring (default: 01bar_N2_1000ppm_H2O)")
    args = parser.parse_args()

    atmo_labels, highlight_atmospheres_toml = get_atmosphere_labels_and_highlights()
    atmo_keys = list(atmo_labels.keys())
    highlight_atmospheres = highlight_atmospheres_toml or [args.ref]

    plot_atmosphere_contrasts(
        planet=args.planet,
        atmo_keys=atmo_keys,
        atmo_labels=atmo_labels,
        surface_key=args.surface,
        highlight_atmospheres=highlight_atmospheres,
        contrast_color_ref=args.ref
    )
