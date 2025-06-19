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
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.utils import contrast_ppm, planck, compute_dayside_brightness_temperature
from src.constants import r_earth, r_sun
from src.stat import compute_chi_squared
from src.atmosphere_labels import atmosphere_labels as atmo_labels_full

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

COLORS = cycle([
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
])


def get_atmosphere_labels_and_highlights():
    highlight_atmospheres = []  # Optional: load from a file if desired
    return atmo_labels_full, highlight_atmospheres

def plot_atmosphere_contrasts(
    planet,
    atmo_keys,
    atmo_labels,
    surface_key="greybody",
    highlight_atmospheres=None,
    contrast_color_ref="hematite",
    pressure_filter=None
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

    # Pressure filtering
    if pressure_filter:
        pressure_prefix = f"{pressure_filter}bar_"
        filtered_df = bayes_df[
            (bayes_df["surface"] == surface_key) &
            (bayes_df["atmosphere"].str.startswith(pressure_prefix))
        ].copy()
        filtered_df.sort_values("logZ", ascending=False, inplace=True)
        top3 = filtered_df.head(3)["atmosphere"].tolist()
        bottom3 = filtered_df.tail(3)["atmosphere"].tolist()
        atmo_keys = top3 + bottom3
        highlight_atmospheres = [top3[0]]  # Best overall gets bold

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

    accepted = [a for a in delta_lnZ if delta_lnZ[a] >= -2.3]
    rejected = [a for a in delta_lnZ if delta_lnZ[a] < -2.3]
    intermediate = [a for a in delta_lnZ if -2.3 <= delta_lnZ[a] < -1.2]

    blue_cmap = plt.get_cmap('Blues', len(accepted)+1)
    red_cmap = plt.get_cmap('Reds', len(rejected)+10)
    yellow_cmap = plt.get_cmap('YlOrRd', len(rejected)+10)
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyle_cycle = cycle(LINESTYLES)
    highlight_atmospheres = highlight_atmospheres or [contrast_color_ref]

    fig, ax = plt.subplots(figsize=(12, 6))
    bold_font = FontProperties(weight='bold', size=14)
    normal_font = FontProperties(weight='normal', size=14)
    legend_handles, legend_labels, legend_fonts = [], [], []

    all_atmospheres = accepted + rejected
    model_types = {a: ('accepted' if a in accepted else 'rejected') for a in all_atmospheres}

    for i, atmo in enumerate(all_atmospheres):
        is_highlight = atmo in highlight_atmospheres
        color = (
            blue_cmap(i + 1) if model_types[atmo] == 'accepted' 
            else red_cmap(i + 1) if model_types[atmo] == 'rejected' 
            else yellow_cmap(i + 1)
        )
        lw = 3 if is_highlight else 2
        ls = next(linestyle_cycle)
        alpha = 0.9 if is_highlight else 0.7
        zorder = 10 if is_highlight else 3

        chi2_str = ""
        if observed_df is not None and planet != 'trappist-1c':
            chi2 = compute_chi_squared(observed_df, bandcenters[atmo], model_contrasts[atmo])
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

    # Add blackbody reference model
    if bandcenters:
        T_bb = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2/3
        )
        ref_atmo = highlight_atmospheres[0] if highlight_atmospheres[0] in bandcenters else next(iter(bandcenters))
        bandcenter_bb = bandcenters[ref_atmo]
        bb_contrast = contrast_ppm(
            wavelength_nm=bandcenter_bb, T_star=T_star,
            R_planet_m=R_planet, R_star_m=R_star,
            planet_flux=planck(bandcenter_bb, T_bb),
            stellar_spectrum=star_path
        )
        bb_line, = ax.plot(
            bandcenter_bb / 1000, bb_contrast,
            color="black", linestyle="--", linewidth=2,
            label=f"Blackbody (f=2/3, $T$={T_bb:.0f} K)", zorder=12
        )
        legend_handles.append(bb_line)
        legend_labels.append(f"Blackbody (f=2/3, $T$={T_bb:.0f} K)")
        legend_fonts.append(normal_font)

    if observed_df is not None:
        obs = ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label="Zhang et al. (2024)", zorder=100
        )
        legend_handles.append(obs)
        legend_labels.append("Zhang et al. (2024)")
        legend_fonts.append(normal_font)

    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(f"{planet.upper()}: 100 bar atmospheres")
    ax.set_ylim(10, 220)
    ax.set_xlim(4, 12)
    ax.grid(alpha=0.3)

    legend = Legend(ax, legend_handles, legend_labels, ncol=2, loc=2, frameon=False)
    for text_obj, font in zip(legend.get_texts(), legend_fonts):
        text_obj.set_fontproperties(font)
    ax.add_artist(legend)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, "atmo_contrast_bayes.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_surface_contrasts(
    planet,
    surface_keys,
    surface_labels,
    atmosphere_key="bare_rock",
    highlight_surfaces=None,
    contrast_color_ref="hematite"
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
        surface: float(row["ΔlnZ"].values[0])
        for surface in surface_keys
        if not bayes_df[(bayes_df["surface"] == surface) & (bayes_df["atmosphere"] == atmosphere_key)].empty
        for row in [bayes_df[(bayes_df["surface"] == surface) & (bayes_df["atmosphere"] == atmosphere_key)]]
    }

    model_contrasts, bandcenters = {}, {}
    for surface in delta_lnZ:
        nc_path = os.path.join(CONFIG["output_dir"], planet, surface, atmosphere_key, "atm.nc")
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

    available_surfaces = set(model_contrasts)
    delta_lnZ = {s: delta_lnZ[s] for s in available_surfaces}

    accepted = [s for s in delta_lnZ if delta_lnZ[s] >= -2.3]
    rejected = [s for s in delta_lnZ if delta_lnZ[s] < -2.3]

    blue_cmap = plt.get_cmap('Blues', len(accepted)+2)
    red_cmap = plt.get_cmap('Reds', len(rejected)+10)
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyle_cycle = cycle(LINESTYLES)
    highlight_surfaces = highlight_surfaces or [contrast_color_ref]

    fig, ax = plt.subplots(figsize=(12, 6))
    bold_font = FontProperties(weight='bold', size=14)
    normal_font = FontProperties(weight='normal', size=14)
    legend_handles, legend_labels, legend_fonts = [], [], []

    all_surfaces = accepted + rejected
    model_types = {s: ('accepted' if s in accepted else 'rejected') for s in all_surfaces}

    for i, surface in enumerate(all_surfaces):
        is_highlight = surface in highlight_surfaces
        color = next(COLORS) # blue_cmap(i+1) if model_types[surface] == 'accepted' else red_cmap(i+1) 
        lw = 3 if is_highlight else 2
        ls = next(linestyle_cycle)
        alpha = 0.9 if is_highlight else 0.7
        zorder = 10 if is_highlight else 3

        chi2_str = ""
        if observed_df is not None and planet != 'trappist-1c':
            chi2 = compute_chi_squared(observed_df, bandcenters[surface], model_contrasts[surface])
            chi2_str = rf" ($\chi^2$={chi2:.2f})"

        label_text = surface_labels.get(surface, surface) + chi2_str

        line, = ax.plot(
            bandcenters[surface]/1000,
            model_contrasts[surface],
            color=color, linewidth=lw, linestyle=ls, label=label_text,
            alpha=alpha, zorder=zorder
        )

        legend_handles.append(line)
        legend_labels.append(label_text)
        legend_fonts.append(bold_font if is_highlight else normal_font)

     # Add blackbody reference model
    if bandcenters:
        T_bb = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2/3
        )
        ref_surf = highlight_surfaces[0] if highlight_surfaces and highlight_surfaces[0] in bandcenters else next(iter(bandcenters))
        bandcenter_bb = bandcenters[ref_surf]
        bb_contrast = contrast_ppm(
            wavelength_nm=bandcenter_bb, T_star=T_star,
            R_planet_m=R_planet, R_star_m=R_star,
            planet_flux=planck(bandcenter_bb, T_bb),
            stellar_spectrum=star_path
        )
        bb_line, = ax.plot(
            bandcenter_bb / 1000, bb_contrast,
            color="black", linestyle="--", linewidth=2,
            label=f"Blackbody (f=2/3, $T$={T_bb:.0f} K)", zorder=12
        )
        legend_handles.append(bb_line)
        legend_labels.append(f"Blackbody (f=2/3, $T$={T_bb:.0f} K)")
        legend_fonts.append(normal_font)

    if observed_df is not None:
        obs = ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"], xerr = observed_df["ΔX"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label="Zieba et al. (2024)", zorder=100
        )
        legend_handles.append(obs)
        legend_labels.append("Zieba et al. (2024)")
        legend_fonts.append(normal_font)

    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(f"{planet.upper()}: surface models")
    ax.set_ylim(0, 800)
    ax.set_xlim(4, 20)
    ax.grid(alpha=0.3)

    legend = Legend(ax, legend_handles, legend_labels, ncol=2, loc=2, frameon=False)
    for text_obj, font in zip(legend.get_texts(), legend_fonts):
        text_obj.set_fontproperties(font)
    ax.add_artist(legend)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, "surface_contrast_bayes.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot contrast of atmospheric or surface models for a given planet.")
    parser.add_argument("--planet", required=True, help="Planet name (e.g., 'gj367b')")
    parser.add_argument("--mode", choices=["atmo", "surface"], default="atmo", help="Plot mode: 'atmo' or 'surface'")
    parser.add_argument("--surface", default="greybody", help="Surface key to fix (for atmosphere mode)")
    parser.add_argument("--ref", default="hematite", help="Reference model to highlight")
    parser.add_argument("--pressure", default=None, help="Atmosphere pressure filter (e.g. '001' for 001bar_...)")

    
    args = parser.parse_args()

    if args.mode == "atmo":
        atmo_labels, highlight_atmospheres_toml = get_atmosphere_labels_and_highlights()
        atmo_keys = list(atmo_labels.keys())
        highlight_atmospheres = highlight_atmospheres_toml or [args.ref]

        plot_atmosphere_contrasts(
            planet=args.planet,
            atmo_keys=atmo_keys,
            atmo_labels=atmo_labels,
            surface_key=args.surface,
            highlight_atmospheres=highlight_atmospheres,
            contrast_color_ref=args.ref,
            pressure_filter=args.pressure
        )
    else:
        surface_labels = {
            "ultramafic": "Peridodite",
            "diorite": "Diorite",
            "granite": "Granite",
            "harzburgite": "Harzburgite",
            "albite_dust": "Albite (dust)",
            "trachyte": "Trachyte",
            "magnesium_suplhate": "Magnesium sulfate",
            "trachy_andesite": "Trachyandesite" 
        }
        surface_keys = list(surface_labels.keys())

        plot_surface_contrasts(
            planet=args.planet,
            surface_keys=surface_keys,
            surface_labels=surface_labels,
            atmosphere_key="bare_rock",
            highlight_surfaces=[args.ref],
            contrast_color_ref=args.ref
        )
