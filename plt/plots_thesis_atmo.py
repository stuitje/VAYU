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
from src.nested_sampling import compare_models

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
    atmo_labels,
    surface_key="greybody",
    highlight_atmospheres=None,
    contrast_color_ref="hematite",
    reference_surface=None
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

    # 🎯 Use only these models
    surfaces = [surface_key]
    atmospheres = ["001bar_N2_H2O_O3_case3", "100bar_N2_H2O_O3_case3",  "01bar_N2_H2O_O3_case4", "1bar_N2_H2O_O3_case4", "100bar_H2O_O3_case2"]  # Add more if desired

    # Bayesian model comparison
    print(f"[DEBUG] Running Bayesian comparison for atmospheres: {atmospheres}")
    bayes_df = compare_models(
        planet_name=planet,
        surfaces=surfaces,
        atmospheres=atmospheres,
        reference_surface=reference_surface,
        write_to_csv=False
    )

    print(f"[DEBUG] Bayesian comparison returned {len(bayes_df)} results")
    print(f"[DEBUG] Available combinations:")
    for _, row in bayes_df.iterrows():
        print(f"  {row['surface']} + {row['atmosphere']}: ΔlnZ = {row['ΔlnZ']:.2f}")

    # No filtering by keys — only what's explicitly listed
    delta_lnZ = {
        row["atmosphere"]: row["ΔlnZ"]
        for _, row in bayes_df.iterrows()
        if row["surface"] == surface_key and row["atmosphere"] in atmospheres
    }

    print(f"[DEBUG] After filtering for surface '{surface_key}': {list(delta_lnZ.keys())}")

    model_contrasts, bandcenters = {}, {}
    for atmo in delta_lnZ:
        nc_path = os.path.join(CONFIG["output_dir"], planet, surface_key, atmo, "atm.nc")
        print(f"[DEBUG] Checking file: {nc_path}")
        print(f"[DEBUG] File exists: {os.path.isfile(nc_path)}")
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
        print(f"[DEBUG] Successfully loaded: {atmo}")

    print(f"[DEBUG] Final models to plot: {list(model_contrasts.keys())}")

    available_atmospheres = set(model_contrasts)
    delta_lnZ = {a: delta_lnZ[a] for a in available_atmospheres}

    accepted = [a for a in delta_lnZ if delta_lnZ[a] >= -3]
    rejected = [a for a in delta_lnZ if delta_lnZ[a] < -3]

    blue_cmap = plt.get_cmap('Blues', len(accepted)+1)
    red_cmap = plt.get_cmap('Reds', len(rejected)+10)
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyle_cycle = cycle(LINESTYLES)
    highlight_atmospheres = highlight_atmospheres or [contrast_color_ref]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Fonts
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
        alpha = 0.8 if is_highlight else 0.7
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

    # Blackbody reference
    if bandcenters:
        T_bb = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2/3
        )
        ref_atmo = highlight_atmospheres[0] if highlight_atmospheres and highlight_atmospheres[0] in bandcenters else next(iter(bandcenters))
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

    # Observed data
    if observed_df is not None:
        obs = ax.errorbar(
            observed_df["X"], observed_df["Y"], yerr=observed_df["ΔY"], xerr=observed_df["ΔX"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label="Weiner-Mansfield et al. (2024)", zorder=100
        )
        legend_handles.append(obs)
        legend_labels.append("Weiner-Mansfield et al. (2024)")
        legend_fonts.append(normal_font)

    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(rf"{planet.upper()}: N$_2$, H$_2$O and O$_3$ atmospheres")
    ax.set_ylim(10, 400)
    ax.set_xlim(4, 12)
    ax.grid(alpha=0.3)

    # Legend
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
    parser.add_argument("--ref", default="hematite", help="Reference model for coloring")
    parser.add_argument("--bayes_ref", default="hematite", help="Reference surface for Bayesian comparison (e.g., 'greybody' or 'hematite')")
    args = parser.parse_args()

    atmo_labels, highlight_atmospheres_toml = get_atmosphere_labels_and_highlights()
    atmo_keys = list(atmo_labels.keys())
    highlight_atmospheres = highlight_atmospheres_toml or [args.ref]

    plot_atmosphere_contrasts(
        planet=args.planet,
        atmo_labels=atmo_labels,
        surface_key=args.surface,
        highlight_atmospheres=highlight_atmospheres,
        contrast_color_ref=args.ref,
        reference_surface=args.bayes_ref
    )