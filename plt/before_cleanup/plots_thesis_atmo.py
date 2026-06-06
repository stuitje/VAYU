"""
Plot atmosphere emission contrasts against observed data.

Loads pre-computed Bayes factors from bayes_model_comparison.csv (re-running
nested sampling only if the CSV is missing), computes contrast spectra from
AGNI outputs, and overlays them with the observed JWST spectrum.
"""

import os
import sys
import logging
from itertools import cycle
from typing import Iterable, Optional

import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_agni_output, get_planet_data, load_contrast_data
from src.utils import contrast_ppm, planck, compute_dayside_brightness_temperature
from src.constants import r_earth, r_sun
from src.stat import compute_chi_squared
from src.nested_sampling import compare_models


# ---------- Paths and configuration ----------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


RAW_CONFIG       = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]
CONFIG           = {k: resolve_path(v) for k, v in RAW_CONFIG.items()}
ATMOS_LIST_PATH  = os.path.join(ROOT, "misc", "atmospheres.toml")
BAYES_THRESHOLD  = -3.0   # delta lnZ below this is "rejected"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
})


# ---------- Data loading ----------

def get_atmosphere_labels_and_highlights():
    """Load atmosphere labels and highlight list from TOML."""
    if not os.path.isfile(ATMOS_LIST_PATH):
        raise FileNotFoundError(f"Could not find atmosphere list at {ATMOS_LIST_PATH}")
    data = toml.load(ATMOS_LIST_PATH)
    return data["atmospheres"], data.get("highlights", {}).get("atmospheres", [])


def get_star_spectrum_path(planet: str) -> str:
    star_name   = planet[:-1]
    plain_path  = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}.txt")
    sphinx_path = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    return sphinx_path if os.path.exists(sphinx_path) else plain_path


def load_bayes_results(planet: str, surfaces, atmospheres, reference_surface) -> pd.DataFrame:
    """Load Bayesian comparison results, re-running the sampler only if needed."""
    csv_path = os.path.join(CONFIG["output_dir"], planet, "bayes_model_comparison.csv")

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        needed = set(atmospheres)
        have   = set(df["atmosphere"].unique())
        missing = needed - have
        if not missing:
            logging.info(f"Loaded pre-computed Bayes results from {csv_path}")
            return df
        logging.info(f"Re-running nested sampling: missing {sorted(missing)} from CSV")

    return compare_models(
        planet_name=planet,
        surfaces=surfaces,
        atmospheres=atmospheres,
        reference_surface=reference_surface,
        write_to_csv=False,
    )


def load_atmosphere_models(planet: str, atmospheres: Iterable[str], surface_key: str,
                           T_star, R_planet, R_star, star_path):
    """Return {atmosphere_key: contrast_array} and the shared wavelength array."""
    models = {}
    bandcenter = None
    for atmo in atmospheres:
        nc_path = os.path.join(CONFIG["output_dir"], planet, surface_key, atmo, "atm.nc")
        if not os.path.isfile(nc_path):
            logging.warning(f"Missing AGNI output: {nc_path}")
            continue
        data = load_agni_output(nc_path)
        if bandcenter is None:
            bandcenter = data["bandcenter"]
        models[atmo] = contrast_ppm(
            wavelength_nm=bandcenter,
            T_star=T_star,
            R_planet_m=R_planet,
            R_star_m=R_star,
            planet_flux=data["ba_U_total"],
            stellar_spectrum=star_path,
        )
    return models, bandcenter


# ---------- Plotting ----------

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def plot_atmosphere_contrasts(
    planet: str,
    atmospheres: Iterable[str],
    atmo_labels: dict,
    surface_key: str = "greybody",
    highlight_atmospheres: Optional[Iterable[str]] = None,
    reference_surface: Optional[str] = None,
    title: Optional[str] = None,
    save_filename: str = "atmo_contrast_bayes.png",
):
    """Plot atmospheric contrast spectra grouped by Bayesian consistency with observed data."""

    # System parameters
    pdata    = get_planet_data(planet)
    T_star   = pdata["star_temp"]
    R_star   = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth
    star_path = get_star_spectrum_path(planet)

    # Observations
    observed_df = load_contrast_data(os.path.join(CONFIG["obs_data_dir"], f"{planet}_data.csv"))

    # Bayes factors (loaded from CSV if available)
    atmospheres = list(atmospheres)
    bayes_df = load_bayes_results(planet, [surface_key], atmospheres, reference_surface)
    delta_lnZ = {
        row["atmosphere"]: row["\u0394lnZ"]
        for _, row in bayes_df.iterrows()
        if row["surface"] == surface_key and row["atmosphere"] in atmospheres
    }

    # AGNI model contrasts
    model_contrasts, bandcenter = load_atmosphere_models(
        planet, delta_lnZ.keys(), surface_key, T_star, R_planet, R_star, star_path
    )
    delta_lnZ = {a: delta_lnZ[a] for a in model_contrasts}  # drop atmospheres with no model file

    if not model_contrasts:
        raise RuntimeError(f"No AGNI outputs found for {planet}/{surface_key} matching {atmospheres}")

    # Categorise by Bayes factor
    accepted = [a for a in delta_lnZ if delta_lnZ[a] >= BAYES_THRESHOLD]
    rejected = [a for a in delta_lnZ if delta_lnZ[a] <  BAYES_THRESHOLD]

    highlight_set = set(highlight_atmospheres or [])

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(14, 6))

    bold_font   = FontProperties(weight="bold",   size=14)
    normal_font = FontProperties(weight="normal", size=14)
    legend_handles, legend_labels, legend_fonts = [], [], []

    blue_cmap = plt.get_cmap("Blues", len(accepted) + 2)
    red_cmap  = plt.get_cmap("Reds",  len(rejected) + 2)
    linestyle_cycle = cycle(LINESTYLES)

    def _plot_model(atmo, color, ls):
        is_highlight = atmo in highlight_set
        chi2_str = ""
        if observed_df is not None:
            chi2 = compute_chi_squared(observed_df, bandcenter, model_contrasts[atmo])
            chi2_str = rf" ($\chi^2$={chi2:.2f})"
        label = atmo_labels.get(atmo, atmo) + chi2_str
        line, = ax.plot(
            bandcenter / 1000, model_contrasts[atmo],
            color=color, linewidth=3 if is_highlight else 2,
            linestyle=ls, alpha=0.8 if is_highlight else 0.7,
            zorder=10 if is_highlight else 3, label=label,
        )
        legend_handles.append(line)
        legend_labels.append(label)
        legend_fonts.append(bold_font if is_highlight else normal_font)

    # Per-category indexing avoids the index-overflow bug from before
    for ai, atmo in enumerate(accepted):
        _plot_model(atmo, blue_cmap(ai + 1), next(linestyle_cycle))
    for ri, atmo in enumerate(rejected):
        _plot_model(atmo, red_cmap(ri + 1),  next(linestyle_cycle))

    # Blackbody reference at f = 2/3
    if bandcenter is not None:
        T_bb = compute_dayside_brightness_temperature(
            stellar_temperature=T_star,
            stellar_radius_rsun=pdata["star_radius"],
            distance_au=pdata["planet_a"],
            bond_albedo=0.0,
            redistribution_factor=2 / 3,
        )
        bb_contrast = contrast_ppm(
            wavelength_nm=bandcenter, T_star=T_star,
            R_planet_m=R_planet, R_star_m=R_star,
            planet_flux=planck(bandcenter, T_bb),
            stellar_spectrum=star_path,
        )
        bb_label = f"Blackbody ($f=2/3$, $T$={T_bb:.0f} K)"
        bb_line, = ax.plot(bandcenter / 1000, bb_contrast,
                           color="black", linestyle="--", linewidth=2,
                           label=bb_label, zorder=12)
        legend_handles.append(bb_line)
        legend_labels.append(bb_label)
        legend_fonts.append(normal_font)

    # Observed data
    if observed_df is not None:
        obs_label = "Weiner Mansfield et al. (2024)"
        obs = ax.errorbar(
            observed_df["X"], observed_df["Y"],
            yerr=observed_df["\u0394Y"], xerr=observed_df["\u0394X"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label=obs_label, zorder=100,
        )
        legend_handles.append(obs)
        legend_labels.append(obs_label)
        legend_fonts.append(normal_font)

    # Axes
    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(title or planet.upper())
    ax.set_ylim(10, 400)
    ax.set_xlim(4, 12)
    ax.grid(alpha=0.3)

    legend = Legend(ax, legend_handles, legend_labels, ncol=2, loc="best", frameon=False)
    for text_obj, font in zip(legend.get_texts(), legend_fonts):
        text_obj.set_fontproperties(font)
    ax.add_artist(legend)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, save_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    logging.info(f"Plot saved to {save_path}")
    plt.show()


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot atmospheric model contrasts vs observed JWST data.")
    parser.add_argument("--planet",     required=True, help="Planet name (e.g. 'gj486b')")
    parser.add_argument("--surface",    default="greybody", help="Surface key to fix (default: greybody)")
    parser.add_argument("--bayes-ref",  default="hematite", help="Reference surface for Bayes factors")
    parser.add_argument("--atmospheres", nargs="+",
                        default=[
                            "001bar_N2_H2O_O3_case3",
                            "100bar_N2_H2O_O3_case3",
                            "01bar_N2_H2O_O3_case4",
                            "1bar_N2_H2O_O3_case4",
                            "100bar_H2O_O3_case2",
                        ],
                        help="Atmosphere keys to plot.")
    parser.add_argument("--title",     default=None, help="Custom plot title.")
    parser.add_argument("--save-name", default="atmo_contrast_bayes.png",
                        help="Output filename for the saved figure.")
    parser.add_argument("--verbose",   action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    atmo_labels, highlight_atmospheres = get_atmosphere_labels_and_highlights()
    plot_atmosphere_contrasts(
        planet=args.planet,
        atmospheres=args.atmospheres,
        atmo_labels=atmo_labels,
        surface_key=args.surface,
        highlight_atmospheres=highlight_atmospheres,
        reference_surface=args.bayes_ref,
        title=args.title,
        save_filename=args.save_name,
    )