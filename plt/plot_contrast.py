"""
Plot atmospheric or surface model contrast spectra for a given planet.

By default, the models are colored by their Bayesian consistency with the observed data.

Plotted atmospheres are the top-3 best and bottom-3 worst at a given pressure level (if --pressure is specified),
or all with delta lnZ > -3 otherwise.  

Plotted surfaces are all by default, with "hematite" highlighted as an example of a good fit to the data, but this can be overridden.

Usage:
    python plt/plot_contrast.py --planet gj486b --mode atmo
    python plt/plot_contrast.py --planet gj486b --mode surface

    python plt/plot_contrast.py --planet gj367b --mode atmo --pressure 100
    python plt/plot_contrast.py --planet gj367b --mode surface --highlights pyrite --keys hematite pyrite tephrite magnesium_sulphate albite_dust


"""

import os
import sys
import logging
import argparse
from itertools import cycle
from typing import Iterable, Optional, Tuple

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
from src.atmosphere_labels import atmosphere_labels as ATMOSPHERE_LABELS
from src.surface_labels import surface_labels as SURFACE_LABELS

# Paths and configuration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


RAW_CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]
CONFIG     = {k: resolve_path(v) for k, v in RAW_CONFIG.items()}

# Per-planet citation for the observed-data points
DATA_CITATION = {
    "gj486b":      "Weiner Mansfield et al. (2024)",
    "gj367b":      "Zhang et al. (2024)",
    "trappist-1b": "Greene et al. (2023)",
    "trappist-1c": "Zieba et al. (2023)",
}

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


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


# Helpers

def get_star_spectrum_path(planet: str) -> str:
    star_name   = planet[:-1]
    plain_path  = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}.txt")
    sphinx_path = os.path.join(ROOT, "res", "stellar_spectra", f"{star_name}_SPHINX.txt")
    return sphinx_path if os.path.exists(sphinx_path) else plain_path


def errorbar_with_faded_errors(ax, *args, error_alpha=1.0, **kwargs):
    container = ax.errorbar(*args, **kwargs)
    for bar in container[2]:
        bar.set_alpha(error_alpha)
    return container


# Data loading

def load_bayes_delta_lnZ(planet: str, mode: str, fixed_key: str, keys: Iterable[str]) -> dict:
    """Read precomputed delta lnZ values from the Bayesian comparison CSV.

    Parameters
    ----------
    mode : 'atmo' or 'surface'
        In 'atmo' mode the surface column is fixed and atmospheres vary;
        in 'surface' mode the atmosphere column is fixed and surfaces vary.
    fixed_key : str
        The fixed column value (e.g. 'greybody' for atmo, 'bare_rock' for surface).
    """
    csv_path = os.path.join(CONFIG["output_dir"], planet, "bayes_model_comparison.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Bayes comparison CSV not found at {csv_path}. "
            f"Run nested_sampling.py first."
        )
    df = pd.read_csv(csv_path)

    if mode == "atmo":
        fixed_col, varying_col = "surface", "atmosphere"
    elif mode == "surface":
        fixed_col, varying_col = "atmosphere", "surface"
    else:
        raise ValueError(f"mode must be 'atmo' or 'surface', got {mode!r}")

    mask = (df[fixed_col] == fixed_key) & (df[varying_col].isin(keys))
    filtered = df[mask]
    # Robust numeric coercion: handles strings, NaN/inf written by older CSV runs
    delta_vals = pd.to_numeric(filtered["\u0394lnZ"], errors="coerce")
    return dict(zip(filtered[varying_col], delta_vals))


def select_atmospheres_by_pressure(planet: str, surface_key: str, pressure_filter: str,
                                   max_delta_lnZ_floor: float = -30.0) -> Tuple[list, list]:
    """Pick top-3 best and bottom-3 worst atmospheres at a given pressure level.

    Returns (atmo_keys_to_plot, highlights). The highlight is the single best model.
    """
    csv_path = os.path.join(CONFIG["output_dir"], planet, "bayes_model_comparison.csv")
    bayes_df = pd.read_csv(csv_path)
    bayes_df["\u0394lnZ"] = pd.to_numeric(bayes_df["\u0394lnZ"], errors="coerce")
    bayes_df["logZ"]      = pd.to_numeric(bayes_df["logZ"],      errors="coerce")

    prefixes = [f"{p}bar_" for p in pressure_filter.split("_")]
    mask = (
        (bayes_df["surface"] == surface_key)
        & (bayes_df["\u0394lnZ"] > max_delta_lnZ_floor)
        & (bayes_df["atmosphere"].apply(lambda x: any(x.startswith(p) for p in prefixes)))
    )
    matched = bayes_df[mask].sort_values("logZ", ascending=False)
    if matched.empty:
        raise RuntimeError(
            f"No atmospheres at pressure {pressure_filter} bar with "
            f"delta lnZ > {max_delta_lnZ_floor}"
        )

    top3    = matched.head(3)["atmosphere"].tolist()
    bottom3 = matched.tail(3)["atmosphere"].tolist()
    return top3 + bottom3, [top3[0]]


def load_model_contrasts(planet: str, mode: str, fixed_key: str, varying_keys: Iterable[str],
                         T_star, R_planet, R_star, star_path):
    """Return {key: contrast_array} and the shared bandcenter array."""
    contrasts, bandcenter = {}, None
    for key in varying_keys:
        if mode == "atmo":
            surface, atmosphere = fixed_key, key
        else:
            surface, atmosphere = key, fixed_key

        nc_path = os.path.join(CONFIG["output_dir"], planet, surface, atmosphere, "atm.nc")
        if not os.path.isfile(nc_path):
            logging.warning(f"Missing AGNI output: {nc_path}")
            continue
        data = load_agni_output(nc_path)
        if bandcenter is None:
            bandcenter = data["bandcenter"]
        contrasts[key] = contrast_ppm(
            wavelength_nm=bandcenter,
            T_star=T_star, R_planet_m=R_planet, R_star_m=R_star,
            planet_flux=data["ba_U_total"], stellar_spectrum=star_path,
        )
    return contrasts, bandcenter


# Plotting

def _categorise(delta_lnZ: dict,
                accept_threshold: float,
                intermediate_threshold: Optional[float]) -> Tuple[list, list, list]:
    """Split keys into accepted/intermediate/rejected by delta lnZ."""
    if intermediate_threshold is None:
        accepted = [k for k, v in delta_lnZ.items() if v >= accept_threshold]
        rejected = [k for k, v in delta_lnZ.items() if v <  accept_threshold]
        return accepted, [], rejected
    accepted     = [k for k, v in delta_lnZ.items() if v >= intermediate_threshold]
    intermediate = [k for k, v in delta_lnZ.items() if accept_threshold <= v < intermediate_threshold]
    rejected     = [k for k, v in delta_lnZ.items() if v < accept_threshold]
    return accepted, intermediate, rejected


def plot_contrasts(
    planet: str,
    mode: str,
    keys: Iterable[str],
    labels: dict,
    fixed_key: Optional[str] = None,
    highlights: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Tuple[float, float] = (4, 12),
    accept_threshold: float = -3.0,
    intermediate_threshold: Optional[float] = None,
    save_filename: Optional[str] = None,
):
    """Plot AGNI contrast spectra colored by Bayesian consistency with the observed data.

    Parameters
    ----------
    mode : {'atmo', 'surface'}
        'atmo' varies atmospheric composition at a fixed surface (greybody by default).
        'surface' varies surface composition at a fixed atmosphere (bare_rock by default).
    keys : list of str
        Atmosphere (or surface) identifiers to plot.
    fixed_key : str, optional
        Fixed surface (atmo mode) or fixed atmosphere (surface mode).
    intermediate_threshold : float, optional
        If set, models with accept_threshold <= delta lnZ < intermediate_threshold
        are shown in a third (yellow) category.
    """
    if mode == "atmo":
        fixed_key = fixed_key or "greybody"
    elif mode == "surface":
        fixed_key = fixed_key or "bare_rock"
    else:
        raise ValueError(f"mode must be 'atmo' or 'surface', got {mode!r}")

    keys = list(keys)
    highlights = set(highlights or [])

    pdata    = get_planet_data(planet)
    T_star   = pdata["star_temp"]
    R_star   = pdata["star_radius"] * r_sun
    R_planet = pdata["planet_radius"] * r_earth
    star_path = get_star_spectrum_path(planet)

    observed_df = load_contrast_data(os.path.join(CONFIG["obs_data_dir"], f"{planet}_data.csv"))
    data_citation = DATA_CITATION.get(planet, "Observed")

    delta_lnZ = load_bayes_delta_lnZ(planet, mode, fixed_key, keys)
    contrasts, bandcenter = load_model_contrasts(
        planet, mode, fixed_key, delta_lnZ.keys(),
        T_star, R_planet, R_star, star_path,
    )
    delta_lnZ = {k: delta_lnZ[k] for k in contrasts}

    if not contrasts:
        raise RuntimeError(f"No AGNI outputs found for {planet}, mode={mode}, fixed={fixed_key}")

    accepted, intermediate, rejected = _categorise(delta_lnZ, accept_threshold, intermediate_threshold)

    fig, ax = plt.subplots(figsize=(12, 6))
    bold_font   = FontProperties(weight="bold",   size=14)
    normal_font = FontProperties(weight="normal", size=14)
    legend_handles, legend_labels, legend_fonts = [], [], []

    blue_cmap   = plt.get_cmap("Blues",  len(accepted)     + 2)
    yellow_cmap = plt.get_cmap("YlOrBr", len(intermediate) + 4)
    red_cmap    = plt.get_cmap("Reds",   len(rejected)     + 2)
    linestyle_cycle = cycle(LINESTYLES)

    def _plot_one(key, color):
        is_highlight = key in highlights
        chi2_str = ""
        if observed_df is not None:
            chi2 = compute_chi_squared(observed_df, bandcenter, contrasts[key])
            chi2_str = rf" ($\chi^2$={chi2:.2f})"
        label = labels.get(key, key) + chi2_str
        line, = ax.plot(
            bandcenter / 1000, contrasts[key],
            color=color, linewidth=3 if is_highlight else 2,
            linestyle=next(linestyle_cycle),
            alpha=0.8 if is_highlight else 0.6,
            zorder=10 if is_highlight else 3, label=label,
        )
        legend_handles.append(line)
        legend_labels.append(label)
        legend_fonts.append(bold_font if is_highlight else normal_font)

    for i, key in enumerate(accepted):
        _plot_one(key, blue_cmap(i + 2))
    for i, key in enumerate(intermediate):
        _plot_one(key, yellow_cmap(i + 1))
    for i, key in enumerate(rejected):
        _plot_one(key, red_cmap(i + 1))

    # Blackbody reference at f = 2/3
    T_bb = compute_dayside_brightness_temperature(
        stellar_temperature=T_star, stellar_radius_rsun=pdata["star_radius"],
        distance_au=pdata["planet_a"], bond_albedo=0.0, redistribution_factor=2 / 3,
    )
    bb_contrast = contrast_ppm(
        wavelength_nm=bandcenter, T_star=T_star,
        R_planet_m=R_planet, R_star_m=R_star,
        planet_flux=planck(bandcenter, T_bb), stellar_spectrum=star_path,
    )
    bb_label = f"Blackbody ($f=2/3$, $T$={T_bb:.0f} K)"
    bb_line, = ax.plot(bandcenter / 1000, bb_contrast,
                       color="black", linestyle="--", linewidth=2,
                       label=bb_label, zorder=12)
    bb_handle, bb_handle_label = bb_line, bb_label

    obs_handle, obs_handle_label = None, None
    if observed_df is not None:
        obs_handle = errorbar_with_faded_errors(
            ax,
            observed_df["X"], observed_df["Y"],
            yerr=observed_df["\u0394Y"], xerr=observed_df["\u0394X"],
            fmt="o", color="black", elinewidth=1.5, capsize=3,
            label=data_citation, zorder=100,
        )
        obs_handle_label = data_citation

    ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    ax.set_ylabel("Contrast (ppm)")
    ax.set_title(title or f"{planet.upper()}: {mode} models")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.grid(alpha=0.3)

    main_legend = Legend(ax, legend_handles, legend_labels, ncol=1,
                         loc="upper left", frameon=False)
    for text_obj, font in zip(main_legend.get_texts(), legend_fonts):
        text_obj.set_fontproperties(font)
    ax.add_artist(main_legend)

    ref_handles = [bb_handle] + ([obs_handle] if obs_handle is not None else [])
    ref_labels  = [bb_handle_label] + ([obs_handle_label] if obs_handle_label is not None else [])
    ref_legend = Legend(ax, ref_handles, ref_labels, ncol=1,
                        loc="upper right", frameon=False)
    for text_obj in ref_legend.get_texts():
        text_obj.set_fontproperties(normal_font)
    ax.add_artist(ref_legend)

    if save_filename is None:
        save_filename = f"{mode}_contrast_bayes.pdf"
    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], planet, save_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    logging.info(f"Plot saved to {save_path}")
    return fig, ax


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot atmosphere or surface contrast spectra.")
    parser.add_argument("--planet",    required=True, help="Planet name (e.g. 'gj486b')")
    parser.add_argument("--mode",      choices=["atmo", "surface"], default="atmo")
    parser.add_argument("--fixed-key", default=None,
                        help="Fixed surface (atmo mode) or atmosphere (surface mode). "
                             "Defaults: 'greybody' for atmo, 'bare_rock' for surface.")
    parser.add_argument("--highlights", nargs="+", default=None,
                        help="Model keys to highlight (bold lines).")
    parser.add_argument("--pressure", default=None,
                        help="Atmo mode only: filter to a pressure level (e.g. '100' for 100bar_...) "
                             "and plot top-3 + bottom-3 by logZ.")
    parser.add_argument("--keys", nargs="+", default=None,
                        help="Explicit list of atmosphere/surface keys to plot.")
    parser.add_argument("--title",    default=None)
    parser.add_argument("--ylim",     nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"))
    parser.add_argument("--xlim",     nargs=2, type=float, default=(4.0, 12.0), metavar=("XMIN", "XMAX"))
    parser.add_argument("--intermediate-threshold", type=float, default=None,
                        help="Enable 3-category coloring with this delta lnZ cutoff.")
    parser.add_argument("--save-name", default=None, help="Output PDF filename.")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.mode == "atmo":
        labels = ATMOSPHERE_LABELS
        if args.pressure:
            keys, default_highlights = select_atmospheres_by_pressure(
                args.planet, args.fixed_key or "greybody", args.pressure,
            )
        else:
            keys = args.keys or list(labels.keys())
            default_highlights = []
    else:  # surface
        labels = SURFACE_LABELS
        # greybody is a reference placeholder, not a real rock surface
        default_keys = [k for k in SURFACE_LABELS if k != "greybody"]
        keys = args.keys or default_keys
        default_highlights = ["hematite"]

    plot_contrasts(
        planet=args.planet,
        mode=args.mode,
        keys=keys,
        labels=labels,
        fixed_key=args.fixed_key,
        highlights=args.highlights or default_highlights,
        title=args.title,
        ylim=tuple(args.ylim) if args.ylim else None,
        xlim=tuple(args.xlim),
        intermediate_threshold=args.intermediate_threshold,
        save_filename=args.save_name,
    )