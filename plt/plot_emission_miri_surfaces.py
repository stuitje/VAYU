"""
JWST MIRI emission plotter for GJ 367b surface models.

Uncertainty calibration: pandeia-predicted SNR is scaled empirically
to match the TRAPPIST-1c F1500W observation by Zieba et al. (2023),
following the approach in emission_miri.py. The granite bare-rock model
for TRAPPIST-1c is used as the calibration reference.
"""

import os
import sys
import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerBase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_agni_output, get_planet_data
from src.utils import planck, compute_dayside_brightness_temperature
from src import constants as c
from src.emission_miri import get_throughput, compute_snr, integrate_flux, load_stellar_flux
from src.surface_labels import surface_labels


#  Custom legend handler 

class HandlerLineOnPatch(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        patch_color, line_color = orig_handle
        patch = plt.Rectangle([xdescent, ydescent], width, height,
                               facecolor=patch_color, alpha=0.2, transform=trans)
        line = plt.Line2D([xdescent, xdescent + width],
                          [ydescent + height / 2] * 2,
                          color=line_color, linestyle="-.", transform=trans)
        return [patch, line]


# ---------- Config ----------

def load_config():
    root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root, "..", "agni_config.toml")
    config = toml.load(config_path)["paths"]
    os.environ["pandeia_refdata"] = config["pandeia_dir"]
    return config


#  Uncertainty calibration 

def compute_uncertainty_scaling_factor(config, wave_um, throughputs, output_dir,
                                        obs_contrast_ppm=421.0, obs_uncert_ppm=94.0,
                                        ref_surface="granite", ref_atmosphere="bare_rock"):
    """
    Compute the scaling factor to match pandeia uncertainties to the
    TRAPPIST-1c F1500W observation by Zieba et al. (2023).

    The factor is: snr_observed / snr_model, where snr_observed = 421/94 ≈ 4.48.
    Uncertainties should be DIVIDED by this factor (i.e. they are scaled up).

    Returns
    -------
    scaling_factor : float
        Multiply model uncertainty by (1/scaling_factor) to match observation.
    """
    # TRAPPIST-1c parameters
    t1c = get_planet_data("trappist-1c")
    d_m_t1c   = t1c["planet_d"] * c.pc
    d_au_t1c  = d_m_t1c / c.au
    Rp_m_t1c  = t1c["planet_radius"] * c.r_earth
    T_star_t1c = t1c["star_temp"]
    R_star_t1c = t1c["star_radius"]
    a_au_t1c   = t1c["planet_a"]

    T_planet_t1c = compute_dayside_brightness_temperature(
        T_star_t1c, R_star_t1c, a_au_t1c, 0.0, 2 / 3
    )

    # TRAPPIST-1 stellar flux at Earth
    sphinx_path = os.path.join(config["stellar_spectra_dir"], "trappist-1_SPHINX.txt")
    star_flux_t1c = load_stellar_flux(sphinx_path, wave_um)
    star_flux_t1c *= (1.0 / d_au_t1c) ** 2

    # Reference model (granite + bare_rock for TRAPPIST-1c)
    ref_nc = os.path.join(output_dir, "trappist-1c", ref_surface, ref_atmosphere, "atm.nc")
    if not os.path.exists(ref_nc):
        raise FileNotFoundError(
            f"TRAPPIST-1c reference model missing: {ref_nc}\n"
            f"Run AGNI for trappist-1c/{ref_surface}/{ref_atmosphere} first."
        )

    data = load_agni_output(ref_nc)
    flux_model = np.interp(wave_um, data["bandcenter"] / 1000, data["ba_U_total"] * 1000)
    omega_t1c = np.pi * (Rp_m_t1c / d_m_t1c) ** 2
    model_flux_earth = flux_model * omega_t1c

    snr_model = compute_snr(wave_um, model_flux_earth, star_flux_t1c, throughputs["F1500W"])["snr"]
    snr_obs   = obs_contrast_ppm / obs_uncert_ppm

    scaling_factor = snr_obs / snr_model
    print(f"[CALIBRATION] TRAPPIST-1c {ref_surface}+{ref_atmosphere} pandeia SNR: {snr_model:.3f}")
    print(f"[CALIBRATION] Zieba et al. observed SNR: {snr_obs:.3f}")
    print(f"[CALIBRATION] Scaling factor: {scaling_factor:.4f}  (uncertainties divided by this)")
    return scaling_factor


# Surface processing

def process_surface(srf, output_dir, planet, atmosphere, wave_um, throughputs,
                    T_planet, star_flux, Rp_m, d_m):
    nc_path = os.path.join(output_dir, planet, srf, atmosphere, "atm.nc")
    if not os.path.exists(nc_path):
        print(f"[MISSING] {nc_path}")
        return None

    data = load_agni_output(nc_path)
    flux_model      = np.interp(wave_um, data["bandcenter"] / 1000, data["ba_U_total"] * 1000)
    omega_planet    = np.pi * (Rp_m / d_m) ** 2
    model_flux_earth = flux_model * omega_planet
    bb_flux_earth   = planck(wave_um * 1000, T_planet) * omega_planet * 1000

    row = {"Surface": srf}
    for filt, tp in throughputs.items():
        F_model = integrate_flux(wave_um, model_flux_earth, tp)
        F_bb    = integrate_flux(wave_um, bb_flux_earth,    tp)
        rel     = F_model / F_bb if F_bb > 0 else np.nan
        snr     = compute_snr(wave_um, model_flux_earth, star_flux, tp)
        row[filt]               = rel
        row[f"{filt}_uncert"]   = snr["uncertainty"]
    return row


# Plotting

def plot_emission(df, planet, scaling_factor):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))

    ax.errorbar(x, df["F1500W"], yerr=df["F1500W_uncert"] / scaling_factor,
                fmt="o", markersize=4, capsize=2, color="crimson",   label="F1500W", zorder=3)
    ax.errorbar(x, df["F1280W"], yerr=df["F1280W_uncert"] / scaling_factor,
                fmt="o", markersize=4, capsize=2, color="dodgerblue", label="F1280W", zorder=3)

    ax.axhline(1.0, linestyle="--", color="black", zorder=0)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(0.4, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Surface"].map(surface_labels).fillna(df["Surface"]),
        rotation=45, ha="right", fontsize=12
    )
    ax.set_ylabel("Emission relative to blackbody", fontsize=12)
    ax.set_title(f"{planet.upper()}: predicted JWST MIRI emission, surface models", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    out_dir = os.path.join("out", planet, "emissions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "surface_emissions_only.pdf")
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved to {os.path.abspath(out_path)}")
    plt.show()


# Main 

def main():
    planet     = "gj367b"
    atmosphere = "bare_rock"
    output_dir = "out"

    wave_um = np.linspace(10.0, 20.0, 1000)
    config  = load_config()

    # GJ 367b system parameters
    pdata  = get_planet_data(planet)
    T_star   = pdata["star_temp"]
    R_star   = pdata["star_radius"]
    R_planet = pdata["planet_radius"]
    d_pc     = pdata["planet_d"]
    a_au     = pdata["planet_a"]

    d_m  = d_pc * c.pc
    d_au = d_m / c.au
    Rp_m = R_planet * c.r_earth
    Rs_m = R_star * c.r_sun

    T_planet = compute_dayside_brightness_temperature(T_star, R_star, a_au, 0.0, 2 / 3)
    print(f"[INFO] T_planet (GJ 367b, f=2/3): {T_planet:.2f} K")

    throughputs = {
        "F1280W": get_throughput(wave_um, "f1280w"),
        "F1500W": get_throughput(wave_um, "f1500w"),
    }

    # GJ 367b stellar flux at Earth distance
    sphinx_path = os.path.join(config["stellar_spectra_dir"], "gj367_SPHINX.txt")
    star_flux   = load_stellar_flux(sphinx_path, wave_um)
    star_flux  *= (1.0 / d_au) ** 2

    # Uncertainty calibration from TRAPPIST-1c Zieba et al. (2023)
    scaling_factor = compute_uncertainty_scaling_factor(
        config, wave_um, throughputs, output_dir
    )

    # Process all surfaces
    results = []
    for srf in surface_labels:
        if srf == "greybody":
            continue
        row = process_surface(
            srf, output_dir, planet, atmosphere,
            wave_um, throughputs, T_planet, star_flux, Rp_m, d_m
        )
        if row:
            results.append(row)

    df = pd.DataFrame(results).sort_values("F1280W")
    print(f"[INFO] Processed {len(df)} surfaces")
    print(f"[INFO] Calibrated F1500W uncertainty (hematite): "
          f"{df.loc[df['Surface']=='hematite', 'F1500W_uncert'].values[0] / scaling_factor:.4f}")

    plot_emission(df, planet, scaling_factor)


if __name__ == "__main__":
    main()