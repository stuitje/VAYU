import os
import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

from src.dataloader import load_agni_output, get_planet_data
from src.utils import planck, compute_dayside_brightness_temperature, contrast_ppm
from src import constants as c
from src.emission_miri import get_throughput, compute_snr, integrate_flux, load_stellar_flux

# --- Custom legend handler to overlay grey line on color patch ---
class HandlerLineOnPatch(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        patch_color, line_color = orig_handle
        patch = plt.Rectangle([xdescent, ydescent], width, height,
                              facecolor=patch_color, alpha=0.2, transform=trans)
        line = plt.Line2D([xdescent, xdescent + width],
                          [ydescent + height / 2] * 2,
                          color=line_color, linestyle="-.", transform=trans)
        return [patch, line]

# --- Surface label mapping ---
surface_labels = {
    "albite_dust": "Albite (dust)", "andesite": "Andesite", "basalt_glass": "Basalt glass",
    "basalt_large": "Basalt (large grain)", "basalt_small": "Basalt (small grain)",
    "basalt_tuff": "Basalt tuff", "diorite": "Diorite", "gabbro": "Gabbro",
    "granite": "Granite", "harzburgite": "Harzburgite", "hematite": "Hematite",
    "lherzolite": "Lherzolite", "lunar_anorthosite": "Lunar anorthosite",
    "lunar_marebasalt": "Lunar mare basalt", "magnesium_sulphate": "Magnesium sulfate",
    "mars_basalticshergottite": "Martian basaltic shergottite", "mars_breccia": "Martian breccia",
    "norite": "Norite", "phonolite": "Phonolite", "pyrite": "Pyrite", "rhyolite": "Rhyolite",
   "tephrite": "Tephrite", "tholeiitic_basalt": "Tholeiitic basalt",
   "trachy_basalt": "Trachybasalt", "trachyte": "Trachyte"
}

def load_config():
    root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root, "..", "agni_config.toml")
    config = toml.load(config_path)["paths"]
    os.environ["pandeia_refdata"] = config["pandeia_dir"]
    return config

def load_star_flux(wave_um, config, d_au):
    path = os.path.join(config["stellar_spectra_dir"], "gj367_SPHINX.txt")
    data = np.loadtxt(path, comments="#")
    wl_nm, flux_erg = data[:, 0], data[:, 1]
    flux = np.interp(wave_um, wl_nm / 1000.0, flux_erg) * (1.0 / d_au)**2
    return flux, path

def process_surface(srf, output_dir, planet, atmosphere, wave_um, throughputs, T_planet, star_flux, Rp_m, d_m):
    nc_path = os.path.join(output_dir, planet, srf, atmosphere, "atm.nc")
    if not os.path.exists(nc_path):
        print(f"[MISSING] {nc_path}")
        return None

    data = load_agni_output(nc_path)
    flux_model = np.interp(wave_um, data["bandcenter"] / 1000, data["ba_U_total"] * 1000)
    omega_planet = np.pi * (Rp_m / d_m)**2
    model_flux_earth = flux_model * omega_planet
    bb_flux_earth = planck(wave_um * 1000, T_planet) * omega_planet * 1000

    row = {"Surface": srf}
    for filt, tp in throughputs.items():
        F_model = integrate_flux(wave_um, model_flux_earth, tp)
        F_bb = integrate_flux(wave_um, bb_flux_earth, tp)
        rel = F_model / F_bb if F_bb > 0 else np.nan
        snr = compute_snr(wave_um, model_flux_earth, star_flux, tp)
        row[filt] = rel
        row[f"{filt}_uncert"] = snr["uncertainty"]
    return row

def plot_emission(df, planet, wave_um, T_star, T_planet, Rp_m, Rs_m, throughputs, config):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    custom_handles = []

    # Plot observed contrast bands
    if planet in ["trappist-1b", "trappist-1c"]:
        try:
            obs_path = os.path.join("res", "planetary_data", f"{planet}_data.csv")
            obs_df = pd.read_csv(obs_path)
            obs_points = {}

            if len(obs_df) == 1:
                obs_points["F1500W"] = obs_df.iloc[0]
            elif len(obs_df) >= 2:
                obs_points["F1280W"] = obs_df.iloc[0]
                obs_points["F1500W"] = obs_df.iloc[1]

            for filt, obs_row in obs_points.items():
                obs_contrast_ppm, obs_uncert_ppm = obs_row["Y"], obs_row["ΔY"]
                stellar_data = np.loadtxt(os.path.join(config["stellar_spectra_dir"], "trappist-1_SPHINX.txt"), comments="#")
                stellar_flux_interp = np.interp(wave_um, stellar_data[:, 0] / 1000, stellar_data[:, 1] * 1e-3 * (c.au / Rs_m)**2)

                Fp = planck(wave_um * 1000, T_planet)
                Fp_int = integrate_flux(wave_um, Fp, throughputs[filt])
                Fs_int = integrate_flux(wave_um, stellar_flux_interp, throughputs[filt])
                bb_band = (Fp_int / Fs_int) * (Rp_m / Rs_m)**2 * 1e6
                rel = obs_contrast_ppm / bb_band
                rel_unc = obs_uncert_ppm / bb_band

                # Perform granite-based scaling once after computing rel_unc
                if planet == "trappist-1c" and filt == "F1500W":
                    try:
                        granite_row = df[df["Surface"] == "granite"].iloc[0]
                        model_uncert = granite_row["F1500W_uncert"]
                        scaling_factor = rel_unc / model_uncert
                        print(f"Scaling uncertainties by {scaling_factor:.3f} (granite, F1500W)")
                        for f in ["F1280W", "F1500W"]:
                            df[f"{f}_uncert"] *= scaling_factor
                    except IndexError:
                        print("Granite row not found for scaling.")

                color = "crimson" if filt == "F1500W" else "dodgerblue"

                ax.fill_between([-0.5, len(df) - 0.5], rel - rel_unc, rel + rel_unc, alpha=0.2, color=color, zorder=1)
                ax.axhline(rel, linestyle="-.", color="gray", zorder=2)
                custom_handles.append(((color, "gray"), fr"Observed $\pm \; 1 \sigma$ ({filt})"))


        except Exception as e:
            print(f"Could not overlay observation: {e}")

    # Plot modeled fluxes
    ax.errorbar(x, df["F1500W"], yerr=df["F1500W_uncert"], fmt="o", markersize=4, capsize=2, color="crimson", label="F1500W", zorder=3)
    ax.errorbar(x, df["F1280W"], yerr=df["F1280W_uncert"], fmt="o", markersize=4, capsize=2, color="dodgerblue", label="F1280W", zorder=3)

    ax.axhline(1.0, linestyle="--", color="black", zorder=0)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Surface"].map(surface_labels).fillna(df["Surface"]), rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Emission relative to blackbody", fontsize=12)
    ax.set_title(f"{planet.upper()}: JWST MIRI emission, surface models", fontsize=17)
    ax.set_ylim(0.4, 1.1)
    ax.grid(True, alpha=0.3)

    # Final legend
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in custom_handles:
        handles.append(handle)
        labels.append(label)

    ax.legend(handles, labels, handler_map={tuple: HandlerLineOnPatch()}, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    out_dir = os.path.join("out", planet, "emissions")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "surface_emissions_only.pdf"), format = 'pdf', dpi=300)
    print(f"Saved emission plot to {out_dir}")

def main():
    planet, atmosphere = "gj367b", "bare_rock"
    wave_um = np.linspace(10.0, 20.0, 1000)
    config = load_config()

    pdata = get_planet_data(planet)
    T_star, R_star, R_planet, d_pc, a_au = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"], pdata["planet_d"], pdata["planet_a"]
    d_m = d_pc * c.pc
    d_au = d_m / c.au
    Rp_m = R_planet * c.r_earth
    Rs_m = R_star * c.r_sun

    T_planet = compute_dayside_brightness_temperature(T_star, R_star, a_au, 0, 2 / 3)
    print(f"Tplanet: {T_planet:.2f} K")

    throughputs = {
        "F1280W": get_throughput(wave_um, "f1280w"),
        "F1500W": get_throughput(wave_um, "f1500w")
    }


    path = os.path.join(config["stellar_spectra_dir"], "gj367_SPHINX.txt")
    star_flux = load_stellar_flux(path, wave_um)
    star_flux *= (1.0 / d_au)**2  # to earth distance
   


    results = []
    for srf in surface_labels.keys():
        row = process_surface(srf, "out", planet, atmosphere, wave_um, throughputs, T_planet, star_flux, Rp_m, d_m)
        if row:
            results.append(row)

    df = pd.DataFrame(results).sort_values("F1280W")

    if planet != "trappist-1c":
        scale = 3.125# From Trappist-1c observation 
        print(f"Scaling uncertainties by {scale}")
        for f in ["F1280W", "F1500W"]:
            df[f"{f}_uncert"] *= scale 

    plot_emission(df, planet, wave_um, T_star, T_planet, Rp_m, Rs_m, throughputs, config)

if __name__ == "__main__":
    main()
