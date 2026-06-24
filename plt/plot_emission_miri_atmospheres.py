"""
JWST MIRI predicted emission vs surface pressure for GJ 367b.

Plots relative emission (model / blackbody) for pure CO2, SO2, H2O, CH4
atmospheres at five pressure levels, in both F1280W and F1500W bands.
A grey band shows the range of plausible bare-rock surface emissions,
bounded by the lowest (magnesium sulphate) and highest (trachyandesite)
emitters in the GJ 367b surface grid.

Uncertainty calibration follows the approach in emission_miri.py:
pandeia-predicted SNR is multiplied by an empirical scaling factor
derived from the TRAPPIST-1c F1500W observation by Zieba et al. (2023).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import load_agni_output, get_planet_data
from src.utils import planck, compute_dayside_brightness_temperature
from src.constants import au, r_earth, pc, r_sun
from src.emission_miri import get_throughput, load_stellar_flux, compute_snr, integrate_flux


# Config 

ROOT   = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]
os.environ["pandeia_refdata"] = CONFIG["pandeia_dir"]

PLANET   = "gj367b"
SURFACE  = "greybody"
OUT_DIR  = os.path.join("out", PLANET, "emissions")

# Atmosphere types and pressure levels
ATM_TYPES = ["CO2", "SO2", "H2O", "CH4"]
PRESSURES = {"001": 0.01, "01": 0.1, "1": 1.0, "10": 10.0, "100": 100.0}

# GJ 367b surface emission extremes
SURFACE_LOW  = "magnesium_sulphate"   # highest albedo -> lowest emission
SURFACE_HIGH = "trachy_andesite"      # lowest albedo  -> highest emission

# Uncertainty scaling factor calibrated on TRAPPIST-1c Zieba et al. (2023)
SCALING_FACTOR = 3.125

# Plot aesthetics
GAS_LABELS = {
    "CO2": r"CO$_2$",
    "SO2": r"SO$_2$",
    "H2O": r"H$_2$O",
    "CH4": r"CH$_4$",
}
GAS_COLORS = {
    "CO2": "crimson",
    "SO2": "dodgerblue",
    "H2O": "orange",
    "CH4": "seagreen",
}


#  System parameters

pdata    = get_planet_data(PLANET)
T_star   = pdata["star_temp"]
R_star   = pdata["star_radius"]
R_planet = pdata["planet_radius"]
d_pc     = pdata["planet_d"]
a_au     = pdata["planet_a"]

d_m  = d_pc * pc
d_au = d_m / au
Rp_m = R_planet * r_earth
Rs_m = R_star * r_sun

T_planet = compute_dayside_brightness_temperature(T_star, R_star, a_au, 0.0, 2 / 3)
print(f"[INFO] T_planet (GJ 367b, f=2/3): {T_planet:.2f} K")

# Stellar flux at Earth distance
starspec_path = os.path.join(CONFIG["stellar_spectra_dir"], "gj367_SPHINX.txt")
waves_um  = np.linspace(10.0, 20.0, 1000)
star_flux = load_stellar_flux(starspec_path, waves_um) * (1.0 / d_au) ** 2

# MIRI filter throughputs
throughputs = {
    "F1280W": get_throughput(waves_um, "f1280w"),
    "F1500W": get_throughput(waves_um, "f1500w"),
}

# Precompute shared quantities
omega_planet  = np.pi * (Rp_m / d_m) ** 2
bb_flux_earth = planck(waves_um * 1000, T_planet) * omega_planet * 1000   # W/m²/μm


#  Helpers

def load_surface_emission(nc_path):
    """Load AGNI output and return per-filter (relative emission, uncertainty)."""
    data = load_agni_output(nc_path)
    flux_um = np.interp(waves_um,
                        data["bandcenter"] / 1000,
                        data["ba_U_total"] * 1000) * omega_planet
    out = {}
    for filt, tp in throughputs.items():
        bb_int    = integrate_flux(waves_um, bb_flux_earth, tp)
        model_int = integrate_flux(waves_um, flux_um, tp)
        rel       = model_int / bb_int if bb_int > 0 else np.nan
        uncert    = compute_snr(waves_um, flux_um, star_flux, tp)["uncertainty"] * SCALING_FACTOR
        out[filt] = (rel, uncert)
    return out


def load_atmo_row(nc_path, p_val):
    """Return a results dict for one (atmosphere, pressure) combination."""
    data     = load_agni_output(nc_path)
    flux_um  = np.interp(waves_um,
                         data["bandcenter"] / 1000,
                         data["ba_U_total"] * 1000) * omega_planet
    row = {"Pressure": p_val}
    for filt, tp in throughputs.items():
        bb_int    = integrate_flux(waves_um, bb_flux_earth, tp)
        model_int = integrate_flux(waves_um, flux_um, tp)
        snr_dict  = compute_snr(waves_um, flux_um, star_flux, tp)
        row[f"{filt}_value"]  = model_int / bb_int if bb_int > 0 else np.nan
        row[f"{filt}_uncert"] = snr_dict["uncertainty"] * SCALING_FACTOR
    return row


# Plot

fig, ax = plt.subplots(figsize=(13, 6))

# - Surface emission range (grey band) -
nc_low  = os.path.join(CONFIG["output_dir"], PLANET, SURFACE_LOW,  "bare_rock", "atm.nc")
nc_high = os.path.join(CONFIG["output_dir"], PLANET, SURFACE_HIGH, "bare_rock", "atm.nc")

if os.path.isfile(nc_low) and os.path.isfile(nc_high):
    em_low  = load_surface_emission(nc_low)
    em_high = load_surface_emission(nc_high)

    # Single combined band: min lower bound and max upper bound across both filters
    all_lows  = [em_low[f][0]  - em_low[f][1]  for f in throughputs]
    all_highs = [em_high[f][0] + em_high[f][1] for f in throughputs]

    ax.fill_between(
        [6e-3, 180],
        min(all_lows),
        max(all_highs),
        color="gray", alpha=0.15, linewidth=0,
        label="Surface emission range",
    )
    print(f"[INFO] Surface range: {SURFACE_LOW} (low) to {SURFACE_HIGH} (high)")
else:
    print(f"[WARNING] Missing surface reference files:\n  {nc_low}\n  {nc_high}")

# - Atmosphere curves -
for atmo_type in ATM_TYPES:
    results = []
    for p_label, p_val in PRESSURES.items():
        atmo_name = f"{p_label}bar_{atmo_type}"
        nc_path   = os.path.join(CONFIG["output_dir"], PLANET, SURFACE, atmo_name, "atm.nc")
        if not os.path.isfile(nc_path):
            print(f"[MISSING] {nc_path}")
            continue
        results.append(load_atmo_row(nc_path, p_val))

    if not results:
        continue

    df    = pd.DataFrame(results).sort_values("Pressure")
    label = GAS_LABELS.get(atmo_type, atmo_type)
    color = GAS_COLORS.get(atmo_type, "black")

    # F1280W — filled markers, solid line
    ax.errorbar(
        df["Pressure"], df["F1280W_value"], yerr=df["F1280W_uncert"],
        fmt="-o", capsize=3, markersize=6,
        color=color, alpha=0.8,
        label=f"{label} F1280W",
    )
    # F1500W — open markers, dashed line
    ax.errorbar(
        df["Pressure"], df["F1500W_value"], yerr=df["F1500W_uncert"],
        linestyle="--", marker="o", capsize=3, markersize=6,
        markerfacecolor="white", markeredgecolor=color, color=color,
        label=f"{label} F1500W",
    )

# - Reference lines -
ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5,
           label=r"Blackbody ($T_\mathrm{db}$, $f=2/3$)")

# - Axes -
ax.set_xscale("log")
ax.set_xlabel("Surface pressure [bar]", fontsize=12)
ax.set_ylabel("Emission relative to blackbody", fontsize=12)
gas_str = ", ".join(GAS_LABELS[g] for g in ATM_TYPES)
ax.set_title(f"{PLANET.upper()}: predicted MIRI emission, pure {gas_str} atmospheres",
             fontsize=14)
ax.set_ylim(0.3, 1.4)
ax.set_xlim(6e-3, 180)
ax.legend(ncol=5, fontsize=11, frameon=False)
ax.grid(True, which="both", linestyle=":", alpha=0.5)

# - Save -
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "emission_vs_pressure_multiple_gases.pdf")
plt.tight_layout()
plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
print(f"[INFO] Plot saved to {os.path.abspath(out_path)}")
plt.show()