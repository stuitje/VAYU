import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toml

from fractions import Fraction
from src.dataloader import load_agni_output, get_planet_data
from src.utils import planck, compute_dayside_brightness_temperature
from src.constants import au, r_earth, r_sun, pc
from src.emission_miri import get_throughput, load_stellar_flux, compute_snr, integrate_flux

# Set up paths
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]
os.environ["pandeia_refdata"] = CONFIG["pandeia_dir"]

planet = "gj367b"
surface = "greybody"
atmo_types = ["CO2", "SO2", "H2O", "CH4"]
out_dir = os.path.join("out", planet, "emissions")
waves_um = np.linspace(10.0, 20.0, 1000)

# Throughputs
throughputs = {
    "F1280W": get_throughput(waves_um, "f1280w"),
    "F1500W": get_throughput(waves_um, "f1500w")
}

pdata = get_planet_data(planet)
T_star, R_star, R_planet = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"]
d_pc = pdata["planet_d"]
d_m = d_pc * pc
d_au = d_m / au 
a_au = pdata["planet_a"]
Rp_m = R_planet * r_earth
Rs_m = R_star * r_sun

T_planet = compute_dayside_brightness_temperature(T_star, R_star, a_au, 0, 2/3)

starspec_path = os.path.join(CONFIG["stellar_spectra_dir"], "gj367_SPHINX.txt")
star_flux = load_stellar_flux(starspec_path, waves_um) * (1.0 / d_au)**2

# Target pressures and redistribution modes
pressure_redistribution = {
    0.01: (2/3, "dayside"),
    0.1: (2/3, "dayside"),
    1: (27/48, "less"),
    10: (17/48, "more"),
    100: (1/4, "full")
}

scaling_factor = 3.125 # From Trappist-1c observation 

gas_labels = {
    "CO2": r"CO$_2$",
    "SO2": r"SO$_2$",
    "H2O": r"H$_2$O",
    "CH4": r"CH$_4$"
}

colors = {
    "CO2": "crimson",
    "SO2": "dodgerblue",
    "H2O": "orange",
    "CH4": "seagreen"
}

fig, ax = plt.subplots(figsize=(13, 6))

# Reference surface models: albite and trachyte
ref_nc_path_1 = os.path.join(CONFIG["output_dir"], planet, "magnesium_sulphate", "bare_rock", "atm.nc")
ref_nc_path_2 = os.path.join(CONFIG["output_dir"], planet, "basalt_small", "bare_rock", "atm.nc")

if os.path.isfile(ref_nc_path_1) and os.path.isfile(ref_nc_path_2):
    data_1 = load_agni_output(ref_nc_path_1)
    data_2 = load_agni_output(ref_nc_path_2)

    # Albite
    model_flux_um_1 = data_1["ba_U_total"] * 1000
    wl_model_um_1 = data_1["bandcenter"] / 1000
    interp_model_1 = np.interp(waves_um, wl_model_um_1, model_flux_um_1)

    # Trachyte
    model_flux_um_2 = data_2["ba_U_total"] * 1000
    wl_model_um_2 = data_2["bandcenter"] / 1000
    interp_model_2 = np.interp(waves_um, wl_model_um_2, model_flux_um_2)

    omega_planet = np.pi * (Rp_m / d_m)**2
    interp_model_earth_1 = interp_model_1 * omega_planet
    interp_model_earth_2 = interp_model_2 * omega_planet

    bb_flux_earth = planck(waves_um * 1000, T_planet) * omega_planet * 1000

    # Integrate flux
    model_int_1 = integrate_flux(waves_um, interp_model_earth_1, throughputs["F1280W"])
    model_int_2 = integrate_flux(waves_um, interp_model_earth_2, throughputs["F1500W"]) #f1500w is a bit higher
    bb_int_1 = integrate_flux(waves_um, bb_flux_earth, throughputs["F1280W"])
    bb_int_2 = integrate_flux(waves_um, bb_flux_earth, throughputs["F1500W"])

    albite_emission = model_int_1 / bb_int_1 if bb_int_1 > 0 else np.nan
    trachyte_emission = model_int_2 / bb_int_2 if bb_int_2 > 0 else np.nan


    snr_albite = compute_snr(waves_um, interp_model_earth_1, star_flux, throughputs["F1280W"])
    snr_trachyte = compute_snr(waves_um, interp_model_earth_2, star_flux, throughputs["F1280W"])

    albite_uncert = snr_albite["uncertainty"] * scaling_factor if snr_albite["snr"] > 0 else np.nan
    trachyte_uncert = snr_trachyte["uncertainty"] * scaling_factor if snr_trachyte["snr"] > 0 else np.nan

    x_fill = np.logspace(-2.5, 3.0, 100)

    ax.fill_between(
        x_fill,
        albite_emission - albite_uncert,
        trachyte_emission + trachyte_uncert,
        color="gray",
        alpha=0.2,
        linewidth=0,
        label="Possible surface emissions"
    )
else:
    print(f"[WARNING] Missing albite_dust or trachyte: {ref_nc_path_1} or {ref_nc_path_2}")


for atmo_type in atmo_types:
    df_rows = []
    for p_val, (f_factor, mode_name) in pressure_redistribution.items():
        T_planet = compute_dayside_brightness_temperature(T_star, R_star, a_au, 0, 2/3)
        p_label = str(p_val).replace('.', '').replace('0', '', 1)
        atmo_name = f"{p_label}bar_{atmo_type}_{mode_name}" if mode_name != "dayside" else f"{p_label}bar_{atmo_type}"
        nc_path = os.path.join(CONFIG["output_dir"], planet, surface, atmo_name, "atm.nc")
        print(f"Retrieving {nc_path}")

        if not os.path.isfile(nc_path):
            print(f"[MISSING] {nc_path}")
            continue

        data = load_agni_output(nc_path)
        model_flux_um = data["ba_U_total"] * 1000
        wl_model_um = data["bandcenter"] / 1000
        interp_model = np.interp(waves_um, wl_model_um, model_flux_um)
        omega_planet = np.pi * (Rp_m / d_m)**2
        interp_model_earth = interp_model * omega_planet
        bb_flux_earth = planck(waves_um * 1000, T_planet) * omega_planet * 1000

        row = {"Pressure": p_val, "f_factor": f_factor}
        for filt in throughputs:
            tp = throughputs[filt]
            model_int = integrate_flux(waves_um, interp_model_earth, tp)
            bb_int = integrate_flux(waves_um, bb_flux_earth, tp)
            rel_emission = model_int / bb_int if bb_int > 0 else np.nan
            snr_result = compute_snr(waves_um, interp_model_earth, star_flux, tp)
            row[f"{filt}_value"] = rel_emission
            row[f"{filt}_uncert"] = snr_result["uncertainty"] * scaling_factor
        df_rows.append(row)

    df = pd.DataFrame(df_rows).sort_values("Pressure")
    label_base = gas_labels.get(atmo_type, atmo_type)
    color = colors.get(atmo_type, "black")

    for filt, style in zip(["F1280W", "F1500W"], ['-o', '--o']):
        ax.errorbar(df["Pressure"], df[f"{filt}_value"], yerr=df[f"{filt}_uncert"], fmt=style, capsize=3,
                    markersize=6, color=color, markerfacecolor='white' if filt == "F1500W" else color,
                    label=f"{label_base} {filt}", alpha=0.8)
        print(df[f"{filt}_value"])

ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5, label=r"Blackbody ($T_\text{db}$)")
ax.set_xscale("log")

# Custom tick labels including f-factors as fractions
pressure_ticks = [0.01, 0.1, 1, 10, 100]
pressure_labels = [f"{p}\n(f={Fraction(pressure_redistribution[p][0]).limit_denominator()})" for p in pressure_ticks]
ax.set_xticks(pressure_ticks)
ax.set_xticklabels(pressure_labels)

ax.set_xlabel("Surface pressure [bar]", fontsize=12)
ax.set_ylabel("Emission relative to blackbody", fontsize=12)
ax.set_title(f"{planet.upper()}: MIRI emission for pure atmospheres, with heat redistribution", fontsize=15)
ax.set_ylim(0.0, 1.2)
ax.set_xlim(0.006, 180)
ax.legend(ncol=5, fontsize=11)
ax.grid(True, which="both", linestyle=":", alpha=0.5)

os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, f"emission_vs_pressure_modes_multiple_gases.pdf"), format = 'pdf', dpi=300)
plt.show()
