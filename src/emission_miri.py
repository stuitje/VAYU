# adapted from https://github.com/rodluger/planetplanet/blob/362003dbe4bb607d5ad7d561f62949cb9553acf4/planetplanet/detect/jwst.py#L35

import os
import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths and config
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]
os.environ["pandeia_refdata"] = CONFIG["pandeia_dir"]

from pandeia.engine.instrument_factory import InstrumentFactory
from src.utils import planck, compute_dayside_brightness_temperature
from src.dataloader import load_agni_output, get_planet_data
from src.temperature_fit import fit_planet_temperature
from src import constants as c
from matplotlib.gridspec import GridSpec
from src.atmosphere_labels import atmosphere_labels

# Load stellar flux
def load_stellar_flux(filepath, target_wave_um):
    data = np.loadtxt(filepath, comments="#")
    wl_nm = data[:, 0]
    flux_erg_cm2_s_nm = data[:, 1]

    # Convert erg/cm^2/s/nm -> W/m^2/micron
    flux_W_m2_um = flux_erg_cm2_s_nm * 1e-3
    wl_um = wl_nm / 1000.0
    return np.interp(target_wave_um, wl_um, flux_W_m2_um)

def get_throughput(wave_um, filter_name):
    conf = {
        "detector": {
            "nexp": 1,
            "ngroup": 10,
            "nint": 1,
            "readout_pattern": "fastr1",
            "subarray": "full"
        },
        "instrument": {
            "aperture": "imager",
            "filter": filter_name,
            "instrument": "miri",
            "mode": "imaging"
        },
    }
    instrument_factory = InstrumentFactory(config=conf)
    return instrument_factory.get_total_eff(wave_um)

def photon_rate(wave_um, flux_wm2um, throughput, atel=c.jwst_collecting_area):
    hc = c.planck_const * c.speed_of_light
    lam_m = wave_um * 1e-6
    dlam = np.gradient(wave_um) * 1e-6
    photons = flux_wm2um * dlam * lam_m / hc * throughput * atel
    return np.sum(photons)

# JWST Background
def jwst_background(wave_um):
    omega = np.pi * (0.42 / 206265. * wave_um / 10.)**2.
    emiss = [4.2e-14, 4.3e-8, 3.35e-7, 9.7e-5, 1.72e-3, 1.48e-2, 1.31e-4]
    temps = [5500., 270., 133.8, 71.0, 62.0, 51.7, 86.7]
    total = np.zeros_like(wave_um)
    for e, T in zip(emiss, temps):
        total += e * planck(wave_um * 1000, T) * omega
    return total

# SNR
def compute_snr(wave_um, model_flux, star_flux, throughput,
                tint_s=36.4*60, nout=4.0, atel=c.jwst_collecting_area,
                thermal=True, n_eclipses=5):
    
    tint_total = tint_s * n_eclipses
    nout_total = nout * n_eclipses

    Np = tint_total * photon_rate(wave_um, model_flux, throughput, atel)
    Ns = tint_total * photon_rate(wave_um, star_flux, throughput, atel)
    Nb = tint_total * photon_rate(wave_um, jwst_background(wave_um), throughput, atel) if thermal else 0.0

    read_noise_e = 6.
    n_pixels = 25
    n_reads = 10 * 1
    detector_noise_squared = (read_noise_e ** 2) * n_pixels * n_reads
    Ndet = 0  # detector_noise_squared can be included if needed

    total_noise = (1 + 1/nout_total)*Ns + (1/nout_total)*Np + (1 + 1/nout_total)*Nb + Ndet
    snr = Np / np.sqrt(total_noise)

    return {
        "snr": snr,
        "uncertainty": 1/snr if snr > 0 else np.nan,
        "model_photons": Np,
        "star_photons": Ns,
        "bg_photons": Nb,
        "detector_noise": Ndet
    }

def integrate_flux(wave_um, flux, throughput):
    return np.trapz(flux * throughput, wave_um)

def compute_relative_emissions(nc_path, T_planet, wave_um, throughputs, star_flux, Rp_m, d_m, n_eclipses=5):
    data = load_agni_output(nc_path)
    model_flux_nm = data["ba_U_total"]
    wl_model_nm = data["bandcenter"]

    model_flux_um = model_flux_nm / 1000
    wl_model_um = wl_model_nm / 1000
    interp_model = np.interp(wave_um, wl_model_um, model_flux_um)

    omega_planet = np.pi * (Rp_m / d_m)**2
    interp_model_earth = interp_model * omega_planet
    bb_flux_earth = planck(wave_um * 1000, T_planet) * omega_planet / 1000

    results = {}
    for filt, tp in throughputs.items():
        model_int = integrate_flux(wave_um, interp_model_earth, tp)
        bb_int = integrate_flux(wave_um, bb_flux_earth, tp)
        results[filt] = model_int / bb_int if bb_int > 0 else np.nan

        snr_dict = compute_snr(wave_um, interp_model_earth, star_flux, tp, n_eclipses=n_eclipses)
        results[f"{filt}_SNR"] = snr_dict["snr"]
        results[f"{filt}_uncert"] = snr_dict["uncertainty"]
    return results

def main():
    planet = "gj486b"
    surface = "greybody"
    atmosphere = "bare_rock"
    output_dir = "out"
    surface_dir = "res/surfaces"
    atmosphere_dir = "res/atmospheres"

    pdata = get_planet_data(planet)
    T_star, R_star, R_planet = pdata["star_temp"], pdata["star_radius"], pdata["planet_radius"]
    d_au = pdata["planet_a"]
    d_m = d_au * c.au
    Rp_m = R_planet * c.r_earth

    T_planet = compute_dayside_brightness_temperature(
        stellar_temperature=T_star,
        stellar_radius_rsun=R_star,
        distance_au=d_au,
        bond_albedo=0,
        redistribution_factor=2/3
    )
    print(f"Tplanet: {T_planet:.2f} K")

    wave_um = np.linspace(10.0, 20.0, 1000)
    throughputs = {
        "F1280W": get_throughput(wave_um, "f1280w"),
        "F1500W": get_throughput(wave_um, "f1500w")
    }

    # Load stellar spectrum
    stellar_flux_path = os.path.join(CONFIG["stellar_spectra_dir"], "gj486_SPHINX.txt")
    star_flux = load_stellar_flux(stellar_flux_path, wave_um)
    star_flux *= (1.0 / d_au)**2

    # Surface
    surfaces = [f.replace(".dat", "") for f in os.listdir(surface_dir) if f.endswith(".dat")]
    results_surface = []
    for srf in surfaces:
        nc_path = os.path.join(output_dir, planet, srf, atmosphere, "atm.nc")
        if os.path.exists(nc_path):
            emissions = compute_relative_emissions(nc_path, T_planet, wave_um, throughputs, star_flux, Rp_m, d_m)
            results_surface.append({"Surface": srf, **emissions})
        else:
            print(f"[WARNING] Missing: {nc_path}")
    df_surface = pd.DataFrame(results_surface).sort_values("F1280W")

    # Atmosphere
    atmospheres = [f.replace(".toml", "") for f in os.listdir(atmosphere_dir) if f.endswith(".toml")]
    results_atmo = []
    for atmo in atmospheres:
        if "_O2" in atmo or "100bar_H2O" in atmo:
            print(f"[SKIPPED] {atmo}")
            continue
        nc_path = os.path.join(output_dir, planet, surface, atmo, "atm.nc")
        if os.path.exists(nc_path):
            emissions = compute_relative_emissions(nc_path, T_planet, wave_um, throughputs, star_flux, Rp_m, d_m)
            results_atmo.append({"Atmosphere": atmo, **emissions})
        else:
            print(f"[WARNING] Missing: {nc_path}")
    df_atmo = pd.DataFrame(results_atmo)

    df_atmo["Label"] = df_atmo["Atmosphere"].map(atmosphere_labels).fillna(df_atmo["Atmosphere"])
    ordered_labels = [atmosphere_labels.get(k, k) for k in atmosphere_labels.keys()]
    df_atmo["Label"] = pd.Categorical(df_atmo["Label"], categories=ordered_labels, ordered=True)
    df_atmo = df_atmo.sort_values("Label")

    scaling_factor = 11.5117  # From Trappist-1c
    for filt in ["F1500W", "F1280W"]:
        df_surface[f"{filt}_uncert"] *= scaling_factor
        df_atmo[f"{filt}_uncert"] *= scaling_factor

    # Plot
    fig = plt.figure(figsize=(22, 5))
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    x1 = np.arange(len(df_surface))
    ax1.axhline(1.0, linestyle="--", color="black")
    ax1.errorbar(x1, df_surface["F1500W"], yerr=df_surface["F1500W_uncert"], fmt=".", markersize=4, capsize=2, color="crimson", label="F1500W")
    ax1.errorbar(x1, df_surface["F1280W"], yerr=df_surface["F1280W_uncert"], fmt=".", markersize=4, capsize=2, color="dodgerblue", label="F1280W")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(df_surface["Surface"], rotation=45, ha="right")
    ax1.set_title(f"{planet}: JWST Surface emission", fontsize=15)
    ax1.set_ylabel("Emission relative to blackbody", fontsize=13)
    ax1.legend(loc=3)
    ax1.set_ylim(0, 1.25)
    ax1.grid(True, alpha=0.3)

    x2 = np.arange(len(df_atmo))
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.errorbar(x2, df_atmo["F1500W"], yerr=df_atmo["F1500W_uncert"], fmt=".", markersize=4, capsize=2, color="crimson", label="F1500W")
    ax2.errorbar(x2, df_atmo["F1280W"], yerr=df_atmo["F1280W_uncert"], fmt=".", markersize=4, capsize=2, color="dodgerblue", label="F1280W")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(df_atmo["Label"], rotation=45, ha="right")
    ax2.set_title(f"{planet}: JWST Atmosphere emission", fontsize=15)
    ax2.legend(loc=4)
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(bottom=0.4)
    out_dir = os.path.join("out", planet)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relative_emission_{atmosphere}_and_atmo.png")
    fig.savefig(out_path, dpi=300)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()

