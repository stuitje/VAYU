import os
import numpy as np
import pandas as pd
import toml
from tomlkit import document, table, inline_table, dumps

from src.constants import r_earth, m_earth, r_sun, l_sun, au, G
from src.utils import planck

# Load config from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]


def generate_blackbody_spectrum(
    teff: float,
    star_radius_rsun: float,
    output_path: str,
    wav_min_nm: int = 100,
    wav_max_nm: int = 50000,
    resolution: int = 1000
) -> None:
    wavelengths_nm = np.logspace(np.log10(wav_min_nm), np.log10(wav_max_nm), resolution)
    flux = planck(wavelengths_nm, teff)

    star_radius_m = star_radius_rsun * r_sun
    scale_factor = (star_radius_m / au) ** 2
    flux_scaled = flux * scale_factor

    spectrum = np.column_stack((wavelengths_nm, flux_scaled))
    np.savetxt(
        output_path,
        spectrum,
        fmt="%.7e",
        delimiter="\t",
        header="WL(nm)\tFlux(W/m^2/nm)",
        comments="# "
    )
    print(f"Wrote synthetic blackbody to: {output_path}")


def load_atmosphere_toml(atmo_path: str):
    data = toml.load(atmo_path)
    comp = data.get("composition", {})
    transparent = comp.get("transparent", False)

    if transparent:
        print(f"[INFO] Atmosphere '{os.path.basename(atmo_path)}' is transparent.")
        return 1, 1, {}, True

    p_surf = comp.get("p_surf", 1e5)
    p_top = comp.get("p_top", 1e-5)
    vmr_dict = comp.get("vmr_dict", {})
    return float(p_surf), float(p_top), vmr_dict, False


def write_agni_config(
    planet_name: str,
    atmosphere_name: str,
    surface_name: str,
    tmp_surf: float
) -> None:
    df = pd.read_csv(CONFIG["planet_csv"])
    row = df[df["planet"].str.lower() == planet_name.lower()]
    if row.empty:
        raise ValueError(f"Planet '{planet_name}' not found in {CONFIG['planet_csv']}")
    row = row.iloc[0]

    planet_mass = float(row["planet_mass"])
    planet_radius = float(row["planet_radius"])
    stellar_luminosity = float(10 ** row["star_lum"])
    distance_from_star = float(row["planet_a"])
    fallback_teff = float(row["star_temp"])
    star_radius_rsun = float(row["star_radius"])
    host_star = row["host"].lower()

    # Atmosphere
    atmo_path = os.path.join(CONFIG["atmosphere_dir"], f"{atmosphere_name}.toml")
    p_surf, p_top, vmr_dict, transparent = load_atmosphere_toml(atmo_path)
    solver = "transparent" if transparent else "gauss"
    solution = 1 #if transparent else 0

    # Surface
    if surface_name != 'greybody':
        surface_path = os.path.join(CONFIG["surface_dir"], f"{surface_name}.dat")
        if not os.path.exists(surface_path):
            raise FileNotFoundError(f"Surface file not found: {surface_path}")
    else:
        surface_path = surface_name

    # Paths
    config_name = f"{planet_name}_{surface_name}_{atmosphere_name}".lower()
    config_dir = os.path.join(CONFIG["config_dir"], config_name)
    os.makedirs(config_dir, exist_ok=True)

    agni_output_dir = os.path.join(CONFIG["output_dir"], planet_name, surface_name, atmosphere_name)
    os.makedirs(agni_output_dir, exist_ok=True)

    star_spectrum_path = os.path.join(CONFIG["stellar_spectra_dir"], f"{host_star}.txt")
    if os.path.exists(star_spectrum_path):
        input_star = star_spectrum_path
    else:
        fallback_path = os.path.join(config_dir, f"bb_{fallback_teff}K.txt")
        generate_blackbody_spectrum(fallback_teff, star_radius_rsun, fallback_path)
        input_star = fallback_path

    instellation = (stellar_luminosity * l_sun) / (4 * np.pi * (distance_from_star * au) ** 2)
    gravity = (G * (planet_mass * m_earth)) / (planet_radius * r_earth) ** 2
    radius = planet_radius * r_earth

    # --- TOML Construction ---
    doc = document()
    doc["title"] = config_name

    planet = table()
    planet["tmp_surf"] = tmp_surf
    planet["instellation"] = round(instellation, 2)
    planet["albedo_b"] = 0.0
    planet["s0_fact"] = 0.6652
    planet["zenith_angle"] = 0.0
    planet["surface_material"] = surface_path
    planet["albedo_s"] = 0.0
    planet["radius"] = radius
    planet["gravity"] = gravity
    planet["skin_d"] = 0.01
    planet["skin_k"] = 2.0
    planet["tmp_magma"] = 3000.0
    planet["flux_int"] = 0.0
    planet["turb_coeff"] = 0.001
    planet["wind_speed"] = 2.0
    doc["planet"] = planet

    files = table()
    files["input_sf"] = CONFIG["spectral_file"]
    files["input_star"] = input_star
    files["output_dir"] = agni_output_dir
    doc["files"] = files

    comp = table()
    comp["p_surf"] = p_surf
    comp["p_top"] = p_top
    if vmr_dict:
        vmr = inline_table()
        for k, v in vmr_dict.items():
            vmr[k] = float(v)
        comp["vmr_dict"] = vmr
    comp["vmr_path"] = ""
    comp["include_all"] = False
    comp["chemistry"] = 0
    comp["condensates"] = []
    comp["transparent"] = transparent
    doc["composition"] = comp

    exec_ = table()
    exec_["clean_output"] = True
    exec_["verbosity"] = 1
    exec_["max_steps"] = 20000
    exec_["max_runtime"] = 1000
    exec_["num_levels"] = 50
    exec_["continua"] = True
    exec_["rayleigh"] = True
    exec_["cloud"] = False
    exec_["aerosol"] = False
    exec_["overlap_method"] = "ee"
    exec_["real_gas"] = True
    exec_["thermo_funct"] = True
    exec_["sensible_heat"] = False
    exec_["latent_heat"] = False
    exec_["convection"] = True
    exec_["rainout"] = False
    exec_["solution_type"] = solution
    exec_["solver"] = solver
    exec_["dx_max"] = 200.0
    exec_["initial_state"] = ["dry", "sat", "H2O"]
    exec_["linesearch"] = 0
    exec_["easy_start"] = False
    exec_["converge_atol"] = 1.0e-3
    exec_["converge_rtol"] = 1.0e-1
    doc["execution"] = exec_

    plots = table()
    plots["at_runtime"] = False
    plots["temperature"] = True
    plots["fluxes"] = False
    plots["contribution"] = False
    plots["emission"] = True
    plots["albedo"] = True
    plots["mixing_ratios"] = False
    plots["animate"] = False
    plots["height"] = False
    doc["plots"] = plots

    output_file = os.path.join(config_dir, "config.toml")
    with open(output_file, "w") as f:
        f.write(dumps(doc))

    print(f"AGNI config file written to: {output_file}")
