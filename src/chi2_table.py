import os
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.utils import load_agni_output, contrast_ppm
from src.constants import r_earth, r_sun
from src.plots import compute_chi_squared
from src.atmosphere_labels import atmosphere_labels



def is_bare_surface(atmo_key: str) -> bool:
    return atmo_key.lower() == "bare_rock"

def generate_chi2_table(
    output_dir: str,
    planet: str,
    contrast_data: pd.DataFrame,
    T_star: float,
    R_star_rsun: float,
    R_planet_rearth: float) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Return two lists of (name, chi2): bare-rock surface results and atmosphere (greybody) results.
    """
    planet_dir = os.path.join(output_dir, planet)
    R_planet = R_planet_rearth * r_earth
    R_star = R_star_rsun * r_sun

    bare_results = []
    atmo_results = []

    for surface in os.listdir(planet_dir):
        surface_dir = os.path.join(planet_dir, surface)
        if not os.path.isdir(surface_dir):
            continue
        for atmo in os.listdir(surface_dir):
            nc_path = os.path.join(surface_dir, atmo, "atm.nc")
            if not os.path.isfile(nc_path):
                continue

            data = load_agni_output(nc_path)
            contrast_model = contrast_ppm(
                wavelength_nm=data["bandcenter"],
                T_star=T_star,
                R_planet_m=R_planet,
                R_star_m=R_star,
                planet_flux=data["ba_U_total"]
            )
            chi2 = compute_chi_squared(contrast_data, data["bandcenter"], contrast_model)

            if is_bare_surface(atmo):
                bare_results.append((surface, chi2))
            elif surface == 'greybody':
                atmo_results.append((atmo, chi2))

    return bare_results, atmo_results

def write_chi2_table(
    planet: str,
    output_dir: str,
    bare_results: List[Tuple[str, float]],
    atmo_results: List[Tuple[str, float]],
    filename: str = "chi2_summary.txt"
):
    lines = []
    lines.append(f"Planet: {planet}\n")
    lines.append("Bare-rock Surfaces        | chi-2   |   Atmospheres                       | chi-2   |")
    lines.append("--------------------------|---------|-------------------------------------|---------|")

    max_len = max(len(bare_results), len(atmo_results))
    for i in range(max_len):
        bare_str = f"{bare_results[i][0]:<25} | {bare_results[i][1]:<8.2f}" if i < len(bare_results) else "                          |       "

        if i < len(atmo_results):
            atmo_key = atmo_results[i][0]
            atmo_label = atmosphere_labels.get(atmo_key, atmo_key)
            clean_label = atmo_label.replace("_", "").replace("$", "")
            atmo_str = f"{clean_label:<35} | {atmo_results[i][1]:<8.2f}"
        else:
            atmo_str = "                                    |       "

        lines.append(f"{bare_str}| {atmo_str:<46}|")

    path = os.path.join(output_dir, planet, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Chi-2 summary written to {path}")
