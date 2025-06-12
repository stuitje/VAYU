import toml
import pandas as pd
import os
from typing import Optional
import netCDF4 as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]

def load_agni_output(nc_path: str) -> dict:
    """
    Load AGNI NetCDF output and return key data arrays, including surface temperature.

    Args:
        nc_path: Path to AGNI .nc file

    Returns:
        Dictionary with bandcenter (nm), longwave and shortwave fluxes,
        total flux, and surface temperature.
    """
    ds = nc.Dataset(nc_path)
    bandmin = ds["bandmin"][:]
    bandmax = ds["bandmax"][:]
    bandcenter = (bandmin + bandmax) / 2 * 1e9  # convert to nm

    bandwidth = (bandmax - bandmin) * 1e9  # nm
    flux_lw = ds["ba_U_LW"][1, :] / bandwidth
    flux_sw = ds["ba_U_SW"][1, :] / bandwidth
    flux_total = flux_lw + flux_sw
    tmp_surf = ds["tmp_surf"][:].item()  # surface temperature in Kelvin

    return {
        "bandcenter": bandcenter,
        "ba_U_LW": flux_lw,
        "ba_U_SW": flux_sw,
        "ba_U_total": flux_total,
        "tmp_surf": tmp_surf
    }

def get_planet_data(name: str) -> dict:
    planet_csv_path = os.path.join(ROOT, CONFIG["planet_csv"])
    df = pd.read_csv(planet_csv_path)
    row = df[df["planet"].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Planet '{name}' not found.")
    return row.iloc[0].to_dict()

def load_contrast_data(path: str) -> Optional[pd.DataFrame]:
    full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.isfile(full_path):
        return pd.read_csv(full_path)
    return None

def load_atmosphere_toml(atmo_path: str):
    full_path = atmo_path if os.path.isabs(atmo_path) else os.path.join(ROOT, atmo_path)
    data = toml.load(full_path)
    comp = data.get("composition", {})
    transparent = comp.get("transparent", False)

    if transparent:
        print(f"[INFO] Atmosphere '{os.path.basename(atmo_path)}' is transparent.")
        return 1, 1, {}, True

    p_surf = comp.get("p_surf", 1e5)
    p_top = comp.get("p_top", 1e-5)
    vmr_dict = comp.get("vmr_dict", {})
    return float(p_surf), float(p_top), vmr_dict, False
