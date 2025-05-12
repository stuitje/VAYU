import toml
import pandas as pd
import os
from typing import Optional
import netCDF4 as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = toml.load(os.path.join(ROOT, "agni_config.toml"))["paths"]

def load_agni_output(nc_path: str) -> dict:
    """
    Load AGNI NetCDF output and return key data arrays.

    Args:
        nc_path: Path to AGNI .nc file

    Returns:
        Dictionary with bandcenter (nm), longwave and shortwave fluxes,
        and total flux. 
    """
    ds = nc.Dataset(nc_path)
    bandmin = ds["bandmin"][:]
    bandmax = ds["bandmax"][:]
    bandcenter = (bandmin + bandmax) / 2 * 1e9  # convert to nm

    bandwidth = (bandmax - bandmin) * 1e9  # nm
    flux_lw = ds["ba_U_LW"][1, :] / bandwidth
    flux_sw = ds["ba_U_SW"][1, :] / bandwidth
    flux_total = flux_lw + flux_sw

    return {
        "bandcenter": bandcenter,
        "ba_U_LW": flux_lw,
        "ba_U_SW": flux_sw,
        "ba_U_total": flux_total
    }

def get_planet_data(name: str) -> dict:
    df = pd.read_csv(CONFIG["planet_csv"])
    row = df[df["planet"].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Planet '{name}' not found.")
    return row.iloc[0].to_dict()

def load_contrast_data(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None