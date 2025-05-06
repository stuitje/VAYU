import numpy as np
from typing import Optional, Union
import netCDF4 as nc
from src.constants import au, stefan_boltzmann, l_sun, planck_const, boltzmann, speed_of_light


def planck(wavelength_nm: Union[np.ndarray, float], temperature_K: float) -> np.ndarray:
    """
    Compute blackbody spectral radiance using the Planck function.

    Args:
        wavelength_nm: Wavelength(s) in nanometers.
        temperature_K: Blackbody temperature in Kelvin.

    Returns:
        Spectral radiance in W/m²/nm integrated over steradians.
    """
    wavelength_m = np.array(wavelength_nm) * 1e-9
    exponent = planck_const * speed_of_light / (wavelength_m * boltzmann * temperature_K)
    spectral_radiance = (2.0 * planck_const * speed_of_light**2) / (wavelength_m**5) / (np.exp(exponent) - 1.0)
    return spectral_radiance * np.pi * 1e-9


def contrast_ppm(
    wavelength_nm: np.ndarray,
    T_star: float,
    R_planet_m: float,
    R_star_m: float,
    T_planet: Optional[float] = None,
    planet_flux: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the planet-star contrast in ppm for either a blackbody planet
    or a given AGNI flux spectrum.

    Args:
        wavelength_nm: Wavelengths at which to compute contrast [nm].
        T_star: Stellar effective temperature [K].
        R_planet_m: Planet radius [m].
        R_star_m: Stellar radius [m].
        T_planet: Planet surface temperature [K] (if blackbody model is used).
        planet_flux: Optional precomputed flux from an AGNI model [W/m²/nm].

    Returns:
        Contrast spectrum [ppm] as a numpy array.

    """
    wavelength_nm = np.array(wavelength_nm)
    star_flux = planck(wavelength_nm, T_star)

    if planet_flux is None:
        planet_flux = planck(wavelength_nm, T_planet)
    else:
        planet_flux = np.array(planet_flux)

    radius_ratio_sq = (R_planet_m / R_star_m) ** 2
    contrast = (planet_flux / star_flux) * radius_ratio_sq * 1e6  # Convert to ppm
    return contrast

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

def compute_equilibrium_temperature(
    stellar_luminosity_logL: float,
    distance_au: float,
    bond_albedo: float = 0.0,
    redistribution_factor: float = 0.5  # 1.0 = full redistribution, 0.5 = dayside only
) -> float:
    """
    Compute equilibrium temperature with heat redistribution.

    Args:
        stellar_luminosity_logL: log10(L / L_sun)
        distance_au: orbital distance [AU]
        bond_albedo: fraction of light reflected
        redistribution_factor: fractional emitting area 

    Returns:
        Equilibrium temperature [K]
    """
    L_star = 10**stellar_luminosity_logL * l_sun
    d_m = distance_au * au

    T_eq = ((1 - bond_albedo) * L_star / (16 * np.pi * stefan_boltzmann * d_m**2 * redistribution_factor))**0.25
    return T_eq


