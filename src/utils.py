import numpy as np
from typing import Optional, Union
import netCDF4 as nc
import os 
from src.constants import au, r_sun, stefan_boltzmann, l_sun, planck_const, boltzmann, speed_of_light
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


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

def planck_diluted(wavelength_nm, T, r_star):
    """Planck function diluted to 1 AU for fitting. r_star in meters."""
    dilution = (r_star / au) ** 2
    bb_surface = planck(wavelength_nm, T)
    bb_1au = bb_surface * dilution
    return bb_1au * 1e3  # convert to erg/s/cm^2/nm for fitting

def fit_to_spectrum(data, r_star):
    wavelength = data[:, 0]
    flux = data[:, 1]  # erg/s/cm^2/nm at 1 AU

    # Fit only in the region of interest 
    mask = (wavelength >= 5000) & (wavelength <= 20000)
    wavelength_fit = wavelength[mask]
    flux_fit = flux[mask]

    # Create a fitting function
    fit_func = lambda wavelength_nm, T: planck_diluted(wavelength_nm, T, r_star)

    # Fit the blackbody model to the observed flux
    popt, pcov = curve_fit(fit_func, wavelength_fit, flux_fit, p0=[3000]) # 3000 as a general initial guess for M-stars
    best_fit_temp = popt[0]

    return best_fit_temp


def contrast_ppm(
    wavelength_nm: np.ndarray,
    T_star: float,
    R_planet_m: float,
    R_star_m: float,
    T_planet: Optional[float] = None,
    planet_flux: Optional[np.ndarray] = None,
    stellar_spectrum: Optional[Union[np.ndarray, str]] = None,
    rescale : Optional[bool] = True,
    T_spectrum: Optional[float] = 3300  ) -> np.ndarray:

    """
    Compute the planet-star contrast in ppm for either a blackbody planet
    or a given AGNI flux spectrum.

    Args:
        wavelength_nm: Wavelengths at which to compute contrast [nm].
        T_star: Stellar effective temperature [K].
        R_planet_m: Planet radius [m].
        R_star_m: Stellar radius [m].
        T_planet: Planet surface temperature [K] (if blackbody model is used).
        planet_flux: Optional precomputed flux from an AGNI model [W/m^2/nm].
        stellar_spectrum: Optional file path to stellar spectrum or flux array [W/m^2/nm].

    Returns:
        Contrast spectrum [ppm] as a numpy array.

    """
    wavelength_nm = np.array(wavelength_nm)

    if stellar_spectrum is None:
        star_flux = planck(wavelength_nm, T_star)
    elif isinstance(stellar_spectrum, str) and os.path.isfile(stellar_spectrum):

        # avoid divide by zero 
        wavelength_mask = wavelength_nm >= 100  # nm
        wavelength_nm = wavelength_nm[wavelength_mask]

        # Load stellar file assuming: wavelength [nm], flux [erg/cm^2/s/nm at 1 AU]
        data = np.loadtxt(stellar_spectrum, comments = '#')

        stellar_wl = data[:, 0]
        stellar_flux = data[:, 1]

        # Erg/cm^2/s/nm to W/m^2/nm
        stellar_flux *= 1e-3

        # De-dilute from 1 AU to stellar surface
        scale_factor = (au / R_star_m) ** 2
        stellar_flux *= scale_factor

        # Create blackbody tail if spectrum falls short (e.g. PHOENIX)
        max_wl_star = stellar_wl.max()
        max_wl_AGNI = wavelength_nm.max()

        if max_wl_AGNI > max_wl_star:
            tail_wavelengths = np.linspace(max_wl_star, max_wl_AGNI, 200)[1:]  
            tail_flux = planck(tail_wavelengths, T_star)  

            # Concatenate real + BB
            stellar_wl = np.concatenate([stellar_wl, tail_wavelengths])
            stellar_flux = np.concatenate([stellar_flux, tail_flux])

        # Interpolate to used wavelength grid
        interp_func = interp1d(stellar_wl, stellar_flux, kind='linear',
                               bounds_error=True)
        star_flux = interp_func(wavelength_nm)

        # Rescale stellar flux to correct stellar temperature 
        if rescale:
            flux_bb_actual = planck(wavelength_nm, T_spectrum) 
            flux_bb_target = planck(wavelength_nm, T_star) 
            bb_scale = flux_bb_target / flux_bb_actual  # element-wise
            star_flux *= bb_scale

    else:
        # Assume it's already a flux array matching wavelength_nm
        star_flux = np.array(stellar_spectrum)


    if planet_flux is None:
        planet_flux = planck(wavelength_nm, T_planet)
    else:
        planet_flux = np.array(planet_flux)
        
        # Make sure planet_flux wavelength is also clipped if a real stellar spectrum is used
        if isinstance(stellar_spectrum, str) and os.path.isfile(stellar_spectrum):
            planet_flux = planet_flux[wavelength_mask]

    radius_ratio_sq = (R_planet_m / R_star_m) ** 2
    contrast = (planet_flux / star_flux) * radius_ratio_sq * 1e6  # Convert to ppm
    return contrast

def compute_equilibrium_temperature(
    stellar_luminosity: float,
    distance_au: float,
    bond_albedo: float = 0.0,
    redistribution_factor: float = 0.5  # 1.0 = full redistribution, 0.5 = dayside only
) -> float:
    """
    Compute equilibrium temperature with heat redistribution.

    Args:
        stellar_luminosity: log10(L / L_sun)
        distance_au: orbital distance [au]
        bond_albedo: fraction of light reflected
        redistribution_factor: fractional emitting area 

    Returns:
        Equilibrium temperature [K]
    """
    L_star = 10**stellar_luminosity * l_sun
    d_m = distance_au * au

    T_eq = ((1 - bond_albedo) * L_star / (16 * np.pi * stefan_boltzmann * d_m**2 * redistribution_factor))**0.25
    return T_eq

import numpy as np

# Constants
au = 1.496e11  # Astronomical unit in meters
r_sun = 6.957e8  # Solar radius in meters

def compute_dayside_brightness_temperature(
    stellar_temperature: float,
    stellar_radius_rsun: float,
    distance_au: float,
    bond_albedo: float = 0.0,
    redistribution_factor: float = 2/3  # 2/3 = no redistribution, 1/4 = full redistribution
) -> float:
    """
    Compute dayside brightness temperature for a tidally locked planet, from Zhang et al (2024).

    Args:
        stellar_temperature: Effective temperature of the star [K]
        stellar_radius_rsun: Stellar radius in solar radii [R_sun]
        distance_au: Orbital distance [AU]
        bond_albedo: Bond albedo of the planet (0 to 1)
        redistribution_factor: f, where 1 is no redistribution, 1/4 is full redistribution

    Returns:
        Dayside brightness temperature [K]
    """
    R_s = stellar_radius_rsun * r_sun
    a = distance_au * au

    T_db = stellar_temperature * np.sqrt(R_s / a) * ((1 - bond_albedo) * redistribution_factor)**0.25
    return T_db



