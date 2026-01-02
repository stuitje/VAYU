import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 2.99792458e8    # Speed of light (m/s)
k = 1.380649e-23    # Boltzmann constant (J/K)
R_sun = 6.957e8     # Solar radius in meters
AU = 1.496e11       # 1 AU in meters

def planck(wavelength_nm, T):
    """Blackbody spectral radiance in W/m²/nm at the stellar surface."""
    wavelength_m = wavelength_nm * 1e-9  # nm to meters
    numerator = 2 * h * c**2 / wavelength_m**5
    exponent = h * c / (wavelength_m * k * T)
    intensity = numerator / (np.exp(exponent) - 1)
    return intensity * 1e-9 * np.pi  # Convert W/m²/m to W/m²/nm and integrate over solid angle

def planck_diluted(wavelength_nm, T):
    """Planck function diluted to 1 AU for fitting."""
    R_star = 0.3243 * R_sun  # meters
    dilution = (R_star / AU) ** 2
    bb_surface = planck(wavelength_nm, T)
    bb_1au = bb_surface * dilution
    return bb_1au * 1e3  # convert to erg/s/cm²/nm

# Load the spectrum data
data = np.loadtxt('../res/stellar_spectra/gj486_SPHINX.txt', comments='#')
wavelength = data[:, 0]
flux = data[:, 1]  # erg/s/cm^2/nm at 1 AU

# Fit in the region of interest (e.g., 5000–20000 nm)
mask = (wavelength >= 1) & (wavelength <= 20000)
wavelength_fit = wavelength[mask]
flux_fit = flux[mask]

# Fit the blackbody model to the observed flux
popt, pcov = curve_fit(planck_diluted, wavelength_fit, flux_fit, p0=[3000])
best_fit_temp = popt[0]
print(f"Best-fit temperature: {best_fit_temp:.2f} K")

# Compute blackbody flux with original guess
T_star = 3317
R_star = 0.3243 * R_sun  # meters
bb_flux_surface = planck(wavelength, T_star)
dilution_factor = (R_star / AU) ** 2
bb_flux_1au = bb_flux_surface * dilution_factor
bb_flux_1au_cgs = bb_flux_1au * 1e3  # W/m²/nm → erg/s/cm²/nm

# Compute best-fit blackbody curve
bb_fit_flux = planck_diluted(wavelength, best_fit_temp)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, label='GJ486 Spectrum (1 AU)')
plt.plot(wavelength, bb_flux_1au_cgs, ':', label='Blackbody (3317 K)')
plt.plot(wavelength, bb_fit_flux, '--', label=f'Best-fit Blackbody ({best_fit_temp:.0f} K)')
plt.xlabel('Wavelength (nm)')
plt.xlim(100, 20000)
plt.ylabel('Flux (erg/s/cm^2/nm)')
plt.title('Spectrum of GJ486 at 1 AU vs. blackbody fit')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig('gj486_with_blackbody_fit.png', dpi=300)
