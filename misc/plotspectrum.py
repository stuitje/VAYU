import numpy as np
import matplotlib.pyplot as plt

# Load the spectrum data
data = np.loadtxt('../res/stellar_spectra/gj486.txt', comments='#')

# Split into wavelength and flux
wavelength = data[:, 0]
flux = data[:, 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, label='GJ486 Spectrum (1 AU)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (erg/s/cm²/nm)')
plt.title('Spectrum of GJ367 at 1 AU')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('gj486_spectrum_plot.png', dpi=300)

# Optionally show the plot
plt.show()
