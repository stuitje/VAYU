# AGNI-integrated Rocky Surfaces and Atmospheres Analysis

A modular Python-based pipeline for modeling exoplanetary atmospheres and surfaces, generating synthetic spectra, and comparing outputs with observational contrast data. The system integrates with a Julia-based radiative transfer model (AGNI) to simulate and analyze a variety of atmospheric and surface conditions.

AGNI: nichollsh.github.io/AGNI/

---

## Overview

This project allows the user to:

- Generate and configure planetary models using stellar and planetary data.
- Create synthetic spectra from both blackbody and radiative transfer simulations.
- Fit planetary surface temperatures to observational contrast data.
- Plot spectral flux and planet-star contrast for model comparison.
- Explore effects of varying atmospheric compositions, surface materials, and instellation levels.

---

## Structure

```
VAYU/
│
├── AGNI/                     # Julia-based AGNI radiative transfer engine
├── res/    
│   ├── atmospheres/          # Basic atmosphere composition TOMLs  
│   ├── config/               # Generated AGNI configuration TOMLs
│   ├── planetary_data/       # Basic planet data, and observational contrasts
│   ├── stellar_spectra/      # Stellar spectra files (.txt)
│   └── surfaces/             # Surface albedo files (.dat)
│
├── out/                      # Model results and plots saved here (per planet)
├── src/
│   ├── atmosphere_labels.py  # Labels for atmosphere keys
│   ├── config_gen.py         # Generates AGNI config TOMLs
│   ├── constants.py          # Physical and astronomical constants
│   ├── pipeline.py           # Main orchestration script
│   ├── plots.py              # Plotting utilities
│   ├── temperature_fit.py    # Fit temperature to observational data
│   ├── throughput.py         # Calculate JWST Miri filter throughput for simulated emission
│   └── utils.py              # Math + I/O helpers (Planck, contrast, etc.)
│
├── agni_config.toml          # Paths to directories
├── atmos_list.toml           # Optional list of atmospheres to loop over
└── README.md                 # This file 
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/stuitje/VAYU.git
   cd VAYU
   ```

2. **Set up Python environment**

   Make sure you have the required dependencies installed:

   ```bash
   pip install -r requirements.txt
   ```

   Required packages include:
   - `numpy`, `pandas`, `matplotlib`, `scipy`, `toml`, `tomlkit`, `netCDF4`

3. **Install AGNI**

   Follow the AGNI installation instructions as explained in: https://nichollsh.github.io/AGNI/dev/setup/. AGNI must be placed inside the root directory (`VAYU/`). From the root directory, you can test AGNI using:
. 

   ```bash
   julia AGNI/test/runtests.jl
   ```

4. **Configure paths**

   If needed, update `agni_config.toml` to reflect your machine's paths. Default: 

   ```toml
    # Base AGNI directory (used to access the model engine)
    agni_dir = "AGNI"

    # Input files and directories 
    planet_csv = "res/planetary_data/exoplanetarchive.csv"
    obs_data_dir = "res/planetary_data"
    atmosphere_dir = "res/atmospheres"
    surface_dir = "res/surfaces"
    stellar_spectra_dir = "res/stellar_spectra"

    # AGNI input files 
    spectral_file = "AGNI/res/spectral_files/Honeyside/256/Honeyside.sf"
    spectral_file_O2 = "AGNI/res/spectral_files/Reach/318/Reach.sf"

    # Outputs
    output_dir = "out"
    config_dir = "res/config"
   ```

---
