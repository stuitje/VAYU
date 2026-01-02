# AGNI-integrated Rocky Surfaces and Atmospheres Analysis

A modular Python-based pipeline for modeling exoplanetary atmospheres and surfaces, generating synthetic eclipse depths, and comparing outputs with observational contrast data. The system integrates with a Julia-based radiative transfer model ([AGNI](https://nichollsh.github.io/AGNI/)) to simulate and analyze a variety of atmospheric and surface conditions.

---

## Overview

This project allows the user to:

- Model generation: Automatically configure and run AGNI using planetary and stellar parameters, surface albedo profiles, and atmospheric compositions.

- Synthetic eclipse depth computation: Simulate secondary eclipse depths for  blackbodies, greybodies, bare-rock surfaces, and full atmosphere-surface radiative transfer models. 

- Temperature fitting: Fit the planet’s dayside brightness temperature from observed contrast (eclipse depth) data.

- Bayesian model comparison: Use nested sampling to compute the Bayesian evidence for each model and evaluate their relative support given the data.

- JWST simulation: Calculate band-integrated thermal emission in JWST/MIRI filters (e.g., F1280W, F1500W), compute signal-to-noise ratios, and assess detectability across surface and atmosphere types.

---

## Structure

```
VAYU/
│
├── AGNI/                     # Julia-based AGNI radiative transfer engine
├── misc/                     # Miscellaneous files
├── res/    
│   ├── atmospheres/          # Basic atmosphere composition TOMLs  
│   ├── config/               # Generated AGNI configuration TOMLs
│   ├── planetary_data/       # Basic planet data, and observational contrasts
│   ├── stellar_spectra/      # Stellar spectra files (.txt)
│   └── surfaces/             # Surface albedo files (.dat)
│
├── out/                      # Model results and plots saved here (per planet)
├── plt/                      # Additional plotting scripts, primarily used for my thesis
├── src/
│   ├── atmosphere_labels.py  # Labels for atmosphere keys
│   ├── chi2_table.py         # Creates chi-2 summary table
│   ├── config_gen.py         # Generates AGNI config TOMLs
│   ├── constants.py          # Physical and astronomical constants
│   ├── dataloader.py         # Loads data 
│   ├── emission_miri.py      # Simulates observed emission with JWST Miri filters
│   ├── nested_sampling.py    # Uses a nested sampling method to compare models to the data (Zhang et al, 2024)
│   ├── pipeline.py           # Main orchestration script
│   ├── plots.py              # Plotting utilities
│   ├── stat.py               # Statistical functions 
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
   - `numpy`, `pandas`, `matplotlib`, `scipy`, `toml`, `tomlkit`, `netCDF4`, `os`

3. **Install AGNI**

   Follow the AGNI installation instructions as explained in: https://nichollsh.github.io/AGNI/dev/setup/. AGNI must be placed inside the root directory (`VAYU/`). From the root directory, you can test AGNI using:
. 

   ```bash
   julia AGNI/test/runtests.jl
   ```

4. **Configure paths**

   If needed, update `agni_config.toml` to reflect your machine's paths. Default: 

   ```[paths]
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
   spectral_file_H2O = "AGNI/res/spectral_files/Frostflow/256/Frostflow.sf"
   spectral_file_O2 = "AGNI/res/spectral_files/Honeyside_with_O2/256/Honeyside.sf"
   spectral_file_Si = "AGNI/res/spectral_files/Rocks/256/Rocks.sf"

   # JWST observational modelling
   pandeia_dir = "/dataserver/users/formingworlds/stuitje/pandeia"

   # Outputs
   output_dir = "out"
   config_dir = "res/config"

   ```

---

## Usage

### Run a full pipeline

```bash
python -m src.pipeline <planet_name> -s <surface_name|all|list> -a <atmosphere_name|all|list> [--flux true|false]
```

- `planet_name`: Name of the planet as in the CSV.
- `-s`: `'all'` to run over all surface files, `'list'` (from `surface_list.toml`), or specific name.
- `-a`: `'all'`, `'list'` (from `atmos_list.toml`), or specific name.
- `-T`: temperature model to use, which modifies the heat redistribution factor used. `'dayside'` is the default, for no heat redistribution but an average dayside flux (f = 2/3); `'substellar'` assumes no heat distribution and only the substellar point (f = 1); and `'full'` assumes full heat redistribution (f = 1/4).
- `--flux-only`: to just generate flux-and-contrast plots (no combined contrast plots).
- `--no-run`: skips running AGNI, but directly uses the atm.nc file generated by AGNI earlier to generate output. Can be used when, for example, modifying the plot itself. 

Example: running a pipeline for Trappist-1c, using a 1 bar CO2 atmosphere with full heat redistribution (f = 1/4). 

```bash
python -m src.pipeline trappist-1c -s greybody -a 1bar_CO2 -T full
```
Example: running a pipeline for Trappist-1c, using a list (`surface_list.toml`) of bare surfaces with no heat redistribution (f = 2/3, default).

```bash
python -m src.pipeline trappist-1c -s list -a bare_rock
```


---