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
│   ├── surfaces/             # Surface albedo files (.dat)
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

