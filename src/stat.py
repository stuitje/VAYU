import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def compute_chi_squared(
    observed_df: pd.DataFrame,
    model_wavelength_nm: np.ndarray,
    model_contrast: np.ndarray
) -> float:
    interp_model = interp1d(model_wavelength_nm, model_contrast, bounds_error=False, fill_value="extrapolate")
    model_vals = interp_model(observed_df["X"] * 1000)  # micron to nm
    chi2 = np.sum(((observed_df["Y"] - model_vals) / observed_df["ΔY"]) ** 2)
    dof = len(observed_df["Y"]) - 1
    return chi2 / dof