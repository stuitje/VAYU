import pandas as pd
import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit
import argparse

from src.constants import r_earth, r_sun
from src.utils import contrast_ppm


def fit_planet_temperature(
    csv_path: str,
    T_star: float,
    R_star: float,
    R_planet: float,
    init_guess: float = 1000
) -> Tuple[float, float]:
    """
    Fit the planet's surface temperature from observed contrast data.

    Args:
        csv_path: Path to CSV file with columns 'X' (micron), 'Y' (ppm), 'ΔY' (ppm error).
        T_star: Stellar effective temperature [K].
        R_star: Stellar radius [solar radii].
        R_planet: Planet radius [Earth radii].
        init_guess: Initial guess for T_planet [K].

    Returns:
        Best-fit temperature and uncertainty [K].
    """
    df = pd.read_csv(csv_path)
    required_cols = {'X', 'Y', 'ΔY'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}")

    wavelength_nm = df['X'].values * 1000  # micron -> nm
    contrast_obs = df['Y'].values
    contrast_err = df['ΔY'].values

    def fit_func(wavelength_nm, T_planet):
        return contrast_ppm(
            wavelength_nm,
            T_star=T_star,
            R_planet_m=R_planet * r_earth,
            R_star_m=R_star * r_sun,
            T_planet=T_planet
        )

    popt, pcov = curve_fit(
        fit_func,
        wavelength_nm,
        contrast_obs,
        sigma=contrast_err,
        p0=[init_guess],
        absolute_sigma=True
    )

    best_fit_temp = popt[0]
    uncertainty = np.sqrt(np.diag(pcov))[0]
    return best_fit_temp, uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a planet's surface temperature from observed contrast data."
    )

    parser.add_argument("csv", type=str, help="CSV file with contrast data (columns: X, Y, ΔY)")
    parser.add_argument("T_star", type=float, help="Stellar temperature [K]")
    parser.add_argument("R_star", type=float, help="Stellar radius [R_sun]")
    parser.add_argument("R_planet", type=float, help="Planet radius [R_earth]")
    parser.add_argument("--guess", type=float, default=1000.0, help="Initial guess for planet temperature [K]")

    args = parser.parse_args()

    try:
        temp, err = fit_planet_temperature(
            csv_path=args.csv,
            T_star=args.T_star,
            R_star=args.R_star,
            R_planet=args.R_planet,
            init_guess=args.guess
        )
        print(f"Best-fit surface temperature: {temp:.2f} ± {err:.2f} K")
    except Exception as e:
        print(f"[ERROR] {e}")
