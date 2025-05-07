import numpy as np
import matplotlib.pyplot as plt
import os
from pandeia.engine.instrument_factory import InstrumentFactory

def main():
    # Wavelength range in microns
    wave_um = np.linspace(10.0, 20.0, 1000)
    wave_nm = wave_um * 1000

    # Pandeia configuration for MIRI F1500W
    conf = {
        "detector": {
            "nexp": 1,
            "ngroup": 10,
            "nint": 1,
            "readout_pattern": "fastr1",
            "subarray": "full"
        },
        "instrument": {
            "aperture": "imager",
            "filter": "f1500w",
            "instrument": "miri",
            "mode": "imaging"
        },
    }

    instrument_factory = InstrumentFactory(config=conf)
    efficiency = instrument_factory.get_total_eff(wave_um)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wave_um, efficiency, label="F1500W Throughput")
    ax.set_title("JWST MIRI F1500W Throughput")
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel("Throughput")
    ax.grid(True, linestyle=":")
    ax.legend()

    out_dir = "out/throughput/F1500W"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "throughput.png"), dpi=300)
    print(f"Saved plot to {os.path.join(out_dir, 'throughput.png')}")

if __name__ == "__main__":
    main()
