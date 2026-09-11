"""
wavelength_shift.py — Wavelength shift extraction across the 1.38 - 1.74 µm
analyte sweep window.

Refactored from two source files:
    - The tail of "main (4).py", which zipped sensitivity results against a
      np.linspace(1.38, 1.74, 100) wavelength axis.
    - "main1 (2).py", which attempted to plot a similar sensitivity-vs-wavelength
      curve but referenced undefined names (`sensitivity_results`,
      `wavelength_range`, `calculate_sensitivity`) at module scope — it would
      raise a NameError if actually run. That plotting logic is preserved
      here but rewritten to compute its own inputs correctly.
"""

import logging
from typing import List, Tuple

import numpy as np

from .mode_properties import calculate_mode_properties
from .waveguide_sensitivity import calculate_sensitivity

logger = logging.getLogger(__name__)


def extract_wavelength_shift(
    wavelength: float = 1.55,
    lattice_constant: float = 0.42,
    holes_diameter: float = 0.25,
    slab_thickness: float = 0.26,
    analyte_refractive_index: float = 1.45,
    wavelength_range: Tuple[float, float] = (1.38, 1.74),
    n_points: int = 100,
) -> List[Tuple[float, float]]:
    """
    Compute sensitivity for each mode, then pair each value against a swept
    wavelength axis (wavelength_range) to report the shift response across
    the sensor's usable analyte window.

    Returns a list of (swept_wavelength_um, sensitivity) tuples.
    """
    mode_properties = calculate_mode_properties(wavelength, lattice_constant, holes_diameter, slab_thickness)
    sensitivity_results = calculate_sensitivity(mode_properties, analyte_refractive_index)

    swept_wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], len(sensitivity_results))
    return list(zip((float(w) for w in swept_wavelengths), sensitivity_results))


def plot_wavelength_shift(results: List[Tuple[float, float]], show: bool = True):
    """Plot sensitivity vs. swept wavelength (mirrors main1 (2).py's intent, now working)."""
    import matplotlib.pyplot as plt

    wavelengths, sensitivities = zip(*results)
    plt.plot(wavelengths, sensitivities, marker="o")
    plt.xlabel("Wavelength (micrometers)")
    plt.ylabel("Sensitivity")
    plt.title("Photonic Crystal Waveguide Sensitivity Analysis")
    plt.grid(True)
    if show:
        plt.show()
    return plt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = extract_wavelength_shift()
    for wl, sensitivity in results:
        print(f"Wavelength: {wl:.4f} \u03bcm, Sensitivity: {sensitivity:.4f}")
