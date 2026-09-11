"""
waveguide_sensitivity.py — Analyte sensitivity analysis for the photonic-crystal
waveguide sensor (n = 1.45).

Refactored from the first half of "main (4).py". The wavelength-shift
extraction portion of that same file (and of "main1 (2).py") now lives in
wavelength_shift.py, since the target repository architecture splits those
into two separate concerns.
"""

from typing import List, Tuple

from .mode_properties import calculate_mode_properties


def calculate_sensitivity(
    mode_properties: List[Tuple[float, float, float]], analyte_refractive_index: float
) -> List[float]:
    """
    Sensitivity of each computed mode to a change in analyte refractive index.

    Uses the original project's simplified formula:
        sensitivity = n_eff * (n_analyte - 1) / n_analyte
    """
    sensitivity_values = []
    for kx, neff, mode_frequency in mode_properties:
        sensitivity = neff * (analyte_refractive_index - 1) / analyte_refractive_index
        sensitivity_values.append(sensitivity)
    return sensitivity_values


def run(
    wavelength: float = 1.55,
    lattice_constant: float = 0.42,
    holes_diameter: float = 0.25,
    slab_thickness: float = 0.26,
    analyte_refractive_index: float = 1.45,
):
    mode_properties = calculate_mode_properties(wavelength, lattice_constant, holes_diameter, slab_thickness)
    sensitivity_results = calculate_sensitivity(mode_properties, analyte_refractive_index)
    return mode_properties, sensitivity_results


if __name__ == "__main__":
    _, sensitivities = run()
    for s in sensitivities:
        print(f"Sensitivity: {s:.4f}")
