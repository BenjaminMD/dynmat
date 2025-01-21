import pytest


def test_imports():
    try:
        from dynmat.cell import UnitCell
        from dynmat.force import (calculate_force_constant,
                                  calculate_periodic_direction,
                                  get_atomic_mass, get_force, get_forces_3x3,
                                  get_harmonic_force)
        from dynmat.geometry import fractional_to_cartesian_transform
        from dynmat.visualize import plot_force_constant_matrix
    except ModuleNotFoundError as e:
        pytest.fail(f"Import failed: {e}")
