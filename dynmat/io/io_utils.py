from typing import Dict

import toml

from ..cell import UnitCell


def load_unit_cell_from_toml(file_path: str) -> UnitCell:
    """
    Load unit cell from a TOML file and convert it to a UnitCell object.

    Args:
        file_path (str): Path to the TOML file.

    Returns:
        UnitCell: The UnitCell object initialized with the data from the file.
    """
    with open(file_path, "r") as f:
        data: Dict = toml.load(f)

    # Extract unit cell dimensions
    lattice_lengths = (
        data["unit_cell"]["a"],
        data["unit_cell"]["b"],
        data["unit_cell"]["c"],
    )
    lattice_angles = (
        data["unit_cell"].get("alpha", 90.0),
        data["unit_cell"].get("beta", 90.0),
        data["unit_cell"].get("gamma", 90.0),
    )

    # Extract atomic basis
    atomic_basis = [
        (atom["element"], tuple(atom["position"]))
        for atom in data["atoms"]["basis"]
    ]

    return UnitCell(
        lattice_lengths=lattice_lengths,
        lattice_angles=lattice_angles,
        atomic_basis=atomic_basis,
    )
