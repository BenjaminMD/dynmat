import numpy as np
from mendeleev import element
from numpy.typing import NDArray

from .utils import periodic_distance


def calculate_force_constant(
    atom1_label: str,
    atom2_label: str,
    reference_frequency: float = 1.0,
) -> float:
    """
    Calculate force constant based on atomic masses and reference frequency.
    """
    mass1 = element(atom1_label).mass
    mass2 = element(atom2_label).mass
    reduced_mass = (mass1 * mass2) / (mass1 + mass2)
    angular_frequency = 2 * np.pi * reference_frequency
    return reduced_mass * angular_frequency**2


def calculate_periodic_direction(point1: NDArray, point2: NDArray) -> NDArray:
    """
    Calculate unit vector considering periodic boundaries.
    """
    delta = point2 - point1
    delta = delta - np.round(delta)
    norm = np.linalg.norm(delta)
    return delta / norm if norm > 0 else np.zeros_like(delta)


def get_force(
    positions: NDArray,
    labels: NDArray,
    atom1: NDArray,
    atom2: NDArray,
    displacement: NDArray,
    distance_matrix: NDArray,
    reference_frequency: float = 1.0,
) -> NDArray:
    """
    Calculate force between two atoms with mass-dependent force constant.
    """
    # Find atom indices and labels
    atom1_idx = np.where(np.all(positions == atom1, axis=1))[0][0]
    atom2_idx = np.where(np.all(positions == atom2, axis=1))[0][0]
    atom1_label = labels[atom1_idx][0]
    atom2_label = labels[atom2_idx][0]

    # Get equilibrium distance
    equilibrium_dist = distance_matrix[atom1_idx, atom2_idx]

    # Calculate mass-dependent force constant
    k = calculate_force_constant(atom1_label, atom2_label, reference_frequency)

    # Calculate new distance and direction after displacement
    displaced_pos = atom2 + displacement
    new_dist = periodic_distance(atom1, displaced_pos)
    direction = calculate_periodic_direction(atom1, displaced_pos)
    equi = np.ravel(equilibrium_dist)[0]
    # Calculate force
    # print( equi)
    return (k * (equi - np.array(new_dist))) / (2 * equi**1.7) * direction


def get_forces_3x3(
    positions: NDArray,
    labels: NDArray,
    atom1: NDArray,
    atom2: NDArray,
    displacements: NDArray,
    distance_matrix: NDArray,
    reference_frequency: float = 1.0,
) -> NDArray:
    """
    Calculate forces for a set of displacements.
    """
    force_xyz = []
    for d in displacements:
        force = get_force(
            positions,
            labels,
            atom1,
            atom2,
            d,
            distance_matrix,
            reference_frequency,
        )
        force_xyz.append(force)
    return np.array(force_xyz)
