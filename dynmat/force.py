from typing import Tuple

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


def calculate_force_constant(mass1: float,
                             mass2: float,
                             reference_frequency: float = 1.0) -> float:
    """
    Calculate the force constant based on reduced mass and a reference frequency.

    Parameters:
        mass1: Mass of first atom in atomic mass units (u)
        mass2: Mass of second atom in atomic mass units (u)
        reference_frequency: Reference vibrational frequency in arbitrary units (default: 1.0)

    Returns:
        float: Force constant k = μω², where μ is reduced mass and ω is angular frequency
    """
    # Calculate reduced mass
    reduced_mass = (mass1 * mass2) / (mass1 + mass2)

    # Convert reference frequency to angular frequency (ω = 2πν)
    angular_frequency = 2 * np.pi * reference_frequency

    # Calculate force constant k = μω²
    force_constant = reduced_mass * angular_frequency**2

    return force_constant


def get_atomic_mass(atom_label: str) -> float:
    """
    Get the atomic mass of an element in atomic mass units (u).

    Parameters:
        atom_label: Chemical symbol of the element (e.g., 'H', 'C', 'O')

    Returns:
        float: Atomic mass in u
    """
    return element(atom_label).mass


def get_harmonic_force(reference_pos: NDArray,
                       reference_labels: NDArray,
                       atom1_pos: NDArray,
                       atom2_pos: NDArray,
                       displacement: NDArray,
                       atom1_label: str,
                       atom2_label: str,
                       distance_matrix: NDArray,
                       reference_frequency: float = 1.0) -> Tuple[NDArray, float]:
    """
    Calculate the harmonic force between two atoms after displacement using mass-dependent force constant.

    Parameters:
        reference_pos: Reference positions of all atoms (N, 3)
        reference_labels: Atomic labels for all positions (N,)
        atom1_pos: Position of first atom (3,)
        atom2_pos: Position of second atom (3,)
        displacement: Displacement vector to apply to atom2 (3,)
        atom1_label: Chemical symbol of first atom
        atom2_label: Chemical symbol of second atom
        distance_matrix: Matrix of equilibrium distances between atoms
        reference_frequency: Reference frequency for force constant calculation (default: 1.0)

    Returns:
        Tuple containing:
        - NDArray: Force vector (3,) acting on atom2 due to displacement
        - float: Force constant used in calculation
    """
    # Convert inputs to numpy arrays and ensure correct shapes
    atom1_pos = np.asarray(atom1_pos, dtype=np.float64).reshape(3)
    atom2_pos = np.asarray(atom2_pos, dtype=np.float64).reshape(3)
    displacement = np.asarray(displacement, dtype=np.float64).reshape(3)

    # Find atom indices in the reference positions
    atom1_mask = np.all(np.isclose(reference_pos, atom1_pos), axis=1)
    atom2_mask = np.all(np.isclose(reference_pos, atom2_pos), axis=1)

    atom1_idx = np.where(atom1_mask)[0]
    atom2_idx = np.where(atom2_mask)[0]

    # Get equilibrium distance from the distance matrix
    equilibrium_distance = distance_matrix[atom1_idx[0], atom2_idx[0]]

    # Calculate new distance after displacement
    displaced_pos = atom2_pos + displacement
    direction = calculate_periodic_direction(atom1_pos, displaced_pos)
    current_distance = periodic_distance(atom1_pos, displaced_pos)

    # Get atomic masses and calculate force constant
    mass1 = get_atomic_mass(atom1_label)
    mass2 = get_atomic_mass(atom2_label)
    force_constant = calculate_force_constant(
        mass1, mass2, reference_frequency)

    # Calculate force (F = -k(r - r_0) * direction)
    force_magnitude = force_constant * \
        (equilibrium_distance - current_distance)
    force_vector = force_magnitude * direction

    return force_vector, force_constant

