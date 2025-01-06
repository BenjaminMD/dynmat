from typing import Tuple

from numpy import array, cos, float64, sin, sqrt
from numpy.typing import NDArray

NPF64 = NDArray[float64]


def fractional_to_cartesian_transform(
    lattice_lengths: Tuple[float, float, float],
    lattice_angles: Tuple[float, float, float]
) -> NPF64:
    """
    Computes the transformation matrix to convert fractional
    coordinates to Cartesian coordinates in a crystallographic context.

    Parameters:
        lattice_lengths (Tuple[float, float, float]): lengths (a, b, c).
        lattice_angles (Tuple[float, float, float]): angles (α, β, γ) in rad.

    Returns:
        NPF64: The 3x3 transformation matrix.
    """
    a, b, c = lattice_lengths
    alpha, beta, gamma = lattice_angles

    # Calculate intermediate terms for the transformation matrix
    z_component = sqrt(
        1 + 2 * cos(alpha) * cos(beta) * cos(gamma)
        - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2
    ) / sin(gamma)

    y_component = (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)

    # Construct the transformation matrix
    row1 = [a, 0, 0]
    row2 = [b * cos(gamma), b * sin(gamma), 0]
    row3 = [c * cos(beta), c * y_component, c * z_component]

    transformation_matrix = array([row1, row2, row3])

    return transformation_matrix.T
