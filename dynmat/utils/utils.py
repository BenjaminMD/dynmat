from typing import List

from numpy import abs, array, dot, eye, isclose, minimum, sqrt, sum
from numpy.typing import NDArray


def periodic_distance(p1, p2, T=eye(3)):
    """
    Computes the periodic distance between two points in a unit cube,
    using a transformation matrix to convert fractto Cartesian coordinates.

    Parameters:
        p1 (array-like): Fractional coord of the first point.
        p2 (array-like): Fractional coord of the second point.
        T (array-like): 3x3 transformation matrix.

    Returns:
        float: The Cartesian periodic distance between the points.
    """
    p1 = array(p1) % 1
    p2 = array(p2) % 1

    p1 = p1.reshape(-1, 3)
    p2 = p2.reshape(-1, 3)

    # Compute delta in fractional space
    delta = abs(p2 - p1)
    delta = minimum(delta, 1 - delta)  # Wrap around the boundary

    # Transform fractional deltas to Cartesian
    delta_cartesian = dot(delta, T.T)

    # Compute Cartesian distance
    distance = sqrt(sum(delta_cartesian**2, axis=1))

    return distance


def periodic_distance_mat(coords):
    """
    Computes the periodic distance between all pairs of points in a unit cube.

    Parameters:
        positions: List of [x, y, z] arrays or array of shape (N, 4)

    Returns:
        distances: Array of shape (N_points, N_points) of pairwise distances
        where distances[i,j] gives distance between atom i and j
    """
    # Convert it to numpy array if not already
    coords = array(coords)

    # Apply periodic boundary conditions to coordinates
    coords = coords % 1

    # Reshape coordinates for broadcasting
    coords1 = coords.reshape(-1, 1, 3)  # Shape: (N, 1, 3)
    coords2 = coords.reshape(1, -1, 3)  # Shape: (1, N, 3)

    # Calculate periodic distances
    delta = abs(coords2 - coords1)  # Shape: (N, N, 3)
    delta = minimum(delta, 1 - delta)  # Wrap around boundary
    distances = sqrt(sum(delta**2, axis=2))  # Shape: (N, N)

    return distances


# def is_nearest_neighbors(
#     reference_atom: int, query_atom: int, distance_matrix: NDArray, order=0
# ) -> bool:
#     """
#     Determines if the query atom is nearest neighbors of the reference atom.
#
#     Parameters:
#         reference_atom: Index of the atom to find neighbors for
#         query_atom: Index of the atom to check if it's a nearest neighbor
#         distance_matrix: Pre-computed matrix of distances between all atoms
#
#     Returns:
#         bool: True if query_atom is one of reference_atom's nearest neighbors
#     """
#     distances_to_reference = distance_matrix[reference_atom, :]
#     # Exclude self-distance by taking distances greater than 0
#     neighbor_distances = distances_to_reference[distances_to_reference > 0]
#
#     if len(neighbor_distances) == 0:
#         return False
#
#     shortest_distance = min(neighbor_distances)
#     query_distance = distance_matrix[reference_atom, query_atom]
def get_unique_neighbor_distances(
    distances: NDArray, tolerance: float = 1e-3
) -> List[float]:
    """
    Get sorted unique distances from array, accounting for numerical precision.

    Parameters:
        distances: Array of distances
        tolerance: Tolerance for considering distances as equal

    Returns:
        List of unique distances in ascending order
    """
    # Remove zeros and sort distances
    positive_distances = sorted(distances[distances > 0])
    if not positive_distances:
        return []

    # Initialize with first distance
    unique_distances = [positive_distances[0]]

    # Add distances that differ by more than tolerance
    for dist in positive_distances[1:]:
        if not isclose(dist, unique_distances[-1], rtol=tolerance):
            unique_distances.append(dist)

    return unique_distances


def is_nearest_neighbors(
    reference_atom: int,
    query_atom: int,
    distance_matrix: NDArray,
    order: int = 0,
    tolerance: float = 1e-3,
) -> bool:
    """
    Determines if query atom is neighbor of the given order of reference atom.

    Parameters:
        reference_atom: Index of the atom to find neighbors for
        query_atom: Index of the atom to check if it's a neighbor
        distance_matrix: Pre-computed matrix of distances between all atoms
        order: Neighbor order (0=nearest, 1=next-nearest, etc.)
        tolerance: Numerical tolerance for comparing distances

    Returns:
        bool: True if query_atom is a neighbor of the specified order
    """
    distances_to_reference = distance_matrix[reference_atom, :]
    # Get unique neighbor distances (shells)
    unique_distances = get_unique_neighbor_distances(
        distances_to_reference, tolerance
    )

    # Check if we have enough shells for the requested order
    if order >= len(unique_distances):
        return False

    # Get the target distance for this order
    target_distance = unique_distances[order]

    # Check if query_atom is at this distance
    query_distance = distance_matrix[reference_atom, query_atom]
    return isclose(query_distance, target_distance, rtol=tolerance)
