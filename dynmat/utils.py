from numpy import abs, array, minimum, sqrt, sum


def periodic_distance(p1, p2):
    """
    Computes the periodic distance between two points in a unit cube (0 to 1 in all dimensions).

    Parameters:
        p1 (array-like): Coordinates of the first point (e.g., [x1, y1, z1]).
        p2 (array-like): Coordinates of the second point (e.g., [x2, y2, z2]).

    Returns:
        float: The periodic distance between the points.
    """
    p1 = array(p1)
    p2 = array(p2)

    delta = abs(p2 - p1)
    delta = minimum(delta, 1 - delta)  # Wrap around the boundary
    distance = sqrt(sum(delta**2))  # Euclidean distance with PBC

    return distance
