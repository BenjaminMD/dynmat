from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class UnitCell:
    """
    Data class for crystallographic unit cell and atomic basis.

    Attributes:
        a (float): Length of the unit cell edge a.
        b (float): Length of the unit cell edge b.
        c (float): Length of the unit cell edge c.
        alpha (float): Angle α in degrees (default is 90.0).
        beta (float): Angle β in degrees (default is 90.0).
        gamma (float): Angle γ in degrees (default is 90.0).
        atomic_basis (List[Tuple[float, float, float]]): atomic fract coord.
    """

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    atomic_basis: List[Tuple[str, float, float, float]] = field(
        default_factory=list
    )

    def __init__(
        self,
        lattice_lengths: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        lattice_angles: Tuple[float, float, float] = (90.0, 90.0, 90.0),
        atomic_basis: Optional[List[Tuple[str, float, float, float]]] = None,
    ):
        self.a, self.b, self.c = lattice_lengths
        self.alpha, self.beta, self.gamma = [
            np.deg2rad(i) for i in lattice_angles
        ]
        self.atomic_basis = atomic_basis if atomic_basis is not None else []

    def make_supercell(self, scale: Tuple[int, int, int]) -> "UnitCell":
        """
        Create a supercell by scaling the unit cell dimensions & atomic basis.

        Args:
            scale (Tuple[int, int, int]): Scaling (nx, ny, nz) for each axis.

        Returns:
            UnitCell: A new UnitCell object representing the supercell.
        """
        nx, ny, nz = scale

        # Handle zero scaling by setting the new lattice vector length to zero
        new_a = self.a * nx
        new_b = self.b * ny
        new_c = self.c * nz

        new_atomic_basis = []
        for x_shift in range(nx if nx > 0 else 1):  # Avoid zero range
            for y_shift in range(ny if ny > 0 else 1):  # Avoid zero range
                for z_shift in range(nz if nz > 0 else 1):  # Avoid zero range
                    for atom in self.atomic_basis:
                        new_x = (atom[1] + x_shift) / nx if nx > 0 else atom[1]
                        new_y = (atom[2] + y_shift) / ny if ny > 0 else atom[2]
                        new_z = (atom[3] + z_shift) / nz if nz > 0 else atom[3]
                        new_atomic_basis.append((atom[0], new_x, new_y, new_z))

        alpha, beta, gamma = [
            np.rad2deg(i) for i in (self.alpha, self.beta, self.gamma)
        ]
        return UnitCell(
            lattice_lengths=(new_a, new_b, new_c),
            lattice_angles=(alpha, beta, gamma),
            atomic_basis=new_atomic_basis,
        )

    def __repr__(self) -> str:
        alpha, beta, gamma = [
            np.rad2deg(i) for i in (self.alpha, self.beta, self.gamma)
        ]
        return (
            "This is a crystallographic unit cell with the following "
            f"UnitCell(a={self.a}, b={self.b}, c={self.c}, "
            f"alpha={alpha}, beta={beta}, gamma={gamma}, "
            f"atomic_basis={self.atomic_basis})"
        )
