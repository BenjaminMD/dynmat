from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

NPF64 = NDArray[np.float64]
NPS = NDArray[np.str_]


"""
Todo:
    - [] Typing of positions and labels is not correct.
    - [] Strucutre of the dataclass is probably not the best, i.e.
      declaration order and such is unclear to me.
"""


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
        basis (List[Tuple[float, float, float]]): atomic fract coord.
    """

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    basis: List[Tuple[str, Tuple[float, float, float]]] = field(
        default_factory=list
    )
    positions: np.ndarray = field(default_factory=NPF64)
    labels: np.ndarray = field(default_factory=NPS)

    def __init__(
        self,
        lattice_lengths: Tuple[float, float, float] = (
            1.0,
            1.0,
            1.0,
        ),
        lattice_angles: Tuple[float, float, float] = (
            90.0,
            90.0,
            90.0,
        ),
        basis: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None,
    ):
        self.a, self.b, self.c = lattice_lengths
        self.alpha, self.beta, self.gamma = [
            np.deg2rad(i) for i in lattice_angles
        ]
        self.basis = basis if basis is not None else []
        self.lattice_lengths = lattice_lengths
        self.lattice_angles = lattice_angles
        self.positions: NPF64 = np.array([atom[1] for atom in self.basis])
        self.labels: NPS = np.array([atom[0] for atom in self.basis])

    def supercell(self, dimensions: Tuple[int, int, int]) -> "UnitCell":
        """
        Create a supercell by scaling the unit cell dimensions & atomic basis.

        Args:
            scale (Tuple[int, int, int]): Scaling (nx, ny, nz) for each axis.

        Returns:
            UnitCell: A new UnitCell object representing the supercell.
        """
        nx, ny, nz = dimensions

        # Handle zero scaling by setting the new lattice vector length to zero
        new_a = self.a * nx
        new_b = self.b * ny
        new_c = self.c * nz

        new_basis = []
        for x_shift in range(nx if nx > 0 else 1):  # Avoid zero range
            for y_shift in range(ny if ny > 0 else 1):  # Avoid zero range
                for z_shift in range(nz if nz > 0 else 1):  # Avoid zero range
                    for label, atom in self.basis:
                        new_x = (atom[0] + x_shift) / nx if nx > 0 else atom[0]
                        new_y = (atom[1] + y_shift) / ny if ny > 0 else atom[1]
                        new_z = (atom[2] + z_shift) / nz if nz > 0 else atom[2]
                        new_basis.append((label, (new_x, new_y, new_z)))

        self.positions: NPF64 = np.array([atom[1] for atom in new_basis])
        self.labels: NPS = np.array([atom[0] for atom in new_basis])

        alpha, beta, gamma = [
            np.rad2deg(i) for i in (self.alpha, self.beta, self.gamma)
        ]
        return UnitCell(
            lattice_lengths=(new_a, new_b, new_c),
            lattice_angles=(alpha, beta, gamma),
            basis=new_basis,
        )

    def __repr__(self) -> str:
        alpha, beta, gamma = [
            np.rad2deg(i) for i in (self.alpha, self.beta, self.gamma)
        ]
        return (
            "This is a crystallographic unit cell with the following\n"
            f"UnitCell(a={self.a}, b={self.b}, c={self.c},\n"
            f"alpha={alpha}, beta={beta}, gamma={gamma},\n"
            f"basis={self.basis})"
        )
