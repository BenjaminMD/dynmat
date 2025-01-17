from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class UnitCell:
    """
    Data class for crystallographic unit cell and atomic basis.
    Uses NumPy arrays as primary storage for atomic data.

    Attributes:
        lattice_lengths (Tuple[float, float, float]): (a, b, c) lengths
        lattice_angles (Tuple[float, float, float]): (α, β, γ) angles in degrees
        positions (NDArray): Nx3 array of fractional coordinates
        labels (NDArray): N-length array of atomic element labels
    """

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
        positions: Optional[NDArray] = None,
        labels: Optional[NDArray] = None,
        basis: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None,
    ):
        # Store lattice parameters
        self.lattice_lengths = lattice_lengths
        self._angles_deg = lattice_angles
        self._angles_rad = tuple(np.deg2rad(angle) for angle in lattice_angles)

        # Initialize atomic data from either positions+labels or basis
        if positions is not None and labels is not None:
            self._positions = np.asarray(positions, dtype=np.float64)
            self._labels = np.asarray(labels, dtype=str)
        elif basis is not None:
            self._labels = np.array([atom[0] for atom in basis], dtype=str)
            self._positions = np.array(
                [atom[1] for atom in basis],
                dtype=np.float64,
            )
        else:
            self._positions = np.empty((0, 3), dtype=np.float64)
            self._labels = np.empty(0, dtype=str)

        # Validate shapes
        if self._positions.shape[0] != self._labels.shape[0]:
            raise ValueError("Number of positions must match number of labels")
        if self._positions.shape[1] != 3:
            raise ValueError("Positions must be Nx3 array")

    @property
    def a(self) -> float:
        return self.lattice_lengths[0]

    @property
    def b(self) -> float:
        return self.lattice_lengths[1]

    @property
    def c(self) -> float:
        return self.lattice_lengths[2]

    @property
    def alpha(self) -> float:
        return self._angles_deg[0]

    @property
    def beta(self) -> float:
        return self._angles_deg[1]

    @property
    def gamma(self) -> float:
        return self._angles_deg[2]

    @property
    def alpha_rad(self) -> float:
        return self._angles_rad[0]

    @property
    def beta_rad(self) -> float:
        return self._angles_rad[1]

    @property
    def gamma_rad(self) -> float:
        return self._angles_rad[2]

    @property
    def positions(self) -> NDArray:
        """Get atomic positions array (read-only)"""
        return self._positions.copy()

    @property
    def labels(self) -> NDArray:
        """Get atomic labels array (read-only)"""
        return self._labels.copy()

    @property
    def basis(
        self,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Generate basis list format from internal arrays"""
        return [
            (label, tuple(pos))
            for label, pos in zip(self._labels, self._positions)
        ]

    def supercell(self, dimensions: Tuple[int, int, int]) -> "UnitCell":
        """
        Create a supercell by scaling the unit cell dimensions & atomic basis.

        Args:
            dimensions (Tuple[int, int, int]): Scaling (nx, ny, nz) for each axis.

        Returns:
            UnitCell: A new UnitCell object representing the supercell.
        """
        nx, ny, nz = dimensions
        if any(d < 0 for d in dimensions):
            raise ValueError("Supercell dimensions must be positive")

        # Scale lattice parameters
        new_lengths = (
            self.a * nx,
            self.b * ny,
            self.c * nz,
        )

        # Generate new atomic positions
        n_atoms = len(self._labels)
        n_cells = (
            (nx if nx > 0 else 1)
            * (ny if ny > 0 else 1)
            * (nz if nz > 0 else 1)
        )
        new_positions = np.empty((n_atoms * n_cells, 3), dtype=np.float64)
        new_labels = np.empty(n_atoms * n_cells, dtype=object)

        idx = 0
        for i in range(nx if nx > 0 else 1):
            for j in range(ny if ny > 0 else 1):
                for k in range(nz if nz > 0 else 1):
                    x_off = i if nx > 0 else 0
                    y_off = j if ny > 0 else 0
                    z_off = k if nz > 0 else 0
                    offset = np.array([x_off, y_off, z_off])

                    # Add atoms for this cell
                    for atom_idx in range(n_atoms):
                        new_positions[idx] = np.array(
                            self._positions[atom_idx] + offset
                        ) / [nx, ny, nz]
                        new_labels[idx] = self._labels[atom_idx]
                        idx += 1

        return UnitCell(
            lattice_lengths=new_lengths,
            lattice_angles=self._angles_deg,
            positions=new_positions,
            labels=new_labels,
        )

    def __repr__(self) -> str:
        return (
            f"UnitCell(\n"
            f"  lattice:"
            f"a={self.a:.3f}, "
            f"b={self.b:.3f}, "
            f"c={self.c:.3f}\n"
            f"  angles: alpha={self.alpha:.1f}, "
            f"beta={self.beta:.1f}, "
            f"gamma={self.gamma:.1f} deg\n"
            f"atoms: {len(self._labels)} "
            f"({', '.join(np.unique(self._labels))})\n"
            ")"
        )
