import numpy as np


def plot_force_constant_matrix(fc_matrix, atomic_labels):
    """
    Plot force constant matrix with 3x3 block grid lines and atom labels.

    Parameters:
        fc_matrix: The force constant matrix (3N x 3N)
        atomic_labels: List of atomic symbols for each atom
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for visualization")
    n_atoms = len(atomic_labels)
    n = 3 * n_atoms

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the matrix
    vmax = 0.8 * np.max(np.abs(fc_matrix))
    im = ax.imshow(
        fc_matrix, cmap="PuOr", aspect="equal", vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im)

    # Add grid lines to separate 3x3 blocks
    for i in range(n_atoms):
        ax.axhline(y=i * 3 - 0.5, color="black", linewidth=2)
        ax.axvline(x=i * 3 - 0.5, color="black", linewidth=2)

    # Create labels for each coordinate of each atom
    coord_labels = []
    for atom in atomic_labels:
        coord_labels.extend([f"{atom}-x", f"{atom}-y", f"{atom}-z"])

    # Set ticks at the center of each coordinate
    ticks = np.arange(n)

    # Add labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(coord_labels, rotation=45, ha="right")
    ax.set_yticklabels(coord_labels)

    # Add title
    plt.title("Force Constant Matrix")

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    return fig, ax
