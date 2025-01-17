from dynmat import load_unit_cell_from_toml

VERBOSE = True


def main():
    START_MSG = "Starting dynmat..."
    print(START_MSG) if VERBOSE else None

    dimensions = (10, 10, 10)
    u = load_unit_cell_from_toml("dynmat/examples/NaCl.toml").supercell(dimensions)

    print(u) if VERBOSE else None

    u.basis


if __name__ == "__main__":
    main()
