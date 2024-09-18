def get_cube_inertia_matrix(mass, x, y, z):
    """Given mass and dimensions of a cube return intertia matrix.
    :return: ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz
    From https://www.wolframalpha.com/input/?i=moment+of+inertia+cube"""
    ixx = (1.0 / 3.0) * (y**2 + z**2) * mass
    iyy = (1.0 / 3.0) * (x**2 + z**2) * mass
    izz = (1.0 / 3.0) * (x**2 + y**2) * mass
    ixy = (1.0 / 4.0) * x * y * mass
    ixz = (1.0 / 4.0) * x * z * mass
    iyz = (1.0 / 4.0) * y * z * mass
    # return [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    return [ixx, iyy, izz, ixy, ixz, iyz]

print(get_cube_inertia_matrix(5, 1.2, 0.2, 0.05))