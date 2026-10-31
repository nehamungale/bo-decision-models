import numpy as np

# Branin on [-5, 10] x [0, 15]; global minimum ~0.397887
BRANIN_BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])
BRANIN_MIN = 0.397887


def branin(x: np.ndarray) -> float:
    x1, x2 = x
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)
