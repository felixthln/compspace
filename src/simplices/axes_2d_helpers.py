import numpy as np


def gen_vertices(n: int) -> np.ndarray:

    """
    Create an n-gon (equal edge lengths) in counter clock wise order, then uniformly scale & center it into a standard
    plotting frame: x ∈ [0, 1]  and  y ∈ [0, sqrt(3)/2]. The identical frame across all n keeps padding, label offsets,
    and tick lengths consistent for ternary/quaternary/quinary plots.
    """

    # Everything below ternary is not supported
    if n < 3:
        raise ValueError('n ≥ 3 required.')
    # Start with a unit-circle regular n-gon centered at origin (equal edges)
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def bary_to_cart(bary: np.ndarray, vertices: np.ndarray) -> np.ndarray:

    """
    Convert barycentric compositions to 2D Cartesian points.
    """

    # Validate inputs
    if bary.ndim != 2 or vertices.ndim != 2 or vertices.shape != (bary.shape[1], 2):
        raise ValueError("Shapes incompatible: B(m, n) with V(n, 2) required.")
    # Normalize rows robustly (accept 1.0, 100.0, slightly noisy sums, etc.)
    rs = bary.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    bary_norm = bary / rs
    # Perform the conversion and return
    return bary_norm @ vertices
