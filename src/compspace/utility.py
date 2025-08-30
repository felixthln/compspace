import numpy as np


def bary_to_cart(bary: np.ndarray, vertices: np.ndarray) -> np.ndarray:

    """
    Convert barycentric compositions to 2D Cartesian points.

    :param bary: (m, n) array of m barycentric compositions with n components
    :param vertices: (n, 2) array of n Cartesian vertex coordinates
    """

    # Normalize rows robustly (accept 1.0, 100.0, slightly noisy sums, etc.)
    rs = bary.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    bary_norm = bary / rs
    # Perform the conversion and return
    return bary_norm @ vertices


def remove_handles(handles, keyword: str = None) -> None:

    """
    Method to remove artists stored in a list and clear the list.

    :param handles: list of artists all artists
    :param keyword: keyword for an attribute to filter the artists by
    :return:
    """

    # Filter the artists by the keyword if provided
    handles = [h for h in handles if getattr(h, keyword, False) is not True] if keyword else handles
    # Remove artists, try-except in case already removed by garbage collection
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
    handles.clear()
