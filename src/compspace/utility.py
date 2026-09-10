import numpy as np


def bary_to_cart(bary: np.ndarray, vertices: np.ndarray = None) -> np.ndarray:

    """
    Convert barycentric compositions to 2D or 3D Cartesian points.

    :param bary: (m, n) array of m barycentric compositions with n components
    :param vertices: (n, 2 or 3) array of n Cartesian vertex coordinates. If not provided, the vertices are inferred
    from the input compositions and mapped on a 2D surface
    """

    # Generate the vertices if not defined
    if vertices is None:
        from .axes_2d import _gen_vertices
        vertices = _gen_vertices(n=bary.shape[1])
    # Normalize rows robustly (accept 1.0, 100.0, slightly noisy sums, etc.)
    rs = bary.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    bary_norm = bary / rs
    # Perform the conversion and return
    return bary_norm @ vertices


def remove_handles(handles: list, keyword: str = None) -> None:

    """
    Method to remove artists stored in a list and drop them from that list.

    :param handles: list of artists, is modified in place
    :param keyword: name of an artist attribute marking the artists to remove, e.g. '_is_prim_label'. If not
    provided, all artists are removed
    :return:
    """

    # Select the artists to remove by the keyword if provided, otherwise remove all of them
    remove = [h for h in handles if getattr(h, keyword, False) is True] if keyword else list(handles)
    # Keep the artists which were filtered out, compare by identity as artists are not hashable by content
    keep = [h for h in handles if not any(h is r for r in remove)]
    # Remove artists, try-except in case already removed by garbage collection
    for h in remove:
        try:
            h.remove()
        except Exception:
            pass
    # Update the original list in place so the caller sees the remaining artists
    handles[:] = keep
