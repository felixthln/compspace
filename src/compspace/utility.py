import numpy as np
import pandas as pd


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


def group_comps(comps: np.ndarray | pd.DataFrame,
                labels: list[str | list[str]] | None) -> tuple[np.ndarray | pd.DataFrame, list[str] | None]:

    """
    Sum up components which are grouped together in the labels and merge their names. A group is a nested list of
    labels, e.g. ['Cr', ['Cu', 'Ni'], 'Mo'] turns the four components Cr, Cu, Ni and Mo into the three components
    Cr, Cu+Ni and Mo. For a DataFrame the labels have to name its columns, the grouped components do not have to be
    neighbours and the resulting order follows the labels. For an array the labels are matched to the columns from
    left to right instead. Compositions without any group are returned unchanged.

    :param comps: (m, n) DataFrame or array of m compositions with n components
    :param labels: list of component labels, a nested list of labels groups those components together
    :returns: the grouped compositions and the merged labels
    """

    # Without labels or without any group there is nothing to sum up
    if labels is None or not any(isinstance(label, (list, tuple)) for label in labels):
        return comps, labels
    # Wrap the single components in a list as well to treat all labels the same way
    groups = [list(label) if isinstance(label, (list, tuple)) else [label] for label in labels]
    # Empty groups have no components to sum up
    if not all(groups):
        raise ValueError('Every group of labels must contain at least one component.')
    # Merge the names of the grouped components
    merged = ['+'.join(str(name) for name in group) for group in groups]
    # Flatten the labels to compare them against the columns of the compositions
    flat = [name for group in groups for name in group]

    # Group the columns by their name if a DataFrame is provided
    if isinstance(comps, pd.DataFrame):
        # Every label has to name a column and every column has to be used exactly once
        missing = [name for name in flat if name not in comps.columns]
        unused = [name for name in comps.columns if name not in flat]
        duplicates = [name for name in dict.fromkeys(flat) if flat.count(name) > 1]
        if missing or unused or duplicates:
            raise ValueError(f'The labels must name every column exactly once. Unknown labels: {missing}, columns '
                             f'without a label: {unused}, labels used more than once: {duplicates}.')
        # Sum up the columns of each group, the index of the compositions is kept
        return pd.DataFrame({name: comps[group].sum(axis=1) for name, group in zip(merged, groups)}), merged

    # Without column names the labels are matched to the columns from left to right
    if len(flat) != comps.shape[1]:
        raise ValueError(f'The labels must cover every column exactly once, got {len(flat)} labels for '
                         f'{comps.shape[1]} columns.')
    # Sum up the columns of each group, the groups follow each other from left to right
    columns, start = [], 0
    for group in groups:
        columns.append(comps[:, start:start + len(group)].sum(axis=1))
        start += len(group)
    return np.stack(columns, axis=1), merged


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
