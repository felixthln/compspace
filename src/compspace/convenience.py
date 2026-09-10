import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .utility import group_comps


def plot_on_comp_space(compositions: pd.DataFrame | np.ndarray | list[pd.DataFrame | np.ndarray],
                       *args,
                       labels: list[str | list[str]] | None = None, always_2d: bool = False,
                       **kwargs) -> tuple[Figure, Axes | Axes3D]:

    """
    Convenience function to plot data on a composition space. The function plots in 3D if possible, otherwise in 2D.

    :param compositions: A DataFrame or list of DataFrames containing the compositions to plot
    :param args: additional positional arguments passed to the scatter function
    :param labels: list of labels for the components, if none are provided, the labels will be inferred from the
    DataFrame columns. Nesting labels, e.g. ['Cr', ['Cu', 'Ni']], sums up those components and plots them as a
    single axis named 'Cu+Ni'
    :param always_2d: set if the plot should always be in 2D, even if 3D is possible
    """

    # Wrap the data in a list if a single DataFrame is provided
    compositions = [compositions] if not isinstance(compositions, list) else compositions

    # Check that all DataFrames or numpy arrays have the same number of columns
    counts = [c.shape[1] for c in compositions]
    if not all(c == counts[0] for c in counts):
        raise ValueError('All sets of compositions must have the same number of columns.')
    # Check that all DataFrames have the same columns
    cols = [comp.columns.tolist() for comp in compositions if isinstance(comp, pd.DataFrame)]
    if not all(c == cols[0] for c in cols):
        raise ValueError('All composition sets must have the same column headers.')

    # Sum up components which are grouped together in the labels, this reduces the number of components to plot
    grouped = [group_comps(comp, labels) for comp in compositions]
    compositions, labels = [comp for comp, _ in grouped], grouped[0][1]

    # Plot in 3D if possible, otherwise in 2D
    projection = 'compspace3D' if compositions[0].shape[1] in [4, 5] and not always_2d else 'compspace2D'
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': projection})
    # Plot the data
    for comp in compositions:
        ax.scatter(comp, *args, labels=labels, **kwargs)
    # Return the figure and axis
    return fig, ax
