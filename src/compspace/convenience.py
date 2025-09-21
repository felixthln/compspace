import pandas as pd
import matplotlib.pyplot as plt


def plot_on_comp_space(compositions: pd.DataFrame | list[pd.DataFrame], *args, always_2d: bool = False, **kwargs):

    """
    Convenience function to plot data on a composition space. The function plots in 3D if possible, otherwise in 2D.

    :param compositions: A DataFrame or list of DataFrames containing the compositions to plot
    :param always_2d: set if the plot should always be in 2D, even if 3D is possible
    """

    # Wrap the data in a list if a single DataFrame is provided
    compositions = [compositions] if isinstance(compositions, pd.DataFrame) else compositions
    # Plot in 3D if possible, otherwise in 2D
    projection = 'compspace3D' if compositions[0].shape[1] in [4, 5] and not always_2d else 'compspace2D'
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': projection})
    # Plot the data
    for comp in compositions:
        ax.scatter(comp, *args, **kwargs)
    # Show the plot
    plt.show()
