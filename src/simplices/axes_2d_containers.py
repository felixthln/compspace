from matplotlib.container import Container
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

from .axes_2d_helpers import bary_to_cart


class Simplices2DScatter(Container):

    def __init__(self, artists: list[PathCollection], vertices: np.ndarray):
        super().__init__(artists)

        # Store the paths (the result of the ax.scatter call)
        self.artists = artists[0]
        # Store the vertices of the simplex for conversion of the coordinates
        self._vertices = vertices

    def set_offsets(self, arr: np.ndarray | pd.DataFrame):

        # Convert the input to a numpy array
        arr = np.asarray(arr)
        # Either plot as compositions or as xy
        if arr.ndim == 2 and arr.shape[1] >= 3:
            arr = bary_to_cart(arr, self._vertices)
        # Update the offsets
        self.paths.set_offsets(arr)
        # Enable method chaining
        return self

    # Pass-throughs for convenience
    def __getattr__(self, name):
        return getattr(self.paths, name)
