from matplotlib.container import Container
from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D
import numpy as np
import pandas as pd

from .utility import bary_to_cart


class CompSpaceScatter(Container):

    # Artist to manipulate
    artists: PathCollection | Path3DCollection = None

    def __init__(self, artists: list[PathCollection | Path3DCollection], vertices: np.ndarray):
        super().__init__(artists)

        # Store the paths (the result of the ax.scatter call)
        self.artist = artists[0]
        # Store the vertices of the simplex for conversion of the coordinates
        self._vertices = vertices

    def set_offsets(self, arr: np.ndarray | pd.DataFrame):

        # Convert the input to a numpy array
        arr = arr.values if isinstance(arr, pd.DataFrame) else arr
        # Plot as compositions if possible
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr = bary_to_cart(arr, self._vertices)
        # Update the 2D or 3D offsets
        if isinstance(self.artist, Path3DCollection):
            self.artist._offsets3d = (arr[:, 0], arr[:, 1], arr[:, 2])
        elif isinstance(self.artist, PathCollection):
            self.artist.set_offsets(arr)
        # Enable method chaining
        return self

    # Forward everything else to the underlying collection object
    def __getattr__(self, name):
        # Pass either to PathCollection or Path3DCollection
        if isinstance(self.artist, PathCollection):
            return getattr(self.artist, name)
        elif isinstance(self.artist, Path3DCollection):
            return getattr(self.artist, name)


class CompSpaceLine(Container):

    # Artist to manipulate
    artist: Line2D | Line3D = None

    def __init__(self, artists: list[Line2D | Line3D], vertices: np.ndarray):
        super().__init__(artists)

        # Store the line (the result of the ax.plot call)
        self.artist = artists[0]
        # Store the vertices of the simplex for conversion of the coordinates
        self._vertices = vertices

    def set_data(self, arr: np.ndarray | pd.DataFrame):

        # Convert the input to a numpy array
        arr = arr.values if isinstance(arr, pd.DataFrame) else arr
        # Convert compositions to Cartesian if a 2D array with enough columns is provided
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr = bary_to_cart(arr, self._vertices)
        # Update the 2D or 3D line data
        if isinstance(self.artist, Line3D):
            self.artist.set_data_3d(arr[:, 0], arr[:, 1], arr[:, 2])
        elif isinstance(self.artist, Line2D):
            self.artist.set_data(arr[:, 0], arr[:, 1])
        # Enable method chaining
        return self

    # Forward everything else to the underlying line object
    def __getattr__(self, name):
        if isinstance(self.artist, Line2D):
            return getattr(self.artist, name)
        elif isinstance(self.artist, Line3D):
            return getattr(self.artist, name)
