import numpy as np
import pandas as pd
from itertools import combinations
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .utility import bary_to_cart, remove_handles
from .containers import CompSpaceScatter


def _gen_vertices(n_dim: int) -> np.ndarray:

    """
    Creates a tetrahedron for quaternary and a pyramid for quinary compositions, then centers them at the origin of the
    cartesian coordinate system.
    """

    # Everything below ternary is not supported
    if n_dim < 4 or n_dim > 5:
        raise ValueError('4 ≤ n ≤ 5 required.')
    # Define the vertices of and the base depending on the number of dimensions
    verts, base = None, None
    if n_dim == 4:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]])
        base = np.array([0.5, np.sqrt(3) / 6, np.sqrt(2 / 3) / 2])
    elif n_dim == 5:
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, np.sqrt(0.5)]])
        base = np.array([0.5, 0.5, np.sqrt(0.5) / 2])
    # Return the translated vertices
    return verts - base


class CompSpace3DAxes(Axes3D):

    # Name used when registering the projection
    name = 'compspace3D'

    # Label and tick parameters
    _label_space: float = 0.25

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Number of components/dimensions
        self._n_dim: int = 4
        # Primary (vertices) label parameters
        self._labels: list[str] | None = None
        self._label_handles: list = []
        # For keeping track of background artists
        self._bg_handles: list = []
        # Vertices, by default plot a tetrahedron
        self._vertices: np.ndarray = _gen_vertices(4)
        # Whether we have plotted any data yet
        self._has_data: bool = False

        # By default, (without any data), plot a tetrahedron
        self._apply_base_axes_style()
        self._draw_background()

    def _apply_base_axes_style(self):

        """
        Hides the cartesian spines and ticks, sets equal aspect ratio.

        :return:
        """

        # Set equal aspect ratio, hide axes
        self.set_aspect('equal')
        self.set_axis_off()
        # Get the min and max of the vertices of each dimension
        min_, max_ = np.min(self._vertices, axis=0), np.max(self._vertices, axis=0)
        # The largest difference between min and max will be used to set the limits of all axes
        diff = np.max(max_ - min_) / 2
        self.set_xlim(-diff, diff)
        self.set_ylim(-diff, diff)
        self.set_zlim(-diff, diff)

    def _draw_vertices(self):

        """
        Draws the vertices for the 3D simplex

        :returns: numpy array with the start and end positions of the lines
        """

        # Get all combinations in order to get the start and finish coordinates of the lines connecting the vertices
        axes = np.array(list(combinations(self._vertices, 2)))
        self._bg_handles.append(
            self.add_collection(Line3DCollection(axes, colors='black', linewidth=1.2, zorder=0))
        )

    def _draw_background(self) -> None:

        """
        Redraws the polygon outline and vertex labels.

        :return:
        """

        # Remove all previously drawn background artists
        remove_handles(self._bg_handles)
        # Generate the axes from the vertices
        self._draw_vertices()

    def _draw_labels(self) -> None:

        """
        Draw the primary (vertex) labels slightly outside the simplex along an averaged outward normal.
        """

        # Incorporate the space by calculating a vector from the center to the vertices and extending it by the spacing
        pos = (1 + self._label_space) * self._vertices
        # Plot the text on the vertices, silence a wrong pycharm warning
        for i in range(pos.shape[0]):
            # Create the text artist
            txt = self.text(pos[i, 0], pos[i, 1], pos[i, 2], self._labels[i], ha='center', va='center',
                            weight='bold', fontsize=10, zorder=0)
            self._label_handles.append(txt)

    def _redraw_background(self) -> None:

        """
        Replace the simplex when the detected number of components differs from the current background n.
        """

        # Calculate the new vertices
        self._vertices = _gen_vertices(self._n_dim)
        # Redraw the new background
        self._apply_base_axes_style()
        self._draw_background()

    def _redraw_labels(self) -> None:

        """
        Redraw vertex labels.
        """

        # Remove all previously drawn label artists
        remove_handles(self._label_handles)
        # Draw the vertex labels again
        self._draw_labels()

    def scatter(self, comps: np.ndarray | pd.DataFrame, *args, labels: list[str] = None,
                **kwargs) -> CompSpaceScatter:

        # Convert the compositions to a numpy array if a DataFrame is provided, store the column names as labels
        labels = comps.columns.to_list() if isinstance(comps, pd.DataFrame) else labels
        comps = comps.values if isinstance(comps, pd.DataFrame) else comps
        # Get the number of components from the compositions
        n_dim = comps.shape[1]
        # Raise error if n is out of bounds
        if n_dim < 4 or n_dim > 5:
            raise ValueError('Supported number of components is 4 and 5.')
        # Make sure the provided labels are valid
        if labels is not None and len(labels) != n_dim:
            raise ValueError(f'The provided labels must have the same length as the data n={n_dim}.')
        # If this is the first data or n changed, (re)build background
        if (not self._has_data) and (n_dim != self._n_dim):
            self._n_dim = n_dim
            self._redraw_background()
        # Otherwise update the vertex labels if provided
        if labels is not None and labels != self._labels:
            self._labels = labels
            self._redraw_labels()
        # Convert barycentric rows to XY and call the base Axes.scatter
        cart = bary_to_cart(comps, self._vertices)
        self._has_data = True
        # Forward the scatter call to the parent class
        paths = super().scatter(cart[:, 0], cart[:, 1], cart[:, 2], *args, **kwargs)
        # Wrap the collection path in a container to allow updating the data
        return CompSpaceScatter([paths], self._vertices)
