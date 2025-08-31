import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from .utility import bary_to_cart, remove_handles
from .containers import CompSpaceScatter


def _gen_vertices(n: int) -> np.ndarray:

    """
    Create an n-gon (equal edge lengths) in counter clock wise order, then uniformly scale & center it into a
    standard plotting frame: x ∈ [0, 1]  and  y ∈ [0, sqrt(3)/2]. The identical frame across all n keeps padding,
    label offsets, and tick lengths consistent for ternary/quaternary/quinary plots.
    """

    # Everything below ternary is not supported
    if n < 3:
        raise ValueError('n ≥ 3 required.')
    # Start with a unit-circle regular n-gon centered at origin (equal edges)
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


class CompSpace2DAxes(Axes):

    # Name used when registering the projection
    name = 'compspace2D'

    # Label and tick parameters
    _prim_label_space: float = 0.3
    _sec_label_space: float = 0.25
    _tick_len: float = 0.02
    _tick_label_space: float = 0.08

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Number of components/dimensions
        self._n_dim: int = 3
        # Primary (vertices) and secondary (edge) label parameters
        self._prim_labels: list[str] | None = None
        self._sec_labels: list[str] | None = None
        self._label_handles: list = []
        # Tick parameters
        self._tick_positions = np.linspace(0, 1, 11)
        self._show_ticks: bool = True
        # Grid parameters
        self._grid_pos = np.linspace(0, 1, 11)
        self._show_grid: bool = True
        # For keeping track of background artists
        self._bg_handles: list = []
        # Vertices, default to ternary
        self._vertices: np.ndarray = _gen_vertices(3)
        # Whether we have plotted any data yet
        self._has_data: bool = False

        # Draw the initial ternary background
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

    def _draw_vertices(self):

        # Generate the axes from the vertices
        axes = np.array(list(combinations(self._vertices, 2)))
        self._bg_handles.append(
            self.add_collection(LineCollection(axes, colors='black', linewidth=1.2, zorder=0))
        )

    def _draw_prim_labels(self) -> None:

        """
        Draw the primary (vertex) labels slightly outside the polygon along an averaged outward normal.
        """

        # Incorporate the space by calculating a vector from the center to the vertices and extending it by the spacing
        pos = (1 + self._prim_label_space) * self._vertices
        # Plot the text on the vertices, silence a wrong pycharm warning
        for i in range(pos.shape[0]):
            # Create the text artist, create a custom attribute to identify it later
            txt = self.text(pos[i, 0], pos[i, 1], self._prim_labels[i], ha='center', va='center', weight='bold',
                            fontsize=10, zorder=0)
            txt._is_prim_label = True
            self._label_handles.append(txt)
        # Extend the limits based on the label positions
        self.set_xlim(np.min(pos[:, 0]), np.max(pos[:, 0]))
        self.set_ylim(np.min(pos[:, 1]), np.max(pos[:, 1]))

    def _gen_grid(self, positions: list[float] | np.ndarray) -> np.ndarray:

        # Calculate consecutive edge triplets of the polygon
        idx = np.arange(self._n_dim)
        triplets = np.stack([idx, (idx + 1) % self._n_dim, (idx + 2) % self._n_dim], axis=1)
        # Iterate over each edge pair
        segments = []
        for trip in triplets:
            for pos, pos_r in zip(positions, np.flip(positions)):
                # Each edge pair (i,j), (j,k) maps to a segment on the iso-fraction polyline:
                # X(t) = level*V[j] + (1-level) * ((1-t)*V[i] + t*V[k]), t in [0,1]
                # Add it as a single segment between the two endpoints (t=0 and t=1)
                i, j, k = trip
                s = pos * self._vertices[i] + (1 - pos) * self._vertices[j]  # start at t=0
                e = pos_r * self._vertices[j] + (1 - pos_r) * self._vertices[k]  # end at t=1
                segments.append(np.stack([s, e], axis=0))
        # Combine all segments into a single array
        return np.stack(segments, axis=0)

    def _draw_grid(self) -> None:

        # Generate the grid
        grid = self._gen_grid(self._grid_pos)
        # If the grid is enabled, add it as a LineCollection
        self._bg_handles.append(
            self.add_collection(LineCollection(grid, colors='black', linewidths=0.6, alpha=0.35, zorder=0))
        )

    @staticmethod
    def _gen_ticks(grid: np.ndarray, tick_len: float) -> np.ndarray:

        # Split the grid into start and end points
        p1, p2 = grid[:, 0, :], grid[:, 1, :]
        # Calculate direction vectors and unit directions
        d = p2 - p1
        # Manage division warnings for zero-length segments
        with np.errstate(invalid='ignore', divide='ignore'):
            u = d / np.linalg.norm(d, axis=1, keepdims=True)
        # Replace a nan row with the following row, this happens at the corners of the polygon
        mask = np.isnan(u).all(axis=1)
        u[mask] = u[np.where(mask)[0] + 1]
        # Extend the end points by the tick length
        p3 = p2 + u * tick_len
        # Stack the original end points and the extended points to form tick segments
        return np.stack([p2, p3], axis=1)

    def _draw_ticks(self) -> np.ndarray:

        # Generate the ticks from a grid
        grid = self._gen_grid(self._tick_positions)
        ticks = self._gen_ticks(grid, self._tick_len)
        # Draw the ticks as a LineCollection
        self._bg_handles.append(
            self.add_collection(LineCollection(ticks, colors='black', linewidths=1.0, zorder=0))
        )
        # Return the ticks to reuse them for the labels
        return ticks

    def _draw_tick_labels(self, ticks: np.ndarray) -> None:

        # Generate the tick labels based on the number of segments
        labels = (self._tick_positions * 100).astype(int).astype(str)
        # Position the labels along the tick direction, further out than the tick end
        label_pos = self._gen_ticks(ticks, self._tick_label_space)[:, 1, :]
        # Plot the text next to the ticks
        for (x, y), t in zip(label_pos, np.tile(labels, self._vertices.shape[0])):
            self._bg_handles.append(
                self.text(x, y, t, fontsize=9, ha='center', va='center', zorder=0)
            )

    def _draw_background(self) -> None:

        """
        Redraws the polygon outline, vertex labels, optionally ticks and grid.

        :return:
        """

        # Remove all previously drawn background artists
        remove_handles(self._bg_handles)
        # Generate the axes from the vertices
        self._draw_vertices()
        # If the ticks are enabled, draw ticks and labels
        if self._show_ticks:
            ticks = self._draw_ticks()
            self._draw_tick_labels(ticks)
        # If the grid is enabled, draw grid
        if self._show_grid:
            self._draw_grid()
        # Set the axes limits based on the vertices
        self.set_xlim(np.min(self._vertices[:, 0]) - 0.1, np.max(self._vertices[:, 0]) + 0.1)
        self.set_ylim(np.min(self._vertices[:, 1]) - 0.1, np.max(self._vertices[:, 1]) + 0.1)

    def _redraw_background(self) -> None:

        """
        Replace the polygon when the detected number of components differs from the current background n.
        """

        # Calculate the new vertices
        self._vertices = _gen_vertices(self._n_dim)
        # Redraw the new background
        self._apply_base_axes_style()
        self._draw_background()

    def _redraw_labels(self) -> None:

        """
        Redraw vertex labels (keep lines/collections).
        """

        # Remove all previously drawn vertices label artists
        remove_handles(self._label_handles, keyword='is_prim_label')
        # Draw the vertex labels again
        self._draw_prim_labels()

    def _draw_sec_labels(self) -> None:

        # Function to calculate the rotation angle of the text
        def rot(_v0: np.ndarray, _v1: np.ndarray):

            # Calculate the raw angle of the line segment from v0 to v1
            dx, dy = (_v1 - _v0)
            angle = np.degrees(np.arctan2(dy, dx))
            # Re-map to (-90, 90]: rotate upside-down text by 180°
            angle += 180 if angle <= -90 else -180 if angle > 90 else 0
            return angle

        # Iterate over all consecutive pairs of vertices
        _verts_r = np.roll(self._vertices, -1, axis=0)
        labels = np.roll(self._sec_labels, -1)
        for i, (v0, v1, label) in enumerate(zip(self._vertices, _verts_r, labels)):
            # Calculate their midpoint
            midpoint = 0.5 * (v0 + v1)
            # Rotate clockwise to get the normal (+y, -x) and then flip to turn outwards
            v01 = v1 - v0
            r_cw = np.array([v01[1], -v01[0]])
            # Get the normal vector
            n_vec = r_cw / np.linalg.norm(r_cw)
            # Use the midpoint and the normal vector to determine the text position
            x, y = midpoint + self._sec_label_space * n_vec
            # Draw the text, create a custom attribute to identify it later
            txt = self.text(x, y, label, ha='center', va='center', rotation=rot(v0, v1), rotation_mode='anchor',
                            transform_rotates_text=True, fontsize=10, zorder=0)
            txt._is_sec_label = True
            self._label_handles.append(txt)

    def scatter(self, comps: np.ndarray | pd.DataFrame, *args, labels: list[str] = None,
                **kwargs) -> CompSpaceScatter:

        # Convert the compositions to a numpy array if a DataFrame is provided, store the column names as labels
        labels = comps.columns.to_list() if isinstance(comps, pd.DataFrame) else labels
        comps = comps.values if isinstance(comps, pd.DataFrame) else comps
        # Get the number of components from the compositions
        n_dim = comps.shape[1]
        # Raise error if n is out of bounds
        if n_dim < 3 or n_dim > 8:
            raise ValueError('Supported number of components is 3 to 8.')
        # Make sure the provided labels are valid
        if labels is not None and len(labels) != n_dim:
            raise ValueError(f'The provided labels must have the same length as the data n={n_dim}.')
        # By default, only show the grid and ticks for ternary plots
        self._show_ticks, self._show_grid = n_dim in [3, 4], n_dim in [3, 4]
        # If this is the first data or n changed, (re)build background
        if (not self._has_data) and (n_dim != self._n_dim):
            self._n_dim = n_dim
            self._redraw_background()
        # Otherwise update the vertex labels if provided
        if labels is not None and labels != self._prim_labels:
            self._prim_labels = labels
            self._redraw_labels()
        # Convert barycentric rows to XY and call the base Axes.scatter
        cart = bary_to_cart(comps, self._vertices)
        self._has_data = True
        # Forward the scatter call to the parent class
        paths = super().scatter(cart[:, 0], cart[:, 1], *args, **kwargs)
        # Wrap the collection path in a container to allow updating the data
        return CompSpaceScatter([paths], self._vertices)

    def set_ticks(self, show: bool = True, ticks: list[str] = None):

        # Convert the ticks to a numpy array and convert percentages to fractions
        ticks = np.asarray(ticks) / 100 if ticks is not None else np.linspace(0, 1, 11)
        self._tick_positions = ticks
        # Toggle the ticks
        self._show_ticks = show
        # Redraw the background
        self._redraw_background()

    def set_grid(self, show: bool = True, grid_pos: list[float] | np.ndarray = None):

        # Convert the grid to a numpy array and convert percentages to fractions
        grid = np.asarray(grid_pos) / 100 if grid_pos is not None else np.linspace(0, 1, 11)
        self._grid_pos = grid
        # Toggle the grid
        self._show_grid = show
        # Redraw the background
        self._redraw_background()

    def set_labels(self, labels: list[str]):

        # Store the labels internally
        self._prim_labels = labels
        # Redraw the vertex labels
        self._redraw_labels()

    def set_sec_labels(self, labels: list[str]):

        # Store the secondary labels internally
        self._sec_labels = labels
        # Draw the secondary labels
        self._draw_sec_labels()
