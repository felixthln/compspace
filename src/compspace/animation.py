from pathlib import Path
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections.abc import Iterable
import ffmpeg  # Needed for saving the animation with matplotlib


def rot_animation(fig: Figure, ax: Axes3D | list[Axes3D], path: str | Path, elev: float = 10, dpi: int = 200, fps: int = 30) -> None:

    """
    Creates and saves a 3D rotation animation of a given figure and axis.

    :param fig: matplotlib figure object
    :param ax: matplotlib axes object with 3D configurations
    :param path: combination of path and filename the animation will be stored in
    :param elev: elevation of the 3D figure in degrees, defaults to 10°
    :param dpi: resolution of the saved animation, defaults to 200
    :param fps: frames per second of the saved animation, defaults to 30
    """

    # Wrap the axes in a list if only one was provided
    ax = [ax] if not isinstance(ax, Iterable) else ax

    # Make sure all axes are 3D
    for a in ax:
        if a.name != '3d' and a.name != 'compspace3D':
            raise ValueError('The specified axis needs to be a 3D object. Create it e.g. by plt.axes(projection=\'3d\').')

    # Define the rotation animation
    def animate(i):
        # Animate all axes
        for a in ax:
            a.view_init(elev=elev, azim=i)

    # Create the animation object
    anim = FuncAnimation(fig, animate, frames=360, interval=20)
    # Save
    anim.save(path, dpi=dpi, fps=fps)
