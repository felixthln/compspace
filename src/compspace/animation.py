from pathlib import Path
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation
import ffmpeg  # Needed for saving the animation with matplotlib


def rot_animation(fig: Figure, ax: Axes3D, path: str | Path, elev: float = 10, dpi: int = 200, fps: int = 30) -> None:

    """
    Creates and saves a 3D rotation animation of a given figure and axis.

    :param fig: matplotlib figure object
    :param ax: matplotlib axes object with 3D configurations
    :param path: combination of path and filename the animation will be stored in
    :param elev: elevation of the 3D figure in degrees, defaults to 10°
    :param dpi: resolution of the saved animation, defaults to 200
    :param fps: frames per second of the saved animation, defaults to 30
    """

    # Check if the given axis is configured for 3D and raise error if not
    if ax.name != '3d':
        raise ValueError('The specified axis needs to be a 3D object. Create it e.g. by plt.axes(projection=\'3d\').')

    # Define the rotation animation
    def animate(i):
        ax.view_init(elev=elev, azim=i)

    # Create the animation object
    anim = FuncAnimation(fig, animate, frames=360, interval=20)
    # Save
    anim.save(path, dpi=dpi, fps=fps)
