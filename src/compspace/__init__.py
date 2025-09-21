import matplotlib as mpl
from matplotlib.projections import register_projection
from .axes_2d import CompSpace2DAxes
from .axes_3d import CompSpace3DAxes
from .animation import rot_animation
from .convenience import plot_on_comp_space


# Restore old 3D rotation style for Matplotlib >= 3.10
def _force_old_3d_rotation_style():

    # Only set if Matplotlib knows this key (>= 3.10)
    try:
        if 'axes3d.mouserotationstyle' in mpl.rcParams:
            mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'
    except Exception:
        # swallow any edge-case errors; old Matplotlib uses 'azel' anyway
        pass


# Register the projections
register_projection(CompSpace2DAxes)
register_projection(CompSpace3DAxes)
# Force old 3D rotation style
_force_old_3d_rotation_style()
