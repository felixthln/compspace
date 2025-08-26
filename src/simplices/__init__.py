from matplotlib.projections import register_projection
from .axes_2d import Simplices2DAxes

# Register the projections
register_projection(Simplices2DAxes)
