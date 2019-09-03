import groups_convert
import plots
from files import *

properties = ['Mass', 'HalfMassRad']
component = 'g'

x, y = groups_convert.Convert.convert_to_physical_units(properties, component)  # Load the data and convert units to physical.
x, y, figure, ax1, ax2 = plots.generate_subplots(properties, x, y, 'log', 'log', (1e8, 1e12), (1e-2, 1e2), title)  # Generate the plot.
plots.make_subplots(x, y, figure, ax1, ax2, 'h', 'log', 'log')  # Make the plot.