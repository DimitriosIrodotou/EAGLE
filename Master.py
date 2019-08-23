# Import required python libraries #
import time

import di_io
import di_plots

# Start the time #
start_time = time.time()

# Ask the user to define #
properties, prop_num = di_io.AskUser.io('self')

# Load the data and convert units to physical #
X = di_io.convert_to_physical_units(properties[0])
Y = di_io.convert_to_physical_units(properties[1])

if prop_num != '2':
	Z = di_io.convert_to_physical_units(properties[1])

# Make a plot #
style, xscale, yscale = di_io.AskUser.plot('self')
di_plots.make_plot(di_plots.generate_figure(1, properties, xscale, yscale, title='0021-z004p688'), style, X, Y, xscale, yscale)

# Finish time #
print("--- %s seconds ---" % (time.time() - start_time))