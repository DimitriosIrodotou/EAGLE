# Import required python libraries #
import time

import numpy as np

import di_io
import di_plots

# Start the time #
start_time = time.time()

# Ask the user to define #
properties, prop_num = di_io.ask_user()

# Load the data #
X = np.load('./Data/' + properties[0] + '.npy')
Y = np.load('./Data/' + properties[1] + '.npy')
if prop_num != '2':
    z = np.load('./Data/' + properties[2] + '.npy')

# Make a plot #
di_plots.set_figure_param(1, di_plots.generate_figure(1), X, Y, xscale='log', yscale='log', xlabel=properties[0], ylabel=properties[1])
di_plots.make_plot('scatter', X, Y)

# Finish time #
print("--- %s seconds ---" % (time.time() - start_time))