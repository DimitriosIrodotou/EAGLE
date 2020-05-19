import re
import time
import warnings
import argparse
import matplotlib

matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec
from morpho_kinematics import MorphoKinematic

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.

lon = []
lat = []

# Set the style of the plots #
sns.set()
sns.set_style('ticks')
sns.set_context('notebook', font_scale=1.6)

# Generate the figure and define its parameters #
plt.close()
figure = plt.figure(0, figsize=(20, 15))

gs = gridspec.GridSpec(2, 2)
ax00 = plt.subplot(gs[0, 0], projection="mollweide")
ax01 = plt.subplot(gs[0, 1])
ax10 = plt.subplot(gs[1, 0])
ax11 = plt.subplot(gs[1, 1])

ax00.grid(True)
ax10.grid(True)
ax01.grid(True)
ax00.set_xlabel('RA ($\degree$)')
ax00.set_ylabel('Dec ($\degree$)')
ax10.set_ylabel('Particles per hexbin')
ax01.set_ylabel('Particles per hexbin')
ax10.set_xlabel('Angular distance from X ($\degree$)')
ax01.set_xlabel('Angular distance from densest hexbin ($\degree$)')

ax01.set_xlim(-10, 190)
ax10.set_xlim(-10, 190)

y_tick_labels = np.array(['', '-60', '', '-30', '', '0', '', '30', '', 60])
x_tick_labels = np.array(['', '-120', '', '-60', '', '0', '', '60', '', 120])
ax00.set_xticklabels(x_tick_labels)
ax00.set_yticklabels(y_tick_labels)

ax01.set_xticks(np.arange(0, 181, 20))
ax10.set_xticks(np.arange(0, 181, 20))

# Generate the RA and Dec projection #
l = np.sqrt(1/3)
x0 = [0, 0.5, 0.5]
x1 = [l, l, l]
x2 = [0, 1, 0]
x3 = [0, -1, 0]
x4 = [1, 0, 0]
x5 = [-1, 0, 0]

vectors = np.vstack([x1])
glx_angular_momentum = np.sum(vectors, axis=0)
ax00.scatter(np.arctan2(vectors[:, 1], vectors[:, 0]), np.arcsin(vectors[:, 2]), zorder=-1)  # Element-wise arctan of x1/x2.

print(vectors, np.shape(vectors))
print(glx_angular_momentum, np.shape(glx_angular_momentum))
# ax00.scatter(np.arctan2(glx_angular_momentum[1], glx_angular_momentum[0]), np.arcsin(glx_angular_momentum[2]), s=300, color='red', marker='X',
#                     zorder=5)  # Position of the galactic angular momentum.

# Calculate and plot the angular separation again but use angular trigonometry this time #
# angular_theta_from_densest = np.arccos(
#     np.sin(position_densest[0, 1]) * np.sin(position_other[:, 1]) + np.cos(position_densest[0, 1]) * np.cos(position_other[:, 1]) * np.cos(
#         position_densest[0, 0] - position_other[:, 0]))  # In radians.
# ax01.scatter(angular_theta_from_densest * np.divide(180.0, np.pi), counts, c='red', s=50)  # In degrees.

# ax01.axvline(x=30, c='blue', lw=3, linestyle='dashed')  # Vertical line at 30 degrees.

# Calculate and plot the angular distance in degrees between the densest and all the other hexbins #
# position_X = np.vstack([np.arctan2(vectors[1], vectors[0]), np.arcsin(vectors[2])]).T
#
# angular_theta_from_X = np.arccos(
#     np.sin(position_X[0, 1]) * np.sin(position_other[:, 1]) + np.cos(position_X[0, 1]) * np.cos(position_other[:, 1]) * np.cos(
#         position_X[0, 0] - position_other[:, 0]))  # In radians.
# ax10.scatter(angular_theta_from_X * np.divide(180.0, np.pi), counts, c='red', s=50)  # In degrees.
#
# distance = np.linalg.norm(np.subtract(position_X, position_other), axis=1)
# index, = np.where(distance < np.divide(np.pi, 6.0))
# ax00.scatter(position_other[index, 0], position_other[index, 1], s=10, c='pink')
#

# ax10.axvline(x=30, c='red', lw=3, linestyle='dashed')  # Vertical line at 30 degrees.

# Save the plot #
outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RDSD/G-EAGLE/'  # Path to save plots.
SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/RDSD/G-EAGLE/'  # Path to save/load data.
plt.savefig(outdir + 'Test' + '-' + date + '.png', bbox_inches='tight')