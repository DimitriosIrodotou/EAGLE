import os
import re
import time
import random
import warnings
import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.style as style

from astropy_healpix import HEALPix
from plot_tools import RotateCoordinates

style.use("classic")
plt.rcParams.update({'font.family':'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SampleRAEl:
    """
    For a sample of galaxies create: HEALPix histograms.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Generate the figure and define its parameters #
        figure, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 15), subplot_kw={'projection':'mollweide'})

        # Select a random sample from all group numbers #
        group_numbers = np.load(data_path + '/group_numbers.npy')
        group_numbers_sample = random.sample(list(group_numbers), 100)

        for i, axis in enumerate(axes.flatten()):
            start_local_time = time.time()  # Start the local time.

            group_number = i + 1
            subgroup_number = 0

            # Load the data #
            stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            stellar_data_tmp = stellar_data_tmp.item()
            print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')

            # Plot the data #
            start_local_time = time.time()  # Start the local time.

            self.plot(axis, stellar_data_tmp, group_number)
            print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
        # Save and close the figure #
        plt.savefig(plots_path + 'SRAEl' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        print('Finished SampleRAEl for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(axis, stellar_data_tmp, group_number):
        """
        Plot a sample of HEALPix histograms.
        :param axis: from __init__.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :return: None
        """
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        glx_unit_vector = glx_angular_momentum / np.linalg.norm(glx_angular_momentum)

        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        coordinates, velocities, prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(stellar_data_tmp, glx_unit_vector)

        # Calculate the ra and el of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        el = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Plot a HEALPix histogram #
        nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, el * u.deg)  # Create list of HEALPix indices from particles' ra and el.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

        # Sample a 360x180 grid in ra/el #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        el = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, el_grid = np.meshgrid(ra, el)

        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, el_grid)
        density_map = density[coordinate_index]

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axis.pcolormesh(np.radians(ra), np.radians(el), density_map, cmap='nipy_spectral_r')
        # cbar = plt.colorbar(pcm, ax=axis, orientation='horizontal')
        # cbar.ax.tick_params(labelsize=20)

        # Define the figure parameters #
        axis.axis('off')
        plt.text(0.0, 0.95, str(group_number), color='red', fontsize=20, transform=axis.transAxes)

        return pcm


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleRAEl(simulation_path, tag)
