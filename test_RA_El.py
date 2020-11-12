import os
import re
import time
import random
import warnings
import statistics
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import healpy as hlp
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
disc_fractions = []


class TestRAEl:
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
        figure, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20), subplot_kw={'projection':'mollweide'})

        # Select a random sample from all group numbers #
        group_numbers = np.load(data_path + '/group_numbers.npy')
        group_numbers_sample = random.sample(list(group_numbers), 100)

        # for i, axis in enumerate(axes.flatten()):
        #     start_local_time = time.time()  # Start the local time.
        #
        #     group_number = i + 1
        #     subgroup_number = 0
        #
        #     # Load the data #
        #     stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
        #                                allow_pickle=True)
        #     stellar_data_tmp = stellar_data_tmp.item()
        #     print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
        #     print('–––––––––––––––––––––––––––––––––––––––––––––')
        #
        #     # Plot the data #
        #     start_local_time = time.time()  # Start the local time.
        #
        #     disc_fractions = self.plot(axis, stellar_data_tmp, group_number)
        #     print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
        #     print('–––––––––––––––––––––––––––––––––––––––––––––')
        # np.save(data_path + '/disc_fractions_test_3000', disc_fractions)
        #
        # # Save and close the figure #
        # plt.savefig(plots_path + 'tRAEl' + '-' + date + '.png', bbox_inches='tight')
        # plt.close()

        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 1], ylim=[-0.1, 0.1], xlabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$',
                            ylabel=r'$\mathrm{D/T_{\Delta \theta<30\degree} - D/T_{\Delta \theta<30\degree},reduced}$', aspect=None, which='major')
        x = np.load(data_path + 'disc_fractions_test.npy')
        y_300 = np.load(data_path + 'disc_fractions_test_300.npy')
        y_3000 = np.load(data_path + 'disc_fractions_test_3000.npy')
        plt.scatter(x, x - y_300, color='tab:red',label=r'$\mathrm{300}$')
        plt.scatter(x, x - y_3000, color='tab:blue',label=r'$\mathrm{3000}$')
        print(np.mean(x - y_300))
        print(statistics.pstdev(x - y_300))
        print(np.mean(x - y_3000))
        print(statistics.pstdev(x - y_3000))
        axis.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1, scatterpoints=1)
        plt.savefig(plots_path + 'tRAEl2' + '-' + date + '.png', bbox_inches='tight')

        print('Finished TestRAEl for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
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
        ids = np.arange(0, len(stellar_data_tmp['Mass']))
        sample = random.sample(list(ids), 3000)

        for attribute in ['Mass', 'Coordinates', 'Velocity']:
            stellar_data_tmp[attribute] = np.copy(stellar_data_tmp[attribute][sample])
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
        densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

        # Sample a 360x180 grid in ra/el #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        el = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, el_grid = np.meshgrid(ra, el)

        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, el_grid)
        density_map = densities[coordinate_index]
        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axis.pcolormesh(np.radians(ra), np.radians(el), density_map, cmap='nipy_spectral_r')
        cbar = plt.colorbar(pcm, ax=axis, ticks=[0, np.floor(max(np.hstack(density_map)) / 20) * 10, np.floor(max(np.hstack(density_map)) / 11) * 10],
                            orientation='horizontal')
        cbar.ax.tick_params(labelsize=15)

        # Perform a top-hat smoothing on the densities #
        smoothed_densities = np.zeros(hp.npix)
        # Loop over all grid cells #
        for i in range(hp.npix):
            mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30degree cone search around each grid cell.
            smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

        # Find the location of the density maximum and plot its positions and the ra and el of the galactic angular momentum #
        index_densest = np.argmax(smoothed_densities)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2

        # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask = np.where(angular_theta_from_densest < (np.pi / 6.0))
        glx_disc_fractions_IT20 = np.divide(np.sum(stellar_data_tmp['Mass'][disc_mask]), np.sum(stellar_data_tmp['Mass']))
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)

        disc_fractions.append(glx_disc_fractions_IT20)

        # Define the figure parameters #
        axis.axis('off')
        plt.text(0.0, 0.95, str(group_number), color='red', fontsize=20, transform=axis.transAxes)

        return disc_fractions


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = TestRAEl(simulation_path, tag)
