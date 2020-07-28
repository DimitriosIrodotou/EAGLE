import os
import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import gridspec
from astropy_healpix import HEALPix
from plot_tools import RotateCoordinates
from morpho_kinematics import MorphoKinematic

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SampleMultipleDecomposition:
    """
    For each galaxy create: a HEALPix histogram from the angular momentum of particles and angular distance plots.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        group_numbers = [25, 2, 3, 5, 20, 6, 21, 18]

        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(24, 13.5))

        gs = gridspec.GridSpec(4, 6)
        axis00, axis01, axis02, axis03, axis04, axis05 = figure.add_subplot(gs[0, 0], projection='mollweide'), figure.add_subplot(
            gs[0, 1]), figure.add_subplot(gs[0, 2]), figure.add_subplot(gs[0, 3], projection='mollweide'), figure.add_subplot(
            gs[0, 4]), figure.add_subplot(gs[0, 5]),
        axis10, axis11, axis12, axis13, axis14, axis15 = figure.add_subplot(gs[1, 0], projection='mollweide'), figure.add_subplot(
            gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(gs[1, 3], projection='mollweide'), figure.add_subplot(
            gs[1, 4]), figure.add_subplot(gs[1, 5]),
        axis20, axis21, axis22, axis23, axis24, axis25 = figure.add_subplot(gs[2, 0], projection='mollweide'), figure.add_subplot(
            gs[2, 1]), figure.add_subplot(gs[2, 2]), figure.add_subplot(gs[2, 3], projection='mollweide'), figure.add_subplot(
            gs[2, 4]), figure.add_subplot(gs[2, 5])
        axis30, axis31, axis32, axis33, axis34, axis35 = figure.add_subplot(gs[3, 0], projection='mollweide'), figure.add_subplot(
            gs[3, 1]), figure.add_subplot(gs[3, 2]), figure.add_subplot(gs[3, 3], projection='mollweide'), figure.add_subplot(
            gs[3, 4]), figure.add_subplot(gs[3, 5])

        for axis in [axis00, axis10, axis20, axis30]:
            axis.set_xlabel(r'$\mathrm{\alpha\;(\degree)}$')
            axis.set_ylabel(r'$\mathrm{\delta\;(\degree)}$')
        # for axis in [axis01, axis05, axis11, axis15, axis21, axis25, axis31, axis35]:
        #     plot_tools.set_axis(axis, aspect=None, size=10)
        for axis in [axis01, axis04, axis11, axis14, axis21, axis24]:
            plot_tools.set_axis(axis, ylabel=r'$\mathrm{Particles\;per\;grid\;cell}$', aspect=None, size=10)
        for axis in [axis31, axis34]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{\Delta \theta\;(\degree)}$', ylabel=r'$\mathrm{Particles\;per\;grid\;cell}$', aspect=None,
                                size=10)
        for axis in [axis31, axis35]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{\Delta \theta\;(\degree)}$', aspect=None, size=10)
        for axis in [axis02, axis05, axis12, axis15, axis22, axis25, axis32, axis35]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{Angular\;distance\;from\;\vec{J}_{gal}\;(\degree)}$',
                                ylabel=r'$\mathrm{Particles\;per\;grid\;cell}$', aspect=None, size=10)

        all_axes = [[axis00, axis01, axis02], [axis03, axis04, axis05], [axis10, axis11, axis12], [axis13, axis14, axis15], [axis20, axis21, axis22],
                    [axis23, axis24, axis25], [axis30, axis31, axis32], [axis33, axis34, axis35]]

        for group_number, axes in zip(group_numbers, all_axes):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):  # Get centrals only.
                start_local_time = time.time()  # Start the local time.

                # Load data from numpy arrays #
                stellar_data_tmp = np.load(
                    data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy', allow_pickle=True)
                stellar_data_tmp = stellar_data_tmp.item()
                print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

                # Plot the data #
                start_local_time = time.time()  # Start the local time.

                self.plot(axes, stellar_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

        plt.savefig(plots_path + 'SMD' + '-' + date + '.png', bbox_inches='tight')
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(axes, stellar_data_tmp, group_number, subgroup_number):
        """
        Plot a HEALPix histogram from the angular momentum of particles - an angular distance plot - a surface density plot / gri mock image - a
        circularity distribution.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """

        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        glx_unit_vector = glx_angular_momentum / np.linalg.norm(glx_angular_momentum)

        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(stellar_data_tmp,
                                                                                                                                     glx_unit_vector)

        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Plot a HEALPix histogram #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.

        # Find location of density maximum and plot its positions and the ra (lon) and dec (lat) of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        axes[0].annotate(r'$\mathrm{Density\;maximum}$', xy=(lon_densest, lat_densest), xycoords='data', xytext=(0.2, 1.1),
                         textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='-', color='black', connectionstyle='arc3,rad=0'))  # Position of the densest pixel.
        axes[0].scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=100, color='black', marker='X',
                        facecolors='none', zorder=5)  # Position of the galactic angular momentum.

        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)

        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = density[coordinate_index]

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axes[0].pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        cbar = plt.colorbar(pcm, ax=axes[0], orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)

        # # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask = np.where(angular_theta_from_densest < (np.pi / 6.0))
        disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][disc_mask]) / np.sum(stellar_data_tmp['Mass'])

        # Calculate and plot the angular distance (spherical law of cosines) between the densest and all the other grid cells #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.radians(dec_grid.value)) + np.cos(lat_densest) * np.cos(np.radians(dec_grid.value)) * np.cos(
                lon_densest - np.radians(ra_grid.value)))  # In radians.

        axes[1].scatter(angular_theta_from_densest[density_map.nonzero()] * (180.0 / np.pi), density_map[density_map.nonzero()], c='black',
                        s=5)  # In degrees.
        axes[1].axvline(x=30, c='blue', lw=3, linestyle='dashed', label='D/T= %.3f ' % disc_fraction_IT20)  # Vertical line at 30 degrees.
        axes[1].axvspan(0, 30, facecolor='0.2', alpha=0.5)  # Draw a vertical span.

        # Calculate the kinematic diagnostics #
        kappa, disc_fraction, circularity, rotational_over_dispersion, vrots, rotational_velocity, sigma_0, \
        delta = MorphoKinematic.kinematic_diagnostics(
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'], stellar_data_tmp['ParticleBindingEnergy'])

        # Calculate the D/T based on counter-rotation particles #
        j, = np.where(circularity < 0.0)
        l, = np.where((circularity > -1.7) & (circularity < 1.7))
        disc_fraction_00 = 1 - 2 * np.sum(stellar_data_tmp['Mass'][j]) / np.sum(stellar_data_tmp['Mass'][l])

        # Calculate and plot the angular distance between the (unit vector of) the galactic angular momentum and all the other grid cells #
        position_of_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T

        angular_theta_from_X = np.arccos(np.sin(position_of_X[0, 1]) * np.sin(np.radians(dec_grid.value)) + np.cos(position_of_X[0, 1]) * np.cos(
            np.radians(dec_grid.value)) * np.cos(position_of_X[0, 0] - np.radians(ra_grid.value)))  # In radians.
        axes[2].scatter(angular_theta_from_X[density_map.nonzero()] * (180.0 / np.pi), density_map[density_map.nonzero()], c='black',
                        s=5)  # In degrees.
        axes[2].axvline(x=90, c='red', lw=3, linestyle='dashed', label='D/T= %.3f ' % disc_fraction_00)  # Vertical line at 30 degrees.
        axes[2].axvspan(90, 180, facecolor='0.2', alpha=0.5)  # Draw a vertical span.

        # Create the legends and save the figure #
        axes[1].legend(loc='upper center', fontsize=12, frameon=False, scatterpoints=3)
        axes[2].legend(loc='upper center', fontsize=12, frameon=False, scatterpoints=3)
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleMultipleDecomposition(simulation_path, tag)
