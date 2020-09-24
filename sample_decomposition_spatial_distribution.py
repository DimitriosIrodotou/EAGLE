import os
import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import healpy as hlp
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import gridspec
from astropy_healpix import HEALPix
from plot_tools import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SampleDecompositionSpatialDistribution:
    """
    For a sample of galaxies create: a HEALPix histogram from the angular momentum of particles and a spatial distribution of the face-on and
    edge-on projections plots.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        group_numbers = [355, 364, 696, 1009, 1182]

        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(25, 25))

        gs = gridspec.GridSpec(6, 5, wspace=0.4, hspace=0.4, height_ratios=[0.1, 1, 1, 1, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10, axis11, axis12, axis13, axis14 = figure.add_subplot(gs[1, 0], projection='mollweide'), figure.add_subplot(
            gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(gs[1, 3]), figure.add_subplot(gs[1, 4])
        axis20, axis21, axis22, axis23, axis24 = figure.add_subplot(gs[2, 0], projection='mollweide'), figure.add_subplot(
            gs[2, 1]), figure.add_subplot(gs[2, 2]), figure.add_subplot(gs[2, 3]), figure.add_subplot(gs[2, 4])
        axis30, axis31, axis32, axis33, axis34 = figure.add_subplot(gs[3, 0], projection='mollweide'), figure.add_subplot(
            gs[3, 1]), figure.add_subplot(gs[3, 2]), figure.add_subplot(gs[3, 3]), figure.add_subplot(gs[3, 4])
        axis40, axis41, axis42, axis43, axis44 = figure.add_subplot(gs[4, 0], projection='mollweide'), figure.add_subplot(
            gs[4, 1]), figure.add_subplot(gs[4, 2]), figure.add_subplot(gs[4, 3]), figure.add_subplot(gs[4, 4])
        axis50, axis51, axis52, axis53, axis54 = figure.add_subplot(gs[5, 0], projection='mollweide'), figure.add_subplot(
            gs[5, 1]), figure.add_subplot(gs[5, 2]), figure.add_subplot(gs[5, 3]), figure.add_subplot(gs[5, 4])

        for axis in [axis10, axis20, axis30, axis40, axis50]:
            axis.set_xlabel(r'$\mathrm{\alpha/\degree}$', size=20)
            axis.set_ylabel(r'$\mathrm{\delta/\degree}$', size=20)
            axis.set_yticklabels(['', '-60', '', '-30', '', '0', '', '30', '', '60', ''], size=20)
            axis.set_xticklabels(['', '-120', '', '-60', '', '0', '', '60', '', '120', ''], size=20)
        for axis in [axis11, axis21, axis31, axis41, axis51, axis13, axis23, axis33, axis43, axis53]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{y/kpc}$', aspect=None, size=20)
        for axis in [axis12, axis22, axis32, axis42, axis52, axis14, axis24, axis34, axis44, axis54]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$', aspect=None, size=20)

        all_axes = [[axis10, axis11, axis12, axis13, axis14], [axis20, axis21, axis22, axis23, axis24], [axis30, axis31, axis32, axis33, axis34],
                    [axis40, axis41, axis42, axis43, axis44], [axis50, axis51, axis52, axis53, axis54]]

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

                im = self.plot(axes, stellar_data_tmp, group_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Add color bar #
        plot_tools.create_colorbar(axiscbar, im, r'$\mathrm{log_{10}(\Sigma_{\bigstar}/(M_\odot\,kpc^{-2}))}$', 'horizontal', extend='both', size=20)
        # Add text #
        plt.text(0.17, 1.1, r'$\mathrm{Disc\;face-on}$', fontsize=20, transform=axis11.transAxes)
        plt.text(0.17, 1.1, r'$\mathrm{Disc\;edge-on}$', fontsize=20, transform=axis12.transAxes)
        plt.text(0.17, 1.1, r'$\mathrm{Disc\;face-on}$', fontsize=20, transform=axis13.transAxes)
        plt.text(0.17, 1.1, r'$\mathrm{Disc\;edge-on}$', fontsize=20, transform=axis14.transAxes)

        # Save and close the figure #
        plt.savefig(plots_path + 'SDSD' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(axes, stellar_data_tmp, group_number):
        """
        Plot HEALPix histogram from the angular momentum of particles and the spatial distribution of the face-on and edge-on projections.
        :param axes: set of axes
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

        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Plot a HEALPix histogram #
        nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
        hp = HEALPix(nside=nside, order='ring')  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create a list of HEALPix indices from particles' ra and dec.
        densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

        # Perform a top-hat smoothing on the densities #
        smoothed_densities = np.zeros(hp.npix)
        # Loop over all grid cells #
        for i in range(hp.npix):
            mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30degree cone search around each grid cell.
            smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

        # Find the location of density maximum and plot its positions and the ra (lon) and dec (lat) of the galactic angular momentum #
        index_densest = np.argmax(smoothed_densities)

        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        axes[0].scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=100, color='black', marker='X',
                        facecolors='none', zorder=5)  # Position of the galactic angular momentum.

        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)

        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = densities[coordinate_index]

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axes[0].pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        cbar = plt.colorbar(pcm, ax=axes[0], orientation='horizontal')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('$\mathrm{Particles\;per\;grid\;cell}$', size=20)

        # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask, = np.where(angular_theta_from_densest < (np.pi / 6.0))
        disc_cr_mask, = np.where((angular_theta_from_densest < (np.pi / 6.0)) | (angular_theta_from_densest > 5 * (np.pi / 6.0)))
        disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][disc_mask]) / np.sum(stellar_data_tmp['Mass'])
        disc_fraction_IT20_cr = np.sum(stellar_data_tmp['Mass'][disc_cr_mask]) / np.sum(stellar_data_tmp['Mass'])

        # Plot the 2D surface density projection and scatter for the disc #
        # Rotate coordinates and velocities of the disc component so it appears face-on and edge-on #
        coordinates, velocities, component_data = RotateCoordinates.rotate_component(stellar_data_tmp, disc_mask)
        vmin, vmax = 6, 8
        weights = component_data['Mass']
        cmap = matplotlib.cm.get_cmap('nipy_spectral_r')

        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], weights=weights, bins=200, range=[[-30, 30], [-30, 30]])
        im = axes[1].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
                            aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 2], weights=weights, bins=200, range=[[-30, 30], [-30, 30]])
        axes[2].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, aspect='equal')

        # Plot the 2D surface density projection and scatter for the bulge #
        # Rotate coordinates and velocities of the disc component so it appears face-on and edge-on #
        coordinates, velocities, component_data = RotateCoordinates.rotate_component(stellar_data_tmp, disc_cr_mask)
        weights = component_data['Mass']
        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], weights=weights, bins=200, range=[[-30, 30], [-30, 30]])
        axes[3].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 2], weights=weights, bins=200, range=[[-30, 30], [-30, 30]])
        axes[4].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, aspect='equal')

        # Add text and create the legend #
        plt.text(-0.2, 1.1, str(group_number), color='red', fontsize=20, transform=axes[0].transAxes)
        plt.text(0.0, 0.9, r'$\mathrm{D/T_{\Delta \theta<30\degree}=%.2f}$' % disc_fraction_IT20, fontsize=20, transform=axes[1].transAxes)
        plt.text(0.0, 0.9, r'$\mathrm{D/T_{CR}=%.2f}$' % disc_fraction_IT20_cr, fontsize=20, transform=axes[3].transAxes)
        axes[3].legend(loc='upper left', fontsize=20, frameon=False)
        return im


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleDecompositionSpatialDistribution(simulation_path, tag)
