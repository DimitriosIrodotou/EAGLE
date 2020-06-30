import os
import re
import time
import warnings
import matplotlib
import access_database

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import gridspec
from astropy_healpix import HEALPix
from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematic

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class MultipleDecomposition:
    """
    For each galaxy create: a HEALPix histogram from the angular momentum of particles - an angular distance plot - a surface density plot / gri mock
    image - a circularity distribution.
    a circularity plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        for group_number in range(2552, 2555):  # Loop over all masked haloes.
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
                
                self.plot(stellar_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_data_tmp, group_number, subgroup_number):
        """
        Plot a HEALPix histogram from the angular momentum of particles - an angular distance plot - a surface density plot / gri mock image - a
        circularity distribution.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(16, 9))
        
        gs = gridspec.GridSpec(2, 3)
        axis00 = figure.add_subplot(gs[0, 0], projection='mollweide')
        axis01 = figure.add_subplot(gs[0, 1])
        axis02 = figure.add_subplot(gs[0, 2])
        axis10 = figure.add_subplot(gs[1, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        axis12 = figure.add_subplot(gs[1, 2])
        
        for axis in [axis10, axis11, axis12]:
            axis.grid(True, which='both', axis='both')
        
        for axis in [axis10, axis11]:
            axis.set_xlim(-10, 190)
            axis.set_xticks(np.arange(0, 181, 20))
        
        axis00.set_xlabel('RA ($\degree$)')
        axis00.set_ylabel('Dec ($\degree$)')
        axis01.axis('off')
        axis02.axis('off')
        axis10.set_ylabel(r'$\mathrm{Particles\;per\;grid\;cell}$')
        axis10.set_xlabel(r'$\mathrm{Angular\;distance\;from\;X\;(\degree)}$')
        axis11.set_ylabel(r'$\mathrm{Particles\;per\;grid\;cell}$')
        axis11.set_xlabel(r'$\mathrm{Angular\;distance\;from\;densest\;grid\;cell\;(\degree)}$')
        axis12.set_xlabel(r'$\mathrm{\epsilon}$')
        axis12.set_ylabel(r'$\mathrm{f(\epsilon)}$')
        axis01.text(0.0, 0.95, r'$\mathrm{Face-on}$', c='w', fontsize=12, transform=axis01.transAxes)
        axis02.text(0.0, 0.95, r'$\mathrm{Edge-on}$', c='w', fontsize=12, transform=axis02.transAxes)
        
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
        axis00.annotate(r'$\mathrm{Density\;maximum}$', xy=(lon_densest, lat_densest), xycoords='data', xytext=(0.78, 1.00), textcoords='axes fraction',
                      arrowprops=dict(arrowstyle='-', color='black', connectionstyle='arc3,rad=0'))  # Position of the densest pixel.
        axis00.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=100, color='black', marker='X',
                     facecolors='none', zorder=5)  # Position of the galactic angular momentum.
        
        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)
        
        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = density[coordinate_index]
        
        # Display data on a 2D regular raster and create a pseudo-color plot #
        im = axis00.imshow(density_map, cmap='nipy_spectral_r', aspect='auto', norm=matplotlib.colors.LogNorm(vmin=1))
        cbar = plt.colorbar(im, ax=axis00, orientation='horizontal')
        cbar.set_label('$\mathrm{Particles\; per\; grid\; cell}$', size=12)
        axis00.pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        
        # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask = np.where(angular_theta_from_densest < (np.pi / 6.0))
        disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][disc_mask]) / np.sum(stellar_data_tmp['Mass'])
        
        # Plot the 2D surface density projections #
        galaxy_id = access_database.download_image(group_number, subgroup_number)
        img = mpimg.imread(data_path + 'images/galface_' + galaxy_id)
        img1 = mpimg.imread(data_path + 'images/galedge_' + galaxy_id)
        axis01.imshow(img)
        axis02.imshow(img1)
        
        # Calculate and plot the angular distance (spherical law of cosines) between the densest and all the other grid cells #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.radians(dec_grid.value)) + np.cos(lat_densest) * np.cos(np.radians(dec_grid.value)) * np.cos(
                lon_densest - np.radians(ra_grid.value)))  # In radians.
        
        axis11.scatter(angular_theta_from_densest[density_map.nonzero()] * (180.0 / np.pi), density_map[density_map.nonzero()], c='black',
                     s=5)  # In degrees.
        axis11.axvline(x=30, c='blue', lw=3, linestyle='dashed', label='D/T= %.3f ' % disc_fraction_IT20)  # Vertical line at 30 degrees.
        axis11.axvspan(0, 30, facecolor='0.2', alpha=0.5)  # Draw a vertical span.
        
        # Calculate the kinematic diagnostics #
        kappa, disc_fraction, circularity, rotational_over_dispersion, vrots, rotational_velocity, sigma_0, \
        delta = MorphoKinematic.kinematic_diagnostics(
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'], stellar_data_tmp['ParticleBindingEnergy'])
        
        # Calculate and plot the distribution of orbital circularity #
        j, = np.where(circularity < 0.0)
        k, = np.where((circularity > 0.7) & (circularity < 1.7))
        l, = np.where((circularity > -1.7) & (circularity < 1.7))
        disc_fraction_00 = 1 - 2 * np.sum(stellar_data_tmp['Mass'][j]) / np.sum(stellar_data_tmp['Mass'][l])
        disc_fraction_07 = np.sum(stellar_data_tmp['Mass'][k]) / np.sum(stellar_data_tmp['Mass'][l])
        
        ydata, edges = np.histogram(circularity, bins=100, range=[-1.7, 1.7], weights=stellar_data_tmp['Mass'] / np.sum(stellar_data_tmp['Mass']))
        ydata /= edges[1:] - edges[:-1]
        axis12.plot(0.5 * (edges[1:] + edges[:-1]), ydata, label='D/T = %.3f' % disc_fraction_07)
        
        # Calculate and plot the angular distance between the (unit vector of) the galactic angular momentum and all the other grid cells #
        position_of_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T
        
        angular_theta_from_X = np.arccos(np.sin(position_of_X[0, 1]) * np.sin(np.radians(dec_grid.value)) + np.cos(position_of_X[0, 1]) * np.cos(
            np.radians(dec_grid.value)) * np.cos(position_of_X[0, 0] - np.radians(ra_grid.value)))  # In radians.
        axis10.scatter(angular_theta_from_X[density_map.nonzero()] * (180.0 / np.pi), density_map[density_map.nonzero()], c='black', s=5)  # In degrees.
        axis10.axvline(x=90, c='red', lw=3, linestyle='dashed', label='D/T= %.3f ' % disc_fraction_00)  # Vertical line at 30 degrees.
        axis10.axvspan(90, 180, facecolor='0.2', alpha=0.5)  # Draw a vertical span.
        
        # Create the legends and save the figure #
        axis10.legend(loc='upper center', fontsize=12, frameon=False, scatterpoints=3)
        axis11.legend(loc='upper center', fontsize=12, frameon=False, scatterpoints=3)
        axis12.legend(loc='upper left', fontsize=12, frameon=False, scatterpoints=3)
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'MD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/MD/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = MultipleDecomposition(simulation_path, tag)
