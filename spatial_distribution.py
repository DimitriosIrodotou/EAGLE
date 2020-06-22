import os
import re
import time
import warnings
import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import gridspec
from astropy_healpix import HEALPix
from rotate_galaxies import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SpatialDistribution:
    """
    For each galaxy create: a spatial distribution maps.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        for group_number in range(1, 26):  # Loop over all masked haloes.
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
        
        print('Finished SpatialDistribution for ' + re.split('EAGLE/|/data', simulation_path)[2] + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_data_tmp, group_number, subgroup_number):
        """
        Plot spatial distribution maps.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(0, figsize=(20, 15))
        
        gs = gridspec.GridSpec(3, 2, hspace=0.07, wspace=0.07, height_ratios=[0.05, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10 = figure.add_subplot(gs[1, 0])
        axis20 = figure.add_subplot(gs[2, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        axis21 = figure.add_subplot(gs[2, 1])
        
        for axis in [axis10, axis20, axis11, axis21]:
            axis.grid(True, which='both', axis='both')
            axis.set_xlim(-30, 30)
            axis.set_ylim(-30, 30)
            axis.tick_params(direction='out', which='both', top='on', right='on',  labelsize=16)
        
        axis10.set_xticklabels([])
        axis11.set_xticklabels([])
        axis10.set_ylabel(r'$\mathrm{y\,[kpc]}$', size=16)
        axis20.set_xlabel(r'$\mathrm{x\,[kpc]}$', size=16)
        axis20.set_ylabel(r'$\mathrm{z\,[kpc]}$', size=16)
        axis11.set_ylabel(r'$\mathrm{y\,[kpc]}$', size=16)
        axis21.set_xlabel(r'$\mathrm{x\,[kpc]}$', size=16)
        axis21.set_ylabel(r'$\mathrm{z\,[kpc]}$', size=16)
        axis10.annotate(r'$\mathrm{Disc}$', xy=(-25, 25), xycoords='data', size=16)
        axis11.annotate(r'$\mathrm{Bulge}$', xy=(-25, 25), xycoords='data', size=16)
        
        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(
            stellar_data_tmp)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        prc_unit_vector = prc_angular_momentum / np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis]
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
        
        # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        
        # Plot the 2D surface density projection and scatter for the disc #
        disc_mask, = np.where(angular_theta_from_densest < (np.pi / 6.0))
        weights = stellar_data_tmp['Mass'][disc_mask]
        vmin, vmax = 0, 5e7
        
        count, xedges, yedges = np.histogram2d(stellar_data_tmp['Coordinates'][disc_mask, 0], stellar_data_tmp['Coordinates'][disc_mask, 1],
                                               weights=weights, bins=500, range=[[-30, 30], [-30, 30]])
        axis10.imshow(count.T, extent=[-30, 30, -30, 30], origin='lower', cmap='nipy_spectral_r', vmin=vmin, interpolation='gaussian', aspect='equal')
        
        count, xedges, yedges = np.histogram2d(stellar_data_tmp['Coordinates'][disc_mask, 0], stellar_data_tmp['Coordinates'][disc_mask, 2],
                                               weights=weights, bins=500, range=[[-30, 30], [-30, 30]])
        axis20.imshow(count.T, extent=[-30, 30, -30, 30], origin='lower', cmap='nipy_spectral_r', vmin=vmin, interpolation='gaussian', aspect='equal')
        
        # Plot the 2D surface density projection and scatter for the bulge #
        bulge_mask, = np.where(angular_theta_from_densest > (np.pi / 6.0))
        
        weights = stellar_data_tmp['Mass'][bulge_mask]
        count, xedges, yedges = np.histogram2d(stellar_data_tmp['Coordinates'][bulge_mask, 0], stellar_data_tmp['Coordinates'][bulge_mask, 1],
                                               weights=weights, bins=500, range=[[-30, 30], [-30, 30]])
        im = axis11.imshow(count.T, extent=[-30, 30, -30, 30], origin='lower', cmap='nipy_spectral_r', vmin=vmin, interpolation='gaussian',
                         aspect='equal')
        
        count, xedges, yedges = np.histogram2d(stellar_data_tmp['Coordinates'][bulge_mask, 0], stellar_data_tmp['Coordinates'][bulge_mask, 2],
                                               weights=weights, bins=500, range=[[-30, 30], [-30, 30]])
        axis21.imshow(count.T, extent=[-30, 30, -30, 30], origin='lower', cmap='nipy_spectral_r', vmin=vmin, interpolation='gaussian', aspect='equal')
        
        cbar = plt.colorbar(im, cax=axiscbar, orientation='horizontal')
        cbar.set_label(r'$\mathrm{\Sigma_{\bigstar}\,[M_\odot\,kpc^{-2}]}$', size=16)
        axiscbar.xaxis.tick_top()
        axiscbar.xaxis.set_label_position("top")
        axiscbar.tick_params(direction='out', which='both', right='on', labelsize=16)
        
        # Save the figure #
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'SD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/SD/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SpatialDistribution(simulation_path, tag)
