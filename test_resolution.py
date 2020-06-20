import os
import re
import time
import warnings
import matplotlib

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import gridspec
from astropy_healpix import HEALPix
from rotate_galaxies import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class TestResolution:
    """
    For each galaxy create: a HEALPix histogram from the angular momentum of particles for different nside values.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        for group_number in range(25, 26):  # Loop over all masked haloes.
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
        
        print('Finished TestResolution for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
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
        plt.figure(0, figsize=(16, 9))
        
        gs = gridspec.GridSpec(2, 3)
        axis00 = plt.subplot(gs[0, 0], projection='mollweide')
        axis01 = plt.subplot(gs[0, 1], projection='mollweide')
        axis02 = plt.subplot(gs[0, 2], projection='mollweide')
        axis10 = plt.subplot(gs[1, 0], projection='mollweide')
        axis11 = plt.subplot(gs[1, 1], projection='mollweide')
        axis12 = plt.subplot(gs[1, 2], projection='mollweide')
        
        powers = (2, 3, 4, 5, 6, 7)
        axes = [axis00, axis01, axis02, axis10, axis11, axis12]
        for axis, power in zip(axes, powers):
            axis.set_xlabel('RA ($\degree$)')
            axis.set_ylabel('Dec ($\degree$)')
            
            # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum
            # vector #
            print(len(stellar_data_tmp['Mass']))
            print(np.log10(np.sum(stellar_data_tmp['Mass'])))
            prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                      stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
            glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)
            glx_unit_vector = glx_angular_momentum / np.linalg.norm(glx_angular_momentum)
            
            # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(
                stellar_data_tmp, glx_unit_vector)
            
            # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
            ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
            dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
            
            # Plot a HEALPix histogram #
            nside = 2 ** power  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
            hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
            indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
            density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.
            
            # Find location of density maximum and plot its positions and the ra (lon) and dec (lat) of the galactic angular momentum #
            index_densest = np.argmax(density)
            lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
            lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
            axis.annotate(r'$\mathrm{Density\;maximum}$', xy=(lon_densest, lat_densest), xycoords='data', xytext=(0.78, 1.00),
                          textcoords='axes fraction',
                          arrowprops=dict(arrowstyle='-', color='black', connectionstyle='arc3,rad=0'))  # Position of the densest pixel.
            axis.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=100, color='black', marker='X',
                         facecolors='none', zorder=5)  # Position of the galactic angular momentum.
            
            # Sample a 360x180 grid in ra/dec #
            ra = np.linspace(-180.0, 180.0, num=360) * u.deg
            dec = np.linspace(-90.0, 90.0, num=180) * u.deg
            ra_grid, dec_grid = np.meshgrid(ra, dec)
            
            # Find density at each coordinate position #
            coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
            density_map = density[coordinate_index]
            
            # Display data on a 2D regular raster and create a pseudo-color plot #
            im = axis.imshow(density_map, cmap='nipy_spectral_r', aspect='auto', norm=matplotlib.colors.LogNorm(vmin=1))
            cbar = plt.colorbar(im, ax=axis, orientation='horizontal')
            cbar.set_label('$\mathrm{Particles\; per\; grid\; cell}$', size=12)
            axis.pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
            
            # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
            angular_theta_from_densest = np.arccos(np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(
                np.arcsin(prc_unit_vector[:, 2])) * np.cos(lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
            disc_mask = np.where(angular_theta_from_densest < (np.pi / 6.0))
            disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][disc_mask]) / np.sum(stellar_data_tmp['Mass'])
            axis.text(0.3, 1.01, r'$\mathrm{DTT = %.3f}$' % disc_fraction_IT20, c='black', fontsize=12, transform=axis.transAxes)
        
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'TR' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/TR/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = TestResolution(simulation_path, tag)
