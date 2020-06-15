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

from astropy_healpix import HEALPix
from rotate_galaxies import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SampleRAEl:
    """
    Plot a sample of HEALPix histograms.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), subplot_kw={'projection':'mollweide'})
        for i, axis in enumerate(axis.flatten()):
            start_local_time = time.time()  # Start the local time.
            
            group_number = i + 1
            subgroup_number = 0
            
            # Load data from numpy arrays #
            stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            stellar_data_tmp = stellar_data_tmp.item()
            print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            # Plot the data #
            start_local_time = time.time()  # Start the local time.
            
            self.plot(axis, stellar_data_tmp, group_number, subgroup_number)
            print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        plt.savefig(plots_path + 'SRAEl' + '-' + date + '.png', bbox_inches='tight')
        print('Finished SampleRAEl for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(axis, stellar_data_tmp, group_number, subgroup_number):
        """
        Plot a sample of HEALPix histograms.
        :param axis: from __init__.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum
        # vector #
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
        
        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)
        
        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = density[coordinate_index]
        
        # Display data on a 2D regular raster and create a pseudo-color plot #
        axis.pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        
        # Define the figure parameters #
        axis.axis('off')
        plt.tight_layout()
        
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/SRAEl/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleRAEl(simulation_path, tag)
