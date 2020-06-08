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

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DeltaRVsDeltaTheta:
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
        plt.close()
        plt.figure(0, figsize=(10, 10))
        plt.xscale('log')
        plt.xlabel('delta r')
        plt.ylabel('delta theta')
        plt.yscale('log')
        for group_number in range(1, 100):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):  # Get centrals only.
                start_local_time = time.time()  # Start the local time.
                
                # Load data from numpy arrays #
                stellar_data_tmp = np.load(
                    data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy', allow_pickle=True)
                stellar_data_tmp = stellar_data_tmp.item()
                
                subhalo_data_tmp = np.load(
                    data_path + 'subhalo_data_tmps/subhalo_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy', allow_pickle=True)
                subhalo_data_tmp = subhalo_data_tmp.item()
                
                gaseous_data_tmp = np.load(
                    data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy', allow_pickle=True)
                gaseous_data_tmp = gaseous_data_tmp.item()
                
                dark_matter_data_tmp = np.load(
                    data_path + 'dark_matter_data_tmps/dark_matter_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                    allow_pickle=True)
                dark_matter_data_tmp = dark_matter_data_tmp.item()
                
                print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, subhalo_data_tmp, gaseous_data_tmp, dark_matter_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_data_tmp, subhalo_data_tmp, gaseous_data_tmp, dark_matter_data_tmp, group_number, subgroup_number):
        """
        Plot a HEALPix histogram from the angular momentum of particles - an angular distance plot - a surface density plot / gri mock image - a
        circularity distribution.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        
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
        
        # Find location of density maximum and plot its positions and the ra and dec of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate and plot the angular distance between the (unit vector of) the galactic angular momentum and all the other grid cells #
        position_of_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T
        
        angular_theta_from_X = np.arccos(
            np.sin(position_of_X[0, 1]) * np.sin(lat_densest) + np.cos(position_of_X[0, 1]) * np.cos(lat_densest) * np.cos(
                position_of_X[0, 0] - lon_densest))  # In radians.
        
        CoM = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Coordinates'], axis=0) + np.sum(
            gaseous_data_tmp['Mass'][:, np.newaxis] * gaseous_data_tmp['Coordinates'], axis=0) + np.sum(
            dark_matter_data_tmp['Mass'][:, np.newaxis] * dark_matter_data_tmp['Coordinates'], axis=0),
                        np.sum(stellar_data_tmp['Mass'], axis=0) + np.sum(gaseous_data_tmp['Mass'], axis=0) + np.sum(dark_matter_data_tmp['Mass'],
                                                                                                                     axis=0))  # In km s-1.
        
        halo_mask, = np.where((subhalo_data_tmp['GroupNumber'] == group_number) & (subhalo_data_tmp['SubGroupNumber'] == subgroup_number))
        all_masses = np.hstack([stellar_data_tmp['Mass'], gaseous_data_tmp['Mass'], dark_matter_data_tmp['Mass']])
        all_rs = np.hstack(
            [np.sqrt(np.sum(stellar_data_tmp['Coordinates'] ** 2, axis=1)), np.sqrt(np.sum(gaseous_data_tmp['Coordinates'] ** 2, axis=1)),
             np.sqrt(np.sum(dark_matter_data_tmp['Coordinates'] ** 2, axis=1))])
        
        mass, edges = np.histogram(all_rs, bins=50, range=(0, 30), weights=all_masses)
        centers = 0.5 * (edges[1:] + edges[:-1])
        surface = 1.333 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        den = np.divide(mass, surface)
        print(den)
        
        delta_r = np.linalg.norm(subhalo_data_tmp['CentreOfPotential'][halo_mask]) - np.linalg.norm(CoM)
        
        plt.scatter(delta_r, angular_theta_from_X, color='black')
        
        plt.savefig(plots_path + 'DRDT' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/DRDT/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = DeltaRVsDeltaTheta(simulation_path, tag)
