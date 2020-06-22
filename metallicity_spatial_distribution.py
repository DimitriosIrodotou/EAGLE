import os
import re
import time
import warnings
import argparse
import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec
from astropy_healpix import HEALPix
from rotate_galaxies import RotateCoordinates

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create ra and dec plot.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class MetallicitySpatialDistribution:
    """
    For each galaxy create spatial distribution maps colour-coded by metallicity.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        p = 1  # Counter.
        stellar_data_tmp = {}  # Initialise a dictionary to store the data.
        
        if not args.l:
            # Extract particle and subhalo attributes and convert them to astronomical units #
            self.stellar_data, self.subhalo_data = self.read_attributes(simulation_path, tag)
            print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all masked haloes.
        for group_number in range(1, 26):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):  # Get centrals only.
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    # Save data in numpy arrays #
                    stellar_data_tmp = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
                    print('Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                        time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    print('Masked data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                        time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    # Load data from numpy arrays #
                    stellar_data_tmp = np.load(
                        data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                        allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MetallicitySpatialDistribution for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        :return: stellar_data, subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e9)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: stellar_data_tmp
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        halo_mask, = np.where((self.subhalo_data_tmp['GroupNumber'] == group_number) & (self.subhalo_data_tmp['SubGroupNumber'] == subgroup_number))
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        galaxy_mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask], axis=1) <= 30.0))  # In kpc.
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[galaxy_mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = stellar_data_tmp['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask]
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass'], axis=0))  # In km s^-1.
        stellar_data_tmp['Velocity'] = stellar_data_tmp['Velocity'] - CoM_velocity
        
        return stellar_data_tmp
    
    
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
        
        gs = gridspec.GridSpec(3, 2, hspace=0.07, wspace=0.0, height_ratios=[0.05, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10 = figure.add_subplot(gs[1, 0])
        axis20 = figure.add_subplot(gs[2, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        axis21 = figure.add_subplot(gs[2, 1])
        
        for axis in [axis10, axis20, axis11, axis21]:
            axis.grid(True, which='both', axis='both')
            axis.set_xlim(-30, 30)
            axis.set_ylim(-30, 30)
            axis.set_aspect('equal')
            axis.set_facecolor('black')
            axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=16)
        
        axis10.set_xticklabels([])
        axis11.set_xticklabels([])
        axis10.set_ylabel(r'$\mathrm{y\,[kpc]}$', size=16)
        axis20.set_xlabel(r'$\mathrm{x\,[kpc]}$', size=16)
        axis20.set_ylabel(r'$\mathrm{z\,[kpc]}$', size=16)
        axis11.set_ylabel(r'$\mathrm{y\,[kpc]}$', size=16)
        axis21.set_xlabel(r'$\mathrm{x\,[kpc]}$', size=16)
        axis21.set_ylabel(r'$\mathrm{z\,[kpc]}$', size=16)
        axis10.annotate(r'$\mathrm{Disc}$' + '\n' + r'$\mathrm{Face-on}$', xy=(-27, 23), xycoords='data', c='w', size=16)
        axis20.annotate(r'$\mathrm{Disc}$' + '\n' + r'$\mathrm{Edge-on}$', xy=(-27, 23), xycoords='data', c='w', size=16)
        axis11.annotate(r'$\mathrm{Bulge}$' + '\n' + r'$\mathrm{Face-on}$', xy=(-27, 23), xycoords='data', c='w', size=16)
        axis21.annotate(r'$\mathrm{Bulge}$' + '\n' + r'$\mathrm{Edge-on}$', xy=(-27, 23), xycoords='data', c='w', size=16)
        
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
        
        # Plot the 2D surface density projection for the disc colour-coded by metallicity #
        disc_mask, = np.where(angular_theta_from_densest < (np.pi / 6.0))
        bulge_mask, = np.where(angular_theta_from_densest > (np.pi / 6.0))
        
        axes1 = [axis10, axis11]
        axes2 = [axis20, axis21]
        masks = [disc_mask, bulge_mask]
        for mask, axis1, axis2 in zip(masks, axes1, axes2):
            axis1.scatter(stellar_data_tmp['Coordinates'][mask, 0], stellar_data_tmp['Coordinates'][mask, 1],
                        c=stellar_data_tmp['Metallicity'][mask] / 0.0134, cmap='Spectral', s=1)
            scatter = axis2.scatter(stellar_data_tmp['Coordinates'][mask, 0], stellar_data_tmp['Coordinates'][mask, 2],
                                  c=stellar_data_tmp['Metallicity'][mask] / 0.0134, cmap='Spectral', s=1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, cax=axiscbar, orientation='horizontal')
        cbar.set_label(r'$\mathrm{Z\;[Z_{\odot}]}$', size=16)
        axiscbar.xaxis.tick_top()
        axiscbar.xaxis.set_label_position("top")
        axiscbar.tick_params(direction='out', which='both', right='on', labelsize=16)
        
        # Save the figure #
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'MSP' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/MSP/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = MetallicitySpatialDistribution(simulation_path, tag)
