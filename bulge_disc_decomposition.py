import re
import time
import warnings
import argparse

import numpy as np
import seaborn as sns
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create 2 dimensional histograms of the bulge/disc decomposition.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class BulgeDiscDecomposition:
    """
    A class to create 2 dimensional histograms of the bulge/disc decomposition.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise empty arrays to hold the data #
        disc_fractions = []
        glx_stellar_masses = []
        
        if not args.l:
            self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all the accepted haloes
        for group_number in range(1, 5):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, disc_fraction = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    disc_fractions.append(disc_fraction)
                    glx_stellar_masses.append(np.sum(stellar_data_tmp['Mass']))
                    
                    # Save data in numpy arrays #
                    if group_number == 4:
                        np.save(SavePath + 'disc_fractions', disc_fractions)
                        np.save(SavePath + 'glx_stellar_masses', glx_stellar_masses)
                    print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, disc_fraction = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    disc_fractions.append(disc_fraction)
                    glx_stellar_masses.append(np.sum(stellar_data_tmp['Mass']))
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1  # Increace the count by one.
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    disc_fractions = np.load(SavePath + 'disc_fractions' + '.npy')
                    glx_stellar_masses = np.load(SavePath + 'glx_stellar_masses' +  '.npy')
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(glx_stellar_masses, disc_fractions)
        print('Plotted data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished BulgeDiscDecomposition for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(sim, tag):
        """
         A static method to extract particle and subhalo attributes.
        :param sim: simulation directory
        :param tag: redshift folder
        :return: stellar_data, subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, sim, tag, '/Subhalo/' + attribute, numThreads=4)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=4)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        A method to mask haloes.
        :return: subhalo_data_tmp
        """
        
        # Mask the data to select haloes more #
        mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 2.5e8)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        A method to mask galaxies.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, disc_fraction
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data_tmp['GroupNumber'] == group_number)[0][subgroup_number]
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(np.subtract(self.stellar_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index]), axis=1) <= 30.0))
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass']))  # km s-1
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)
        
        # Compute the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # Msun kpc km s-1
        
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # Msun kpc km s-1
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        # Select particles whose angular momentum projected along the rotation axis is negative and compute the disc-to-total ratio #
        norm_projection = np.sum(np.multiply(prc_unit_vector, glx_unit_vector), axis=1)
        index = np.where(np.arccos(norm_projection) > np.pi / 2)
        disc_fraction = 1 - np.divide(2 * np.sum(stellar_data_tmp['Mass'][index]), np.sum(stellar_data_tmp['Mass']))
        # if disc_fraction < 0.0:
        #     print(len(norm_projection))
        #     print(len(index[0]))
        #     print(len(stellar_data_tmp['Mass']))
        #     print(disc_fraction)
        #     print(glx_unit_vector)
        #
        #     plt.close()
        #     figure = plt.figure(0, figsize=(10, 10))
        #     plt.text(0.0, 100, disc_fraction)
        #     plt.hist(norm_projection)
        #     plt.savefig(outdir + 'BDD' + '-' + str(group_number) + '.png', bbox_inches='tight')
        
        return stellar_data_tmp, disc_fraction
    
    
    def plot(self, glx_stellar_masses, disc_fractions):
        """
        A method to plot a hexbin histogram.
        :return: None
        """
        
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)
        
        # Generate the figures #
        plt.close()
        figure = plt.figure(0, figsize=(10, 10))
        
        gs = gridspec.GridSpec(1, 2, width_ratios=(30, 1))
        ax = plt.subplot(gs[:, :1])
        axcbar = plt.subplot(gs[:, 1])
        
        # plt.xlim(-15, 15)
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(r'$\mathrm{D/T}$')
        ax.set_xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
        ax.tick_params(direction='in', which='both', top='on', right='on')
        
        # Generate the XY projection #
        hexbin = ax.scatter(glx_stellar_masses, disc_fractions)  # , xscale='log', cmap='bone', gridsize=50, edgecolor='none', mincnt=1)
        
        # Generate the color bar #
        # cbar = plt.colorbar(hexbin, cax=axcbar)
        # cbar.set_label('$\mathrm{Particles\; per\; hexbin}$')
        
        # Save the plot #
        plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + 'BDD' + '-' + date + '.png', bbox_inches='tight')
        
        return None


if __name__ == '__main__':
    tag = '010_z005p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_16/data/'
    outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/BDD/G-EAGLE/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/BDD/G-EAGLE/'  # Path to save/load data.
    # tag = '027_z000p101'
    # sim = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
    # outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RDSD/EAGLE/'  # Path to save plots.
    # SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/RDSD/EAGLE/'  # Path to save/load data.
    x = BulgeDiscDecomposition(sim, tag)