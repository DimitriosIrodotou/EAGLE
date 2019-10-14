import argparse
import re
import time
import warnings
import astropy.units as u
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import gridspec
import eagle_IO.eagle_IO.eagle_IO as E

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create 2 dimensional histograms of the bulge/disc decomposition.')
parser.add_argument('-r', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/'  # Path to save plots.
SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/BDD/'  # Path to save data.
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

        disc_fraction = []
        glx_stellar_mass = []

        if args.r:  # Read data.
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Reading data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' took %.5s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
            # Loop over all distinct GroupNumber and SubGroupNumber, mask the data and save to numpy arrays #
            for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):
                for subgroup_number in range(0, 1):
                    start_local_time = time.time()  # Start the local time.
                    self.stellar_data_tmp, self.disc_fraction = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    disc_fraction.append(self.disc_fraction)
                    glx_stellar_mass.append(np.sum(self.stellar_data_tmp['Mass']))
                    print('Masking data for halo ' + str(group_number) + ' took %.5s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
                    np.save(SavePath + 'disc_fraction', disc_fraction)
                    np.save(SavePath + 'glx_stellar_mass', glx_stellar_mass)

        else:  # Load data.
            disc_fraction = np.load(SavePath + 'disc_fraction.npy')
            glx_stellar_mass = np.load(SavePath + 'glx_stellar_mass.npy')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        self.plot(glx_stellar_mass, disc_fraction)
        print('Plotting data took %.5s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished BulgeDiscDecomposition.py in %.5s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')


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
        stellar_data['Velocity'] *= u.cm.to(u.km)
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
        mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e8)

        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[mask]

        return subhalo_data_tmp


    def mask_galaxies(self, group_number, subgroup_number):
        """
        A method to mask galaxies.
        :param group_number: from list(set(self.subhalo_data['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data['SubGroupNumber']))
        :return: stellar_data_tmp, mask
        """

        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data['GroupNumber'] == group_number)[0][subgroup_number]

        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data['CentreOfPotential'][index], axis=1) <= 30.0))

        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[mask]

        # Normalise the coordinates and velocities wrt the centre of mass of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data['CentreOfPotential'][index])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass']))
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)

        # Compute the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'])
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))

        # Select particles whose angular momentum projected along the rotation axis is negative and compute the disc-to-total ratio #
        index = np.where(np.sum(unit_vector * prc_angular_momentum, axis=0) <= 0.0)
        disc_fraction = 1 - np.divide(2 * np.sum(stellar_data_tmp['Mass'][index]), np.sum(stellar_data_tmp['Mass']))

        return stellar_data_tmp, disc_fraction


    def plot(self, glx_stellar_mass, disc_fraction):
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
        ax.set_ylim(-0.3, 1.2)
        ax.set_ylabel(r'$\mathrm{D/T}$')
        ax.set_xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
        ax.tick_params(direction='in', which='both', top='on', right='on')

        # Generate the XY projection #
        hexbin = ax.hexbin(glx_stellar_mass, disc_fraction, bins='log', xscale='log', cmap='bone', gridsize=50, edgecolor='none', mincnt=1)

        # Generate the color bar #
        cbar = plt.colorbar(hexbin, cax=axcbar)
        cbar.set_label('$\mathrm{log_{10}(Particles\; per\; hexbin)}$')

        # Save the plot #
        plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + 'BDD' + '-' + date + '.png', bbox_inches='tight')

        return None


if __name__ == '__main__':
    tag = '010_z005p000'
    for i in range(6, 7):
        sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_0' + str(i) + '/data/'
        x = BulgeDiscDecomposition(sim, tag)