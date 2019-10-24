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
parser = argparse.ArgumentParser(description='Create RA and Dec.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RD/'  # Path to save plots.
SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/RD/'  # Path to save data.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADec:
    """
    A class to create RA and Dec.
    """


    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """

        p = 1  # Counter.

        self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
        self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
        print('Read data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––')

        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.

        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all the accepted haloes
        for group_number in range(1, 2):  # Loop over all the accepted haloes
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.

                    self.stellar_data_tmp, unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.

                    # Save data in nampy arrays #
                    np.save(SavePath + 'group_number_' + str(group_number), group_number)
                    np.save(SavePath + 'subgroup_number_' + str(group_number), subgroup_number)
                    np.save(SavePath + 'unit_vector_' + str(group_number), unit_vector)
                    print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1

                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.

                    self.stellar_data_tmp, unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1

                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.

                    group_number = np.load(SavePath + 'group_number_' + str(group_number) + '.npy')
                    subgroup_number = np.load(SavePath + 'subgroup_number_' + str(group_number) + '.npy')
                    unit_vector = np.load(SavePath + 'unit_vector_' + str(group_number) + '.npy')
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')

                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                self.plot(self.stellar_data_tmp, unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')

        print('Finished RADec for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')


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
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'SubGroupNumber', 'ParticleBindingEnergy', 'Velocity']:
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
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, unit_vector
        """

        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data_tmp['GroupNumber'] == group_number)[0][subgroup_number]

        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][index], axis=1) <= 30.0))

        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[mask]

        # Normalise the coordinates and velocities wrt the centre of mass of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass']))
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)

        # Compute the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'])
        unit_vector = (prc_angular_momentum / np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])

        return stellar_data_tmp, unit_vector


    @staticmethod
    def plot(stellar_data_tmp, unit_vector, group_number, subgroup_number):
        """
        A method to plot a hexbin histogram.
        :param unit_vector: from mask_galaxies
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: None
        """

        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)

        # Generate the figures #
        plt.close()
        figure = plt.figure(0, figsize=(10, 7.5))

        gs = gridspec.GridSpec(1, 2, width_ratios=(30, 1))
        ax = plt.subplot(gs[:, :1], projection="mollweide")
        axcbar = plt.subplot(gs[:, 1])

        ax.grid(True)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

        y_tick_labels = np.array(['', '-60', '', '-30', '', 0, '', '30', '', 60])
        x_tick_labels = np.array(['', '-120', '', '-60', '', 0, '', '60', '', 120])
        ax.set_yticklabels(y_tick_labels)
        ax.set_xticklabels(x_tick_labels)
        ax.tick_params(axis='x', colors='white')

        # Generate the RA and Dec projection #
        # hexbin = ax.hexbin(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]), bins='log', cmap='copper', gridsize=100,
        #                    edgecolor='none', mincnt=1, zorder=-1)  # Element-wise arctan of x1/x2.
        hexbin = ax.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]),
                            c=np.linalg.norm(stellar_data_tmp['Velocity'],axis=1), cmap='hsv', s=1)

        # Generate the color bar #
        cbar = plt.colorbar(hexbin, cax=axcbar)
        cbar.set_label('$\mathrm{Particles\; per\; hexbin}$')

        # Save the plot #
        plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + 'RD' + '-' + str(group_number) + str(subgroup_number) + '-' + date + '.png', bbox_inches='tight')

        return None


if __name__ == '__main__':
    # tag = '027_z000p101'
    tag = '010_z005p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    # sim = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
    x = RADec(sim, tag)