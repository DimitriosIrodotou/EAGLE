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

date = time.strftime('%d_%m_%y_%H%M')  # Date
outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/'  # Path to save plots.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class MassSize:
    """
    A class to create 2 dimensional histograms of the mass-size relation of galaxies.
    """


    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """

        # Load data #
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
        print('--- Finished reading the data in %.5s seconds ---' % (time.time() - start_global_time))  # Print reading time.
        print('–––––––––––––––––––––––––––––––––––––––––')

        print('Found ' + str(len(list(set(self.subhalo_data['GroupNumber'])))) + ' FoF groups and ' + str(
            len(list(set(self.subhalo_data['SubGroupNumber'])))) + ' subfind groups ' + 'for G-EAGLE_' + re.split('GEAGLE_|/data', sim)[2])

        for group_number in np.sort(list(set(self.subhalo_data['GroupNumber']))):  # Loop over all distinct GroupNumber.
            for subgroup_number in range(0, 1):  # Loop over all distinct SubGroupNumber.
                stellar_data_tmp, mask = self.mask_galaxies(group_number, subgroup_number)  # Mask the data

                # Plot the data #
                if len(mask[0]) > 0.0:
                    start_local_time = time.time()  # Start the local time.
                    self.plot(stellar_data_tmp, group_number, subgroup_number)
                    print('--- Finished plotting the data in %.5s seconds ---' % (time.time() - start_local_time))  # Print plotting time.
                    print('–––––––––––––––––––––––––––––––––––––––––')

        print('--- Finished MassSize.py in %.5s seconds ---' % (time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def read_galaxies(sim, tag):
        """
         Extract particle and subhalo attributes and convert them to astronomical units.
        :param sim: simulation directory
        :param tag: redshift folder
        :return: stellar_data, subhalo_data
        """

        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['GroupNumber', 'SubGroupNumber', 'CentreOfPotential', 'ApertureMeasurements/Mass/030kpc']:
            subhalo_data[attribute] = E.read_array(file_type, sim, tag, '/Subhalo/' + attribute, numThreads=4)

        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['GroupNumber', 'SubGroupNumber', 'Coordinates']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=4)

        # Convert to astronomical units #
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)

        return stellar_data, subhalo_data


    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from list(set(self.subhalo_data['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data['SubGroupNumber']))
        :return: stellar_data_tmp, mask
        """
        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data['GroupNumber'] == group_number)[0][subgroup_number]

        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.sqrt(np.sum((self.stellar_data['Coordinates'] - self.subhalo_data['CentreOfPotential'][index]) ** 2, axis=1)) <= 30.0) & (
                            self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4][index] > 1e8))

        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[mask]

        # Normalise the coordinates wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data['CentreOfPotential'][index])

        return stellar_data_tmp, mask


    @staticmethod
    def plot(stellar_data_tmp, group_number, subgroup_number):
        """
        A method to plot a hexbin histogram.
        :param stellar_data_tmp: temporary data
        :param group_number: from list(set(self.subhalo_data['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data['SubGroupNumber']))
        :return: None
        """

        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)

        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(0, figsize=(10, 10))

        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=(20, 1))
        gs.update(hspace=0.2)
        axtop = plt.subplot(gs[0, 0])
        axbot = plt.subplot(gs[1, 0])
        axcbar = plt.subplot(gs[:, 1])

        # Generate the XY projection #
        axtop.set_xlim(-15, 15)
        axtop.set_ylim(-15, 15)
        axtop.set_xlabel(r'$\mathrm{x/kpc}$')
        axtop.set_ylabel(r'$\mathrm{y/kpc}$')
        axtop.tick_params(direction='in', which='both', top='on', right='on')
        axtop.set_facecolor('k')

        pltop = axtop.hexbin(list(zip(*stellar_data_tmp['Coordinates']))[0], list(zip(*stellar_data_tmp['Coordinates']))[1], bins='log', cmap='bone',
                             gridsize=300, edgecolor='none')

        # Generate the XZ projection #
        axbot.set_xlim(-15, 15)
        axbot.set_ylim(-15, 15)
        axbot.set_xlabel(r'$\mathrm{x/kpc}$')
        axbot.set_ylabel(r'$\mathrm{z/kpc}$')
        axbot.set_facecolor('k')
        plbot = axbot.hexbin(list(zip(*stellar_data_tmp['Coordinates']))[0], list(zip(*stellar_data_tmp['Coordinates']))[2], bins='log', cmap='bone',
                             gridsize=300, edgecolor='none')

        # Generate the color bar #
        cbar = plt.colorbar(pltop, cax=axcbar)
        cbar.set_label('$\mathrm{log_{10}(Particles\; per\; hexbin)}$')

        # Save the plot #
        plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + 'PH' + '-' + str(group_number) + str(subgroup_number) + '-' + date + '.png', bbox_inches='tight')

        return None


if __name__ == '__main__':
    tag = '010_z005p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    x = MassSize(sim, tag)