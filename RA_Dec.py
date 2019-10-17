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
outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RADec/'  # Path to save plots.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class RADec:
    """
    A class to create RA and Dec plot.
    """


    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """

        # Load data #
        self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
        self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
        print('Reading data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' took %.5s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––')

        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.

        for group_number in range(1, self.Ngroups):  # Loop over all GroupNumber.
            for subgroup_number in range(0, 1):  # Loop over all distinct SubGroupNumber.
                start_local_time = time.time()  # Start the local time.
                stellar_data_tmp, mask = self.mask_galaxies(group_number, subgroup_number)  # Mask the data
                print('Masking data for halo ' + str(group_number) + ' took %.5s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')

                # Plot the data #
                if len(mask[0]) > 0.0:
                    start_local_time = time.time()  # Start the local time.
                    self.plot(stellar_data_tmp, group_number, subgroup_number)
                    print('Plotting data took %.5s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––')

        print('Finished PositionHistogram.py in %.5s s' % (time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def read_galaxies(sim, tag):
        """
         A static method to extract particle and subhalo attribute.
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
        for attribute in ['Coordinates', 'GroupNumber', 'SubGroupNumber']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=4)

        # Convert to astronomical units #
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
        :return: stellar_data_tmp, mask
        """

        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data_tmp['GroupNumber'] == group_number)[0][subgroup_number]

        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][index], axis=1) <= 30.0) & (
                            self.subhalo_data_tmp['ApertureMeasurements/Mass/030kpc'][:, 4][index] > 1e8))

        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute][mask])

        # Normalise the coordinates wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index])

        return stellar_data_tmp, mask


    @staticmethod
    def plot(stellar_data_tmp, group_number, subgroup_number):
        """
        A method to plot a hexbin histogram.
        :param stellar_data_tmp: temporary data
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
        plt.savefig(outdir + 'RADec' + '-' + str(group_number) + str(subgroup_number) + '-' + date + '.png', bbox_inches='tight')

        return None


if __name__ == '__main__':
    tag = '010_z005p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    x = PositionHistogram(sim, tag)