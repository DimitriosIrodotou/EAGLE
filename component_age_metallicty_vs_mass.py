import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib import gridspec

style.use("classic")
plt.rcParams.update({'font.family':'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class ComponentAgeMetallicityVsMass:
    """
    For all components create: a component birth-mass weighted mean age and metallicity as a function of galaxy stellar mass plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        disc_weighted_as = np.load(data_path + 'disc_weighted_as.npy')
        spheroid_weighted_as = np.load(data_path + 'spheroid_weighted_as.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_metallicities = np.load(data_path + 'disc_metallicities.npy')
        spheroid_metallicities = np.load(data_path + 'spheroid_metallicities.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(disc_weighted_as, spheroid_weighted_as, glx_stellar_masses, disc_metallicities, spheroid_metallicities, glx_disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished ComponentAgeMetallicityVsMass for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(disc_weighted_as, spheroid_weighted_as, glx_stellar_masses, disc_metallicities, spheroid_metallicities, glx_disc_fractions_IT20):
        """
        Plot the component average birth scale factor as a function of galaxy mass colour-coded by disc to total ratio.
        :param disc_weighted_as: defined as the mean birth scale factor for the disc component.
        :param spheroid_weighted_as: defined as the birth-mass weighted mean birth scale factor for the spheroid component.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param disc_metallicities: defined as the mass-weighted sum of each disc particle's metallicity.
        :param spheroid_metallicities: defined as the mass-weighted sum of each spheroid particle's metallicity.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 15))
        gs = gridspec.GridSpec(2, 1, hspace=0.0)
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        plot_tools.set_axis(axis00, xlim=[1e8, 6e11], ylim=[0.1, 1], xscale='log', ylabel=r'$\mathrm{\overline{\alpha}_{comp}}$', which='major',
                            aspect=None)
        plot_tools.set_axis(axis10, xlim=[1e8, 6e11], ylim=[0, 4.6], xscale='log', xlabel=r'$\mathrm{M_{comp}/M_{\odot}}$',
                            ylabel=r'$\mathrm{Z_{comp}/Z_{\odot}}$', which='major', aspect=None)
        axis00.set_xticklabels([])

        # Plot the average birth scale factor as a function of galaxy mass colour-coded by disc to total ratio #
        axis00.scatter(glx_disc_fractions_IT20 * glx_stellar_masses, disc_weighted_as, c='tab:blue', s=20, cmap='seismic_r', edgecolor='none')
        axis00.scatter((1 - glx_disc_fractions_IT20) * glx_stellar_masses, spheroid_weighted_as, c='tab:red', s=20, cmap='seismic_r',
                       edgecolor='none')

        # Plot median and 1-sigma lines #
        for component, ratio in zip([disc_weighted_as, spheroid_weighted_as], [glx_disc_fractions_IT20, 1 - glx_disc_fractions_IT20]):
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(ratio * glx_stellar_masses, component, bin_type='equal_width', n_bins=25,
                                                                           log=True)
            axis00.plot(x_value, median, color='black', linewidth=3)
            axis00.fill_between(x_value, shigh, slow, color='black', alpha=0.3)

        # Plot the component metallicity-mass relation colour-coded by disc to total ratio #
        d = axis10.scatter(glx_disc_fractions_IT20 * glx_stellar_masses, disc_metallicities, c='tab:blue', s=20, edgecolor='none')
        b = axis10.scatter((1 - glx_disc_fractions_IT20) * glx_stellar_masses, spheroid_metallicities, c='tab:red', s=20, edgecolor='none')

        # Plot median and 1-sigma lines #
        for component, ratio in zip([disc_metallicities, spheroid_metallicities], [glx_disc_fractions_IT20, 1 - glx_disc_fractions_IT20]):
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(ratio * glx_stellar_masses, component, bin_type='equal_width', n_bins=25,
                                                                           log=True)
            median, = axis10.plot(x_value, median, color='black', linewidth=3)
            axis10.fill_between(x_value, shigh, slow, color='black', alpha=0.3)
            fill, = plt.fill(np.NaN, np.NaN, color='black', alpha=0.3)

        # Create a legend, save and close the figure #
        axis10.legend([d, b, median, fill],
                      [r'$\mathrm{Discs}$', r'$\mathrm{Spheroids}$', r'$\mathrm{Median}$', r'$\mathrm{16^{th}-84^{th}\;\%ile}$'], frameon=False,
                      fontsize=20, loc='upper center', ncol=2, numpoints=1, scatterpoints=1)
        plt.savefig(plots_path + 'C_A_M_M' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentAgeMetallicityVsMass(simulation_path, tag)
