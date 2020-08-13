import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

from matplotlib import gridspec

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DTTVsEnvironment:
    """
    For all galaxies create a disc to total ratio as a function of: centrals and satellites PDF, number of satellites, the angle between angular
    momentum of gaseous stellar and components and disc and spheroid plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        group_numbers = np.load(data_path + 'group_numbers.npy')
        subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        glx_gaseous_angular_momenta = np.load(data_path + 'glx_gaseous_angular_momenta.npy')
        disc_stellar_angular_momenta = np.load(data_path + 'disc_stellar_angular_momenta.npy')
        spheroid_stellar_angular_momenta = np.load(data_path + 'spheroid_stellar_angular_momenta.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(group_numbers, subgroup_numbers, glx_disc_fractions_IT20, glx_stellar_angular_momenta, glx_gaseous_angular_momenta,
                  disc_stellar_angular_momenta, spheroid_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished DTTVsEnvironment for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(group_numbers, subgroup_numbers, glx_disc_fractions_IT20, glx_stellar_angular_momenta, glx_gaseous_angular_momenta,
             disc_stellar_angular_momenta, spheroid_stellar_angular_momenta):
        """
        Plot the centrals and satellites PDF, number of satellites, the angle between angular momentum of gaseous stellar and components and disc
        and spheroid.
        :param group_numbers: unique halo number.
        :param subgroup_numbers: unique subhalo number.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        pixel.
        :param glx_stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :param glx_gaseous_angular_momenta: defined as the sum of each gaseous particle's angular momentum.
        :param disc_stellar_angular_momenta: defined as the sum of each disc particle's angular momentum.
        :param spheroid_stellar_angular_momenta: defined as the sum of each spheroid particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(25, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis02, axis03 = figure.add_subplot(gs[0, 2]), figure.add_subplot(gs[0, 3])
        axis10, axis11, axis12, axis13 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(
            gs[1, 3])
        plot_tools.set_axis(axis10, ylim=[0, 1], xlabel=r'$\mathrm{PDF}$', ylabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', aspect=None,
                            which='major')
        plot_tools.set_axis(axis11, xlim=[9e-1, 2e2], ylim=[0, 1], xlabel=r'$\mathrm{N_{satellites}}$', aspect=None, which='major')
        plot_tools.set_axis(axis12, xlim=[-0.99, 1.01],
                            xlabel=r'$\mathrm{(\vec{J}_{disc}\cdot\vec{J}_{spheroid})/(|\vec{J}_{disc}||\vec{J}_{spheroid}|)}$', ylim=[0, 1],
                            aspect=None, which='major')
        plot_tools.set_axis(axis13, xlim=[-0.99, 1.01],
                            xlabel=r'$\mathrm{(\vec{J}_{\bigstar}\cdot\vec{J}_{gas})/(|\vec{J}_{\bigstar}||\vec{J}_{gas}|)}$', ylim=[0, 1],
                            aspect=None, which='major')
        for axis in [axis11, axis12, axis13]:
            axis.set_yticklabels([])

        # Calculate the cosine of angle between gaseous and stellar components #
        angle = np.divide(np.sum(glx_stellar_angular_momenta * glx_gaseous_angular_momenta, axis=1),
                          np.linalg.norm(glx_stellar_angular_momenta, axis=1) * np.linalg.norm(glx_gaseous_angular_momenta, axis=1))

        # Calculate the cosine of angle between disc and spheroid components #
        angle_components = np.divide(np.sum(disc_stellar_angular_momenta * spheroid_stellar_angular_momenta, axis=1),
                                     np.linalg.norm(disc_stellar_angular_momenta, axis=1) * np.linalg.norm(spheroid_stellar_angular_momenta, axis=1))

        # Plot the angle between the angular momentum of gaseous and stellar components and disc and spheroid #
        axes = [axis12, axis13]
        axescbar = [axis02, axis03]
        x_attributes = [angle_components, angle]
        for axis, axiscbar, x_attribute in zip(axes, axescbar, x_attributes):
            hb = axis.hexbin(x_attribute, glx_disc_fractions_IT20, bins='log', cmap='terrain_r')
            plot_tools.create_colorbar(axiscbar, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')

            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(x_attribute, glx_disc_fractions_IT20, bin_type='equal_width', n_bins=25,
                                                                           log=False)
            median, = axis.plot(x_value, median, color='black', linewidth=3)
            axis.fill_between(x_value, shigh, slow, color='black', alpha='0.3')
            fill, = plt.fill(np.NaN, np.NaN, color='black', alpha=0.3)

        # Plot a histogram of the disc to total ratio for centrals and satellites #
        centrals_mask, = np.where(subgroup_numbers == 0)
        satellites_mask, = np.where(subgroup_numbers > 0)
        for mask, label in zip([centrals_mask, satellites_mask], [r'$\mathrm{Centrals}$', r'$\mathrm{Satellites}$']):
            axis10.hist(glx_disc_fractions_IT20[mask], density=True, bins=20, histtype='step', orientation='horizontal', label=label)
            axis10.invert_xaxis()

        # Plot the disc to total ratio as a function of number of satellites #
        unique_elements, counts_elements = np.unique(group_numbers, return_counts=True)
        for i in range(len(unique_elements)):
            mask, = np.where((subgroup_numbers == 0) & (group_numbers == unique_elements[i]))
            if len(mask) > 0:  # Avoid lone satellites.
                axis11.scatter(counts_elements[i], glx_disc_fractions_IT20[mask], color='black')
                axis11.set_xscale('log')

        # Create the legends, save and close the figure #
        axis12.legend([median], [r'$\mathrm{Median}$'], frameon=False, fontsize=16, loc='upper right')
        axis13.legend([fill], [r'$\mathrm{16^{th}-84^{th}\;\%ile}$'], frameon=False, fontsize=16, loc='upper left')
        axis10.legend(loc='upper center', fontsize=16, frameon=False, ncol=2)
        plt.savefig(plots_path + 'DTT_E' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DTTVsEnvironment(simulation_path, tag)
