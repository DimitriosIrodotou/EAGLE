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


class DiscToTotalVsGalacticAttributes:
    """
    For all galaxies create: a disc to total ratio as a function of mass, angular momentum, gas fraction and star formation rate plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        glx_gaseous_masses = np.load(data_path + 'glx_gaseous_masses.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_star_formation_rates = np.load(data_path + 'glx_star_formation_rates.npy')
        glx_stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')

        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_gaseous_masses, glx_stellar_masses, glx_disc_fractions_IT20, glx_star_formation_rates, glx_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print(
            'Finished DTT_GP for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_gaseous_masses, glx_stellar_masses, glx_disc_fractions_IT20, glx_star_formation_rates, glx_stellar_angular_momenta):
        """
        Plot the disc to total ratio as a function of mass, angular momentum, gas fraction and star formation rate.
        :param glx_gaseous_masses: defined as the mass of all gaseous particles within 30kpc from the most bound particle.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_star_formation_rates: defined as the star formation rate of all gaseous particles within 30kpc from the most bound particle.
        :param glx_stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(25, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00, axis01, axis02, axis03 = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]), figure.add_subplot(gs[0, 2]), figure.add_subplot(
            gs[0, 3])
        axis10, axis11, axis12, axis13 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(
            gs[1, 3])

        plot_tools.set_axis(axis10, xlim=(5e9, 5e11), ylim=[0, 1], xlabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                            ylabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', xscale='log', aspect=None, which='major')
        plot_tools.set_axis(axis11, xlim=(1e1, 2e3), ylim=[0, 1], xlabel=r'$\mathrm{(|\vec{J}_{\bigstar}|/M_{\bigstar})/(kpc\;km\;s^{-1})}$',
                            xscale='log', aspect=None, which='major')
        plot_tools.set_axis(axis12, xlim=(1e-3, 5e-1), ylim=[0, 1], xlabel=r'$\mathrm{f_{gas}}$', xscale='log', aspect=None, which='major')
        plot_tools.set_axis(axis13, xlim=(2e-13, 2e-10), ylim=[0, 1], xlabel=r'$\mathrm{sSFR/yr^{-1}}$', xscale='log', aspect=None, which='major')
        for axis in [axis11, axis12, axis13]:
            axis.set_yticklabels([])

        fgas = np.divide(glx_gaseous_masses, glx_gaseous_masses + glx_stellar_masses)
        spc_stellar_angular_momenta = np.linalg.norm(glx_stellar_angular_momenta, axis=1) / glx_stellar_masses
        axes = [axis10, axis11, axis12, axis13]
        axescbar = [axis00, axis01, axis02, axis03]
        x_attributes = [glx_stellar_masses, spc_stellar_angular_momenta, fgas[fgas > 0],
                        glx_star_formation_rates[glx_star_formation_rates > 0] / glx_stellar_masses[glx_star_formation_rates > 0]]
        y_attributes = [glx_disc_fractions_IT20, glx_disc_fractions_IT20, glx_disc_fractions_IT20[fgas > 0],
                        glx_disc_fractions_IT20[glx_star_formation_rates > 0]]
        for axis, axiscbar, x_attribute, y_attribute in zip(axes, axescbar, x_attributes, y_attributes, ):
            # Plot attributes #
            hb = axis.hexbin(x_attribute, y_attribute, bins='log', xscale='log', gridsize=50, cmap='terrain_r')
            plot_tools.create_colorbar(axiscbar, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')

            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(x_attribute, y_attribute, bin_type='equal_width', n_bins=15, log=True)
            median, = axis.plot(x_value, median, color='black', linewidth=3)
            axis.fill_between(x_value, shigh, slow, color='black', alpha='0.3')
            fill, = plt.fill(np.NaN, np.NaN, color='black', alpha=0.3)

        # Create a legend, save and close the figure #
        axis11.legend([median], [r'$\mathrm{Median}$'], frameon=False, fontsize=16, loc='upper right')
        axis12.legend([fill], [r'$\mathrm{16^{th}-84^{th}\;\%ile}$'], frameon=False, fontsize=16, loc='upper left')
        plt.savefig(plots_path + 'DTT_GP' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsGalacticAttributes(simulation_path, tag)
