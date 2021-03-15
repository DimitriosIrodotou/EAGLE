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


class DiscToTotalVsMorphologicalParameters:
    """
    For all galaxies create: a disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over
    dispersion plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        glx_disc_fractions = np.load(data_path + 'glx_disc_fractions.npy')
        glx_circularities = np.load(data_path + 'glx_circularities.npy')
        glx_kappas_corotation = np.load(data_path + 'glx_kappas_corotation.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_rotationals_over_dispersions = np.load(data_path + 'glx_rotationals_over_dispersions.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_disc_fractions, glx_kappas_corotation, glx_disc_fractions_IT20, glx_circularities, glx_rotationals_over_dispersions)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished DiscToTotalVsMorphologicalParameters for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_disc_fractions, glx_kappas_corotation, glx_disc_fractions_IT20, glx_circularities, glx_rotationals_over_dispersions):
        """
        Plot the disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over dispersion.
        :param glx_disc_fractions: where the spheroid is assumed to have zero net angualr momentum.
        :param glx_kappas_corotation: defined as angular kinetic energy over kinetic energy.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :param glx_circularities: defined as Jz/Jz,max(E).
        :param glx_rotationals_over_dispersions: defined as vrot/sigam
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(25, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00, axis01, axis02, axis03 = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]), figure.add_subplot(gs[0, 2]), figure.add_subplot(
            gs[0, 3])
        axis10, axis11, axis12, axis13 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(
            gs[1, 3])

        plot_tools.set_axis(axis10, xlim=[0, 0.95], ylim=[0, 1], xlabel=r'$\mathrm{D/T_{\vec{J}_{b}=0}}$',
                            ylabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', aspect=None, which='major')
        plot_tools.set_axis(axis11, xlim=[0, 0.95], ylim=[0, 1], xlabel=r'$\mathrm{\overline{\epsilon}}$', aspect=None, which='major')
        plot_tools.set_axis(axis12, xlim=[0, 4.20], ylim=[0, 1], xlabel=r'$\mathrm{V_{rot}/\sigma}$', aspect=None, which='major')
        plot_tools.set_axis(axis13, xlim=[0.05, 0.95], ylim=[0, 1], xlabel=r'$\mathrm{\kappa_{co}}$', aspect=None, which='major')
        for axis in [axis11, axis12, axis13]:
            axis.set_yticklabels([])

        axes = [axis10, axis11, axis12, axis13]
        axescbar = [axis00, axis01, axis02, axis03]
        thresholds = [0.50, 0.3, 1.00, 0.4]
        x_attributes = [glx_disc_fractions, glx_circularities, glx_rotationals_over_dispersions, glx_kappas_corotation]
        for axis, axiscbar, x_attribute, threshold in zip(axes, axescbar, x_attributes, thresholds):
            # Plot attributes #
            hb = axis.hexbin(x_attribute, glx_disc_fractions_IT20, bins='log', gridsize=50, cmap='terrain_r')
            plot_tools.create_colorbar(axiscbar, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')

            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(x_attribute, glx_disc_fractions_IT20, bin_type='equal_width', n_bins=20,
                                                                           log=False)
            median, = axis.plot(x_value, median, color='black', linewidth=3)
            axis.fill_between(x_value, shigh, slow, color='black', alpha=0.3)
            fill, = plt.fill(np.NaN, np.NaN, color='black', alpha=0.3)

            axis.axvline(x=threshold, c='tab:red')  # Plot threshold lines.

        # Create the legends, save and close the figure #
        axis11.legend([median], [r'$\mathrm{Median}$'], frameon=False, fontsize=20, loc='upper right')
        axis12.legend([fill], [r'$\mathrm{16^{th}-84^{th}\;\%ile}$'], frameon=False, fontsize=20, loc='upper left')
        plt.savefig(plots_path + 'DTT_MP' + '-' + date + '.pdf', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsMorphologicalParameters(simulation_path, tag)
