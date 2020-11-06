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


class DiscToTotalCRVsDiscToTotal:
    """
    For all galaxies create: a disc to total ratio including counter-rotating particles as a function of original disc to total ratio plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_disc_fractions_IT20_cr = np.load(data_path + 'glx_disc_fractions_IT20_cr_all.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        chi = (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20_cr = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20_cr - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_disc_fractions_IT20, glx_disc_fractions_IT20_cr)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished DiscToTotalVsMorphologicalParameters for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_disc_fractions_IT20, glx_disc_fractions_IT20_cr):
        """
        Plot the disc to total ratio including counter-rotating particles as a function of original disc to total.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :param , glx_disc_fractions_IT20_cr: defined as , glx_disc_fractions_IT20 but also includes in the disc counter-rotating structures.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])

        plot_tools.set_axis(axis10, xlim=[-0.1, 1], ylim=[-0.1, 1], xlabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', ylabel=r'$\mathrm{D/T_{CR}}$',
                            aspect=None, which='major')
        # Plot disc to total ratio including counter-rotating particles as a function of original disc to total #
        hb = axis10.hexbin(glx_disc_fractions_IT20, glx_disc_fractions_IT20_cr, bins='log', gridsize=50, cmap='terrain_r')
        plot_tools.create_colorbar(axis00, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
        axis10.plot([0, 1], [0, 2], c='tab:green', label=r'$\mathrm{1:2}$')

        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(glx_disc_fractions_IT20, glx_disc_fractions_IT20_cr, bin_type='equal_width',
                                                                       n_bins=20, log=False)
        axis10.plot(x_value, median, color='black', linewidth=3, label=r'$\mathrm{Median}$')
        axis10.fill_between(x_value, shigh, slow, color='black', alpha=0.3, label=r'$\mathrm{16^{th}-84^{th}\;\%ile}$')
        plt.fill(np.NaN, np.NaN, color='black', alpha=0.3)

        # Create the legends, save and close the figure #
        axis10.legend(loc='lower right', frameon=False, fontsize=20)
        plt.savefig(plots_path + 'DTTCR_DTT' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalCRVsDiscToTotal(simulation_path, tag)
