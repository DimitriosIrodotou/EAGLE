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

style.use("classic")
plt.rcParams.update({'font.family':'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class ComponentBetaVsDTT:
    """
    For all components create: a component delta as a function of disc to total ratio colour-coded by galaxy's delta plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        spheroid_deltas = np.load(data_path + 'spheroid_betas.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(spheroid_deltas, glx_disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished ComponentBetaVsDTT for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(spheroid_deltas, glx_disc_fractions_IT20):
        """
        Plot the component delta as a function of disc to total ratio colour-coded by galaxy's delta.
        :param spheroid_deltas: defined as the anisotropy parameter for the spheroid component.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 1], ylim=[-0.1, 1.1], xlabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', ylabel=r'$\mathrm{\tau}$',
                            aspect=None)

        # Plot the component delta as a function of disc to total ratio colour-coded by galaxy's delta #
        axis.scatter(glx_disc_fractions_IT20, np.exp(spheroid_deltas - 1), c='tab:red', s=20, label=r'$\mathrm{Spheroids}$', edgecolor='none')

        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(glx_disc_fractions_IT20, np.exp(spheroid_deltas - 1), bin_type='equal_width',
                                                                       n_bins=25, log=False)
        axis.plot(x_value, median, color='black', linewidth=3, label=r'$\mathrm{Median}$')
        axis.fill_between(x_value, shigh, slow, color='black', alpha=0.3)
        plt.fill(np.NaN, np.NaN, color='black', alpha=0.3, label=r'$\mathrm{16^{th}-84^{th}\;\%ile}$')

        # Plot horizontal lines for different delta values #
        axis.axhline(y=np.exp(0 - 1), c='black', lw=3, linestyle='dashed', label=r'$\mathrm{Isotropic}$')

        # Create the legends, save and close the figure #
        plt.legend(loc='upper center', fontsize=20, frameon=False, numpoints=1, scatterpoints=1, ncol=2)
        plt.savefig(plots_path + 'C_B_DTT' + '-' + date + '.pdf', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentBetaVsDTT(simulation_path, tag)
