import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class TestDTT:
    """
    For all galaxies create: a glx_disc_fractions_IT20 versus glx_disc_fractions_IT20_primeplot.
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

        # Normalise disc fractions #
        epsilon = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20_prime = np.divide(1, 1 - epsilon) * (glx_disc_fractions_IT20 - epsilon)

        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_disc_fractions_IT20, glx_disc_fractions_IT20_prime)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished TestDTT for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_disc_fractions_IT20, glx_disc_fractions_IT20_prime):
        """
        Plot glx_disc_fractions_IT20 versus glx_disc_fractions_IT20_prime.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        :param glx_disc_fractions_IT20_prime: normalised glx_disc_fractions_IT20
        :return: None
        """
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[-0.2, 1], ylim=[-0.2, 1], xlabel=r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', ylabel=r'$\mathrm{\acute{D}/T}$',
                            aspect=None)

        # Plot glx_disc_fractions_IT20 versus glx_disc_fractions_IT20_prime #
        plt.scatter(glx_disc_fractions_IT20, glx_disc_fractions_IT20_prime)

        # Create the legend, save and close the figure #
        # axis10.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        plt.savefig(plots_path + 'TDTT' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = TestDTT(simulation_path, tag)
