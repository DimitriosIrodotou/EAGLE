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
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class BlackholeVsSpheroidMass:
    """
    For all galaxies create: a black hole mass as a function of spheroid mass plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        bh_masses = np.load(data_path + 'bh_masses.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(bh_masses, glx_stellar_masses, glx_disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished BlackholeVsSpheroidMass for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(bh_masses, glx_stellar_masses, glx_disc_fractions_IT20):
        """
        Plot the black hole mass as a function of spheroid mass.
        :param bh_masses: defined as the sum of all black holes' mass.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[9, 12], ylim=[5, 11], xlabel=r'$\mathrm{log_{10}(M_{spheroid}/M_{\odot})}$',
                            ylabel=r'$\mathrm{log_{10}(M_{\bullet}/M_{\odot})}$', aspect=None, which='major')

        # Plot the black hole mass as a function of spheroid mass #
        plt.scatter(np.log10((1 - glx_disc_fractions_IT20) * glx_stellar_masses), np.log10(bh_masses), c='tab:red', s=20, edgecolor='none',
                    label=r'$\mathrm{Spheroids}$')

        # Read and  observational data from HR04 #
        HR04 = np.genfromtxt('./observational_data/HR04.csv', delimiter=',', names=['Mb', 'Mbh', 'yplus', 'yminus'])
        yerr = [np.log10(HR04['Mbh'] / HR04['yminus']), np.log10(HR04['yplus'] / HR04['Mbh'])]
        plt.errorbar(np.log10(HR04['Mb'] * 1e10), np.log10(HR04['Mbh'] * 1e10), yerr=yerr, fmt='s', ecolor='black', color='gray', markersize=5,
                     label=r'$\mathrm{HR04}$')

        # Create the legends, save and close the figure #
        plt.legend(loc='upper center', fontsize=20, frameon=False, numpoints=1, scatterpoints=1, ncol=2)
        plt.savefig(plots_path + 'B_B_M' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = BlackholeVsSpheroidMass(simulation_path, tag)
