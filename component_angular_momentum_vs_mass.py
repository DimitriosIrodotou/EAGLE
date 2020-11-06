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


class ComponentAngularMomentumVsMass:
    """
    For all components create: an angular momentum as a function of its stellar mass plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        disc_stellar_angular_momenta = np.load(data_path + 'disc_stellar_angular_momenta.npy')
        spheroid_stellar_angular_momenta = np.load(data_path + 'spheroid_stellar_angular_momenta.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_stellar_masses, glx_disc_fractions_IT20, disc_stellar_angular_momenta, spheroid_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished ComponentAngularMomentumVsMass for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_stellar_masses, glx_disc_fractions_IT20, disc_stellar_angular_momenta, spheroid_stellar_angular_momenta):
        """
        Plot the component angular momentum as a function of its stellar mass.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :param disc_stellar_angular_momenta: defined as the sum of each disc particle's angular momentum.
        :param spheroid_stellar_angular_momenta: defined as the sum of each spheroid particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[1e8, 1e12], ylim=[1e-1, 1e6], xscale='log', yscale='log',
                            xlabel=r'$\mathrm{log_{10}(M_{\bigstar,comp}/M_{\odot})}$',
                            ylabel=r'$\mathrm{(|\vec{J}_{\bigstar,comp}|/M_{\bigstar, comp})/(kpc\;km\;s^{-1})}$', aspect=None, which='major')

        # Calculate component specific angular momentum #
        spc_disc_angular_momenta = np.divide(np.linalg.norm(disc_stellar_angular_momenta, axis=1), glx_disc_fractions_IT20 * glx_stellar_masses)
        spc_spheroid_angular_momenta = np.divide(np.linalg.norm(spheroid_stellar_angular_momenta, axis=1),
                                                 (1 - glx_disc_fractions_IT20) * glx_stellar_masses)

        # Plot galactic angular momentum as a function of stellar mass colour-coded by disc to total ratio #
        d = plt.scatter(glx_disc_fractions_IT20 * glx_stellar_masses, spc_disc_angular_momenta, c='tab:blue', s=20, edgecolor='none')
        b = plt.scatter((1 - glx_disc_fractions_IT20) * glx_stellar_masses, spc_spheroid_angular_momenta, c='tab:red', s=20, edgecolor='none')

        # Read observational data from FR13 and TMS19 #
        FR13_D = np.genfromtxt('./observational_data/FR_1305.1626/Figure2_D.csv', delimiter=',', names=['Md', 'jd'])
        FR13_E = np.genfromtxt('./observational_data/FR_1305.1626/Figure2_E.csv', delimiter=',', names=['Mb', 'jb'])
        TMS19 = np.genfromtxt('./observational_data/TMS_1902.03792/Figure4_middleright_bulge.csv', delimiter=',', names=['Mb', 'jb'])

        # Plot observational data from FR13 and OG14 #
        plt.scatter(np.power(10, FR13_D['Md']), np.power(10, FR13_D['jd']), edgecolor='black', color='grey', s=150, marker='*',
                    label=r'$\mathrm{FR13:Discs}$', zorder=4)
        plt.scatter(np.power(10, FR13_E['Mb']), np.power(10, FR13_E['jb']), edgecolor='black', color='grey', s=50, marker='s',
                    label=r'$\mathrm{FR13:Bulges}$', zorder=4)
        plt.scatter(np.power(10, TMS19['Mb']), np.power(10, TMS19['jb']), edgecolor='black', color='grey', s=50, marker='^',
                    label=r'$\mathrm{Tabor\!+\!19:Bulges}$', zorder=4)

        # Create the legends, save and close the figure #
        legend = plt.legend([d, b], [r'$\mathrm{Discs}$', r'$\mathrm{Spheroids}$'], loc='upper left', fontsize=20, frameon=False, numpoints=1,
                            scatterpoints=1)
        plt.gca().add_artist(legend)
        plt.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1, scatterpoints=1)
        plt.savefig(plots_path + 'C_AM_M' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentAngularMomentumVsMass(simulation_path, tag)
