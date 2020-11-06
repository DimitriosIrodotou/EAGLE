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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

style.use("classic")
plt.rcParams.update({'font.family':'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class AngularMomentumVsMass:
    """
    For all galaxies create: a galactic angular momentum as a function of stellar mass colour-coded by disc to total ratio plot.
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
        glx_stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_stellar_masses, glx_disc_fractions_IT20, glx_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished AngularMomentumVsMass for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_stellar_masses, glx_disc_fractions_IT20, glx_stellar_angular_momenta):
        """
        Plot galactic angular momentum as a function of stellar mass colour-coded by disc to total ratio.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :param glx_stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.05, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        plot_tools.set_axis(axis10, xlim=[5e9, 1e12], ylim=[1e0, 1e5], xscale='log', yscale='log',
                            xlabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$',
                            ylabel=r'$\mathrm{(|\vec{J}_{\bigstar}|/M_{\bigstar})/(kpc\;km\;s^{-1})}$', aspect=None, which='major')

        # Plot the specific galactic angular momentum as a function of stellar mass colour-coded by disc to total ratio #
        spc_stellar_angular_momenta = np.linalg.norm(glx_stellar_angular_momenta, axis=1) / glx_stellar_masses
        sc = axis10.scatter(glx_stellar_masses, spc_stellar_angular_momenta, c=glx_disc_fractions_IT20, s=20, cmap='seismic_r', vmin=0, vmax=1,
                            edgecolor='none')
        plot_tools.create_colorbar(axis00, sc, r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', 'horizontal')

        # Read observational data from OG13, FR18 and MPPF20 #
        OG13 = np.genfromtxt('./observational_data/OG_1312.4543/Figure7_stars.csv', delimiter=',', names=['Mstar', 'jstar'])
        FR18_table1 = np.genfromtxt('./observational_data/FR_1808.02525/Table1.txt', delimiter='\t', names=['Mstar', 'jstar', 'BTT'])
        MPPF20 = np.genfromtxt('./observational_data/MPPF_2009.06645/Figure5_MJ_stars.txt', names=['Mstar', 'jstar'])

        # Plot observational data from OG13, FR18 and MPPF20 #
        axis10.scatter(MPPF20['Mstar'], MPPF20['jstar'], edgecolor='black', color='grey', s=75, marker='P',
                       label=r'$\mathrm{Mancera\,Pi\tilde{n}a\!+\!20}$', zorder=5)
        axis10.scatter(np.power(10, OG13['Mstar']), np.power(10, OG13['jstar']), edgecolor='black', color='grey', s=75, marker='s',
                       label=r'$\mathrm{OG14}$', zorder=5)
        sc2 = axis10.scatter(np.power(10, FR18_table1['Mstar']), np.power(10, FR18_table1['jstar']), edgecolor='black', cmap='seismic_r',
                             c=1 - FR18_table1['BTT'], marker='*', s=200, vmin=0, vmax=1)
        axiscbar = inset_axes(axis10, width='30%', height='3%', loc='upper center')
        plot_tools.create_colorbar(axiscbar, sc2, r'$\mathrm{FR18:D/T}$', 'horizontal', top=False, ticks=[0, 0.5, 1], size=20)

        # Create the legend, save and close the figure #
        axis10.legend(loc='lower right', fontsize=20, frameon=False, numpoints=1, scatterpoints=1)
        plt.savefig(plots_path + 'AM_M' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = AngularMomentumVsMass(simulation_path, tag)
