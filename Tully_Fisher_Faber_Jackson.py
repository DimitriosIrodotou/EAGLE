import os
import re
import time
import warnings
import plot_tools
import matplotlib.cbook

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib import gridspec

style.use("classic")
plt.rcParams.update({'font.family':'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class TullyFisherFaberJackson:
    """
    For all galaxies create: a Tully-Fisher and Faber-Jackson relation plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        glx_sigma_0s = np.load(data_path + 'glx_sigma_0s_re.npy')
        disc_sigma_0s = np.load(data_path + 'disc_sigma_0s_re.npy')
        spheroid_sigma_0s = np.load(data_path + 'spheroid_sigma_0s_re.npy')
        glx_rotationals = np.load(data_path + 'glx_rotationals.npy')
        disc_rotationals = np.load(data_path + 'disc_rotationals.npy')
        spheroid_rotationals = np.load(data_path + 'spheroid_rotationals.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_stellar_masses, glx_rotationals, disc_rotationals, spheroid_rotationals, glx_sigma_0s, disc_sigma_0s, spheroid_sigma_0s,
                  glx_disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished TullyFisherFaberJackson for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_stellar_masses, glx_rotationals, disc_rotationals, spheroid_rotationals, glx_sigma_0s, disc_sigma_0s, spheroid_sigma_0s,
             glx_disc_fractions_IT20):
        """
        Plot the Tully-Fisher and Faber-Jackson relation.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_rotationals: defined as the rotational velocity for the whole galaxy.
        :param disc_rotationals: defined as the rotational velocity for the disc component.
        :param spheroid_rotationals: defined as the rotational velocity for the spheroid component.
        :param glx_sigma_0s: defined as the disc-plane velocity dispersion for the whole galaxy.
        :param disc_sigma_0s: defined as the disc-plane velocity dispersion for the disc component.
        :param spheroid_sigma_0s: defined as the disc-plane velocity dispersion for the spheroid component.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(3, 2, wspace=0.05, hspace=0.05, height_ratios=[0.05, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10, axis11 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])
        axis20, axis21 = figure.add_subplot(gs[2, 0]), figure.add_subplot(gs[2, 1])

        plot_tools.set_axis(axis10, xlim=[0.5, 3.4], ylim=[9.6, 12.5], ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', aspect=None)
        plot_tools.set_axis(axis11, xlim=[0.5, 3.4], ylim=[9.6, 12.5], aspect=None)
        plot_tools.set_axis(axis20, xlim=[0.5, 3.4], ylim=[9.6, 12.5], xlabel=r'$\mathrm{log_{10}(V_{rot}/(km\;s^{-1}))}$',
                            ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', aspect=None)
        plot_tools.set_axis(axis21, xlim=[0.5, 3.4], ylim=[9.6, 12.5], xlabel=r'$\mathrm{log_{10}(\sigma_{0,e}/(km\;s^{-1}))}$', aspect=None)
        for axis in [axis11, axis21]:
            axis.set_yticklabels([])
        for axis in [axis10, axis11]:
            axis.set_xticklabels([])

        # Plot the Tully-Fisher relations #
        sc = axis10.scatter(np.log10(glx_rotationals), np.log10(glx_stellar_masses), c=glx_disc_fractions_IT20, s=20, cmap='seismic_r', vmin=0,
                            vmax=1, edgecolor='none')
        axis20.scatter(np.log10(disc_rotationals), np.log10(glx_stellar_masses), c='tab:blue', s=20, label=r'$\mathrm{Discs}$', edgecolor='none')
        axis20.scatter(np.log10(spheroid_rotationals), np.log10(glx_stellar_masses), c='tab:red', s=20, label=r'$\mathrm{Spheroids}$',
                       edgecolor='none')
        plot_tools.create_colorbar(axiscbar, sc, r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', 'horizontal')

        # Plot the Faber-Jackson relations #
        axis11.scatter(np.log10(glx_sigma_0s), np.log10(glx_stellar_masses), c=glx_disc_fractions_IT20, s=20, cmap='seismic_r', vmin=0, vmax=1,
                       edgecolor='none')
        axis21.scatter(np.log10(disc_sigma_0s), np.log10(glx_stellar_masses), c='tab:blue', s=20, label=r'$\mathrm{Discs}$', edgecolor='none')
        axis21.scatter(np.log10(spheroid_sigma_0s), np.log10(glx_stellar_masses), c='tab:red', s=20, label=r'$\mathrm{Spheroids}$', edgecolor='none')

        # Read and plot observational data from AZF08, TEA11 and OCB20 #
        AZF08 = np.genfromtxt('./observational_data/AZF_0807.0636/Figure1.csv', delimiter=',', names=['Vrot', 'Mstar'])
        OCB20_TF_DD = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_TF_DD.csv', delimiter=',', names=['Vrot', 'Mstar'])
        OCB20_TF_discs = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_TF_discs.csv', delimiter=',', names=['Vrot', 'Mstar'])
        OCB20_TF_bulges = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_TF_bulges.csv', delimiter=',', names=['Vrot', 'Mstar'])
        OCB20_FJ_BD = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_FJ_BD.csv', delimiter=',', names=['sigma', 'Mstar'])
        OCB20_FJ_discs = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_FJ_discs.csv', delimiter=',', names=['sigma', 'Mstar'])
        OCB20_FJ_bulges = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_FJ_bulges.csv', delimiter=',', names=['sigma', 'Mstar'])

        axis10.scatter(AZF08['Mstar'], AZF08['Vrot'], edgecolor='black', color='grey', marker='s', s=20, label=r'$\mathrm{Avila-Reese\!+\!08}$')
        axis10.plot(OCB20_TF_DD['Vrot'], OCB20_TF_DD['Mstar'], color='cyan', label=r'$\mathrm{Oh\!+\!20:B/T<0.2}$')
        axis11.plot(OCB20_FJ_BD['sigma'], OCB20_FJ_BD['Mstar'], color='orange', label=r'$\mathrm{Oh\!+\!20:B/T>0.8}$')
        axis20.plot(OCB20_TF_discs['Vrot'], OCB20_TF_discs['Mstar'], color='cyan', label=r'$\mathrm{Oh\!+\!20:discs}$')
        axis20.plot(OCB20_TF_bulges['Vrot'], OCB20_TF_bulges['Mstar'], color='orange', label=r'$\mathrm{Oh\!+\!20:bulges}$')
        axis21.plot(OCB20_FJ_discs['sigma'], OCB20_FJ_discs['Mstar'], color='cyan', label=r'$\mathrm{Oh\!+\!20:discs}$')
        axis21.plot(OCB20_FJ_bulges['sigma'], OCB20_FJ_bulges['Mstar'], color='orange', label=r'$\mathrm{Oh\!+\!20:bulges}$')

        # Create the legend and save the figure #
        axis10.legend(loc='upper left', fontsize=14, frameon=False, numpoints=1, scatterpoints=1)
        for axis in [axis11, axis20, axis21]:
            axis.legend(loc='upper left', fontsize=14, frameon=False, numpoints=1, scatterpoints=1, ncol=2)
        plt.savefig(plots_path + 'TFFJ' + '-' + date + '.pdf', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = TullyFisherFaberJackson(simulation_path, tag)
