import os
import re
import time
import warnings
import matplotlib
import plot_tools
import matplotlib.cbook

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec

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
        start_local_time = time.time()  # Start the local time.
        
        glx_sigma_0s = np.load(data_path + 'glx_sigma_0s.npy')
        disc_sigma_0s = np.load(data_path + 'disc_sigma_0s.npy')
        bulge_sigma_0s = np.load(data_path + 'bulge_sigma_0s.npy')
        glx_rotationals = np.load(data_path + 'glx_sigma_0s.npy')
        disc_rotationals = np.load(data_path + 'disc_rotationals.npy')
        bulge_rotationals = np.load(data_path + 'bulge_rotationals.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_deltas = np.load(data_path + 'glx_deltas.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(glx_stellar_masses, glx_rotationals, disc_rotationals, bulge_rotationals, glx_sigma_0s, disc_sigma_0s, bulge_sigma_0s,
                  disc_fractions_IT20, glx_deltas)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(glx_stellar_masses, glx_rotationals, disc_rotationals, bulge_rotationals, glx_sigma_0s, disc_sigma_0s, bulge_sigma_0s,
             disc_fractions_IT20, glx_deltas):
        """
        Plot the Tully-Fisher and Faber-Jackson relation.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_rotationals: defined as the rotational velocity for the whole galaxy.
        :param disc_rotationals: defined as the rotational velocity for the disc component.
        :param bulge_rotationals: defined as the rotational velocity for the bulge component.
        :param glx_sigma_0s: defined as the disc-plane velocity dispersion for the whole galaxy.
        :param disc_sigma_0s: defined as the disc-plane velocity dispersion for the disc component.
        :param bulge_sigma_0s: defined as the disc-plane velocity dispersion for the bulge component.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(3, 2, wspace=0.05, hspace=0.05, height_ratios=[0.05, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10 = figure.add_subplot(gs[1, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        axis20 = figure.add_subplot(gs[2, 0])
        axis21 = figure.add_subplot(gs[2, 1])
        plot_tools.set_axis(axis10, xlim=[0.5, 2.9], ylim=[9.5, 11.9], ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$')
        plot_tools.set_axis(axis11, xlim=[0.5, 2.9], ylim=[9.5, 11.9])
        plot_tools.set_axis(axis20, xlim=[0.5, 2.9], ylim=[9.5, 11.9], xlabel=r'$\mathrm{log_{10}(V_{rot}/(km\;s^{-1}))}$',
                            ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$')
        plot_tools.set_axis(axis21, xlim=[0.5, 2.9], ylim=[9.5, 11.9], xlabel=r'$\mathrm{log_{10}(\sigma_{0}/(km\;s^{-1}))}$')
        
        # Plot the Tully-Fisher relations #
        sc = axis10.scatter(np.log10(glx_rotationals), np.log10(glx_stellar_masses), c=disc_fractions_IT20, s=5, cmap='seismic_r')
        axis20.scatter(np.log10(disc_rotationals), np.log10(glx_stellar_masses), c='tab:blue', s=10, alpha=0.5, label=r'$\mathrm{Disc}$')
        axis20.scatter(np.log10(bulge_rotationals), np.log10(glx_stellar_masses), c='tab:red', s=10, alpha=0.5, label=r'$\mathrm{Bulge}$')
        plot_tools.create_colorbar(axiscbar, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        # Plot the Faber-Jackson relations #
        axis11.scatter(np.log10(glx_sigma_0s), np.log10(glx_stellar_masses), c=disc_fractions_IT20, s=5, cmap='seismic_r')
        axis21.scatter(np.log10(disc_sigma_0s), np.log10(glx_stellar_masses), c='tab:blue', s=10, alpha=0.5)
        axis21.scatter(np.log10(bulge_sigma_0s), np.log10(glx_stellar_masses), c='tab:red', s=10, alpha=0.5)
        
        # Read and plot observational data from AZF08, TEA11 and OCB20 #
        AZF08 = np.genfromtxt('./Obs_Data/AZF08.csv', delimiter=',', names=['x', 'y'])
        TEA11 = np.genfromtxt('./Obs_Data/TEA11.csv', delimiter=',', names=['x', 'y'])
        OCB20_TF_DD = np.genfromtxt('./Obs_Data/OCB20_TF_DD.csv', delimiter=',', names=['x', 'y'])
        OCB20_TF_discs = np.genfromtxt('./Obs_Data/OCB20_TF_discs.csv', delimiter=',', names=['x', 'y'])
        OCB20_TF_bulges = np.genfromtxt('./Obs_Data/OCB20_TF_bulges.csv', delimiter=',', names=['x', 'y'])
        OCB20_FJ_BD = np.genfromtxt('./Obs_Data/OCB20_FJ_BD.csv', delimiter=',', names=['x', 'y'])
        OCB20_FJ_discs = np.genfromtxt('./Obs_Data/OCB20_FJ_discs.csv', delimiter=',', names=['x', 'y'])
        OCB20_FJ_bulges = np.genfromtxt('./Obs_Data/OCB20_FJ_bulges.csv', delimiter=',', names=['x', 'y'])
        
        axis10.scatter(AZF08['y'], AZF08['x'], color='lime', s=15, zorder=2, label=r'$\mathrm{Avila-Reese+08}$')
        axis10.scatter(TEA11['x'], TEA11['y'], color='cyan', s=15, marker='s', zorder=2, label=r'$\mathrm{Torres-Flores+11}$')
        axis10.plot(OCB20_TF_DD['x'], OCB20_TF_DD['y'], color='cyan', zorder=2, label=r'$\mathrm{Oh+20\;B/T<0.2}$')
        axis20.plot(OCB20_TF_discs['x'], OCB20_TF_discs['y'], color='cyan', zorder=2, label=r'$\mathrm{Oh+20\;discs}$')
        axis20.plot(OCB20_TF_bulges['x'], OCB20_TF_bulges['y'], color='orange', zorder=2, label=r'$\mathrm{Oh+20\;bulges}$')
        axis11.plot(OCB20_FJ_BD['x'], OCB20_FJ_BD['y'], color='orange', zorder=2, label=r'$\mathrm{Oh+20\;B/T>0.8}$')
        axis21.plot(OCB20_FJ_discs['x'], OCB20_FJ_discs['y'], color='cyan', zorder=2)
        axis21.plot(OCB20_FJ_bulges['x'], OCB20_FJ_bulges['y'], color='orange', zorder=2)
        
        # Create the legend and save the figure #
        axis10.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
        axis20.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
        plt.savefig(plots_path + 'TFFJ' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = TullyFisherFaberJackson(simulation_path, tag)
