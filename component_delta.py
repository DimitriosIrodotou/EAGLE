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


class ComponentDelta:
    """
    For all galaxies create: a disc and bulge delta-DTT relation colour-coded by galaxy's delta plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        glx_deltas = np.load(data_path + 'glx_deltas.npy')
        disc_deltas = np.load(data_path + 'disc_deltas.npy')
        bulge_deltas = np.load(data_path + 'bulge_deltas.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(glx_deltas, disc_deltas, bulge_deltas, disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished ComponentDelta for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
                time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, glx_deltas, disc_deltas, bulge_deltas, disc_fractions_IT20):
        """
        Plot the disc and bulge delta-DTT relation colour-coded by galaxy's delta.
        :param glx_deltas: defined as the anisotropy parameter for the whole galaxy.
        :param disc_deltas: defined as the anisotropy parameter for the disc component.
        :param bulge_deltas: defined as the anisotropy parameter for the bulge component.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, :])
        axis10 = figure.add_subplot(gs[1, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        
        for axis in [axis10, axis11]:
            axis.grid(True, which='both', axis='both')
            axis.set_ylim(-0.1, 1.1)
            axis.set_xlabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
            axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=16)
        axis11.yaxis.tick_right()
        axis11.yaxis.set_label_position("right")
        axis10.set_ylabel(r'$\mathrm{exp(\delta_{bulge}-1)}$', size=16)
        axis11.set_ylabel(r'$\mathrm{exp(\delta_{disc}-1)}$', size=16)
        
        axis10.scatter(disc_fractions_IT20, np.exp(bulge_deltas - 1), c=np.exp(glx_deltas - 1), s=8, vmin=0, vmax=1, cmap='magma')
        sc = axis11.scatter(disc_fractions_IT20, np.exp(disc_deltas - 1), c=np.exp(glx_deltas - 1), s=8, vmin=0, vmax=1, cmap='magma')
        plot_tools.create_colorbar(axis00, sc, r'$\mathrm{exp(\delta_{gal}-1)}$', 'horizontal')
        
        # Plot horizontal lines for different delta values #
        for axis in [axis10, axis11]:
            axis.axhline(y=np.exp(0 - 1), c='black', lw=3, linestyle='dashed', label=r'$\mathrm{\delta=0}$')
            axis.axhline(y=np.exp(1 - 1), c='tab:red', lw=3, linestyle='dashed', label=r'$\mathrm{\delta=1}$')
            axis.axhline(y=np.exp(-np.inf - 1), c='tab:blue', lw=3, linestyle='dashed', label=r'$\mathrm{\delta=-inf}$')
        
        # Save the figure #
        axis10.legend(loc='upper center',ncol=3, fontsize=16, frameon=False)
        axis11.legend(loc='upper center',ncol=3, fontsize=16, frameon=False)
        plt.savefig(plots_path + 'CD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentDelta(simulation_path, tag)
