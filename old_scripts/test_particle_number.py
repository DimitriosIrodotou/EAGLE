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


class TestParticleNumber:
    """
    For all galaxies create: a disc to total ratio as a function of mass, angular momentum.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        particle_numbers = np.load(data_path + 'glx_particle_numbers.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

        # Normalise disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(particle_numbers, glx_disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished TestParticleNumber for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, particle_numbers, glx_disc_fractions_IT20):
        """
        Plot the disc to total ratio as a function of mass, angular momentum.
        :param particle_numbers: defined as the number of all stellar particles within 30kpc from the most bound particle.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        
        plt.ylim(0, 1)
        plt.xscale('log')
        plt.xlim(1e2, 1e6)
        cmap = matplotlib.cm.get_cmap('copper')
        axis10.set_facecolor(cmap(0))
        plt.grid(True, which='major', axis='both')
        plt.ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        plt.xlabel(r'$\mathrm{N_{prc, \bigstar}}$', size=16)
        plt.tick_params(direction='out', which='both', top='on', right='on',  labelsize=16)
        
        hb = plt.hexbin(particle_numbers, glx_disc_fractions_IT20, xscale='log', bins='log', gridsize=100, label=r'$D/T_{\vec{J}_{b} = 0}$', cmap=cmap)
        plot_tools.create_colorbar(axis00, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
        
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(particle_numbers, glx_disc_fractions_IT20, 0.17, log=True)
        plt.plot(x_value, median, color='silver', linewidth=5, zorder=5)
        plt.fill_between(x_value, shigh, slow, color='silver', alpha=0.5, zorder=5)
        
        # Save the figure #
        plt.savefig(plots_path + 'TPN' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = TestParticleNumber(simulation_path, tag)
