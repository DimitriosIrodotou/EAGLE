import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DiscToTotalVsBaryons:
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
        
        stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        gaseous_masses = np.load(data_path + 'glx_gaseous_masses.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(stellar_masses, gaseous_masses, disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print(
            'Finished DTT_B for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, stellar_masses, gaseous_masses, disc_fractions_IT20):
        """
        Plot the disc to total ratio as a function of mass, angular momentum.
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param gaseous_masses: defined as the mass of all gaseous particles within 30kpc from the most bound particle.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        # gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        # axcbar = figure.add_subplot(gs[0, 0])
        # ax10 = figure.add_subplot(gs[1, 0])
        
        plt.ylim(0, 1)
        plt.xscale('log')
        # plt.xlim(1e2, 1e6)
        plt.grid(True, which='major', axis='both')
        plt.ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        plt.xlabel(r'$\mathrm{M_{\bigstar}}$', size=16)
        plt.tick_params(direction='out', which='both', top='on', right='on', left='on', labelsize=16)
        
        # cmap = matplotlib.cm.get_cmap('copper')
        colors = iter(matplotlib.cm.rainbow(np.linspace(0, 1, 10)))
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # s_m = matplotlib.cm.ScalarMappable(cmap='copper', norm=norm)
        for i in np.arange(0, 1, 0.1):
            mask, = np.where((disc_fractions_IT20 < i) & disc_fractions_IT20 > i - 0.1)
            x_value, median, shigh, slow = plot_tools.median_1sigma(stellar_masses[mask], disc_fractions_IT20[mask], 0.17, log=True)
            pl = plt.plot(x_value, median, color=next(colors), linewidth=5,
                          zorder=5)  # plt.fill_between(x_value, shigh, slow, color=s_m.to_rgba(i), alpha='0.5', zorder=5)  #   #
            # plot_tools.create_colorbar(axcbar, pl, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
        
        # Save the figure. #
        plt.savefig(plots_path + 'DTT_B' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsBaryons(simulation_path, tag)
