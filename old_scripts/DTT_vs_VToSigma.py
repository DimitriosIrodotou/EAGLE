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


class DiscToTotalVsVToSigma:
    """
    For all galaxies create: a disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over
    dispersion plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        disc_fractions = np.load(data_path + 'glx_disc_fractions.npy')
        kappas_corotation = np.load(data_path + 'glx_kappas_corotation.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        concentration_indices = np.load(data_path + 'glx_concentration_indices.npy')
        rotationals_over_dispersions = np.load(data_path + 'glx_rotationals_over_dispersions.npy')

        # Normalise disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions, kappas_corotation, glx_disc_fractions_IT20, concentration_indices, rotationals_over_dispersions)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished DiscToTotalVsVToSigma for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions, kappas_corotation, glx_disc_fractions_IT20, concentration_indices, rotationals_over_dispersions):
        """
        Plot the disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over dispersion.
        :param disc_fractions: where the bulge is assumed to have zero net angualr momentum.
        :param kappas_corotation: defined as angular kinetic energy over kinetic energy.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param concentration_indices: defined as R90/R50
        :param rotationals_over_dispersions: defined as vrot/sigam
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        
        # axis10.set_xlim(0, 9)
        cmap = matplotlib.cm.get_cmap('nipy_spectral_r')
        # for axis in [axis10, axis11, axis12, axis13]:
        #     axis.set_ylim(0, 1)
        #     axis.set_facecolor(cmap(0))
        #     axis.grid(True, which='both', axis='both')
        #     axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=16)
        # for axis in [axis11, axis12, axis13]:
        #     axis.set_yticklabels([])
        
        plot_tools.set_axis(axis10, xlabel=r'$\mathrm{V_{rot}/\sigma}$', ylabel=r'$\mathrm{D/T_{30\degree}}$', aspect=None)
        glx_as = np.load(data_path + 'glx_as.npy')
        hb = axis10.scatter(rotationals_over_dispersions, glx_disc_fractions_IT20, c=glx_as, s=5, label=r'$D/T_{\vec{J}_{b} = 0}$', cmap=cmap)
        plot_tools.create_colorbar(axis00, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
        
        # Plot median and 1-sigma lines #
        # x_value, median, shigh, slow = plot_tools.binned_median_1sigma(x_attribute, glx_disc_fractions_IT20, 0.09, log=False)
        # axis.plot(x_value, median, color='silver', linewidth=3, zorder=5)
        # axis.fill_between(x_value, shigh, slow, color='silver', alpha=0.3, zorder=5)
        #
        # axis.axvline(x=threshold, c='tab:red')  # Plot threshold lines.
        #
        # axis.set_xlabel(label, size=16)
        # Save the figure #
        plt.savefig(plots_path + 'DTT_VTS' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsVToSigma(simulation_path, tag)
