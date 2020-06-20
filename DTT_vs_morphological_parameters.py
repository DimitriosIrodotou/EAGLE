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


class DiscToTotalVsMorphologicalParameters:
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
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        concentration_indices = np.load(data_path + 'glx_concentration_indices.npy')
        rotationals_over_dispersions = np.load(data_path + 'glx_rotationals_over_dispersions.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions, kappas_corotation, disc_fractions_IT20, concentration_indices, rotationals_over_dispersions)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished DTT_MP for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions, kappas_corotation, disc_fractions_IT20, concentration_indices, rotationals_over_dispersions):
        """
        Plot the disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over dispersion.
        :param disc_fractions: where the bulge is assumed to have zero net angualr momentum.
        :param kappas_corotation: defined as angular kinetic energy over kinetic energy.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param concentration_indices: defined as R90/R50
        :param rotationals_over_dispersions: defined as vrot/sigam
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis01 = figure.add_subplot(gs[0, 1])
        axis02 = figure.add_subplot(gs[0, 2])
        axis03 = figure.add_subplot(gs[0, 3])
        axis10 = figure.add_subplot(gs[1, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        axis12 = figure.add_subplot(gs[1, 2])
        axis13 = figure.add_subplot(gs[1, 3])
        
        axis10.set_xlim(0, 9)
        axis10.set_ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        cmap = matplotlib.cm.get_cmap('copper')
        for axis in [axis10, axis11, axis12, axis13]:
            axis.set_ylim(0, 1)
            axis.set_facecolor(cmap(0))
            axis.grid(True, which='both', axis='both')
            axis.tick_params(direction='out', which='both', top='on', right='on',  labelsize=16)
        for axis in [axis11, axis12, axis13]:
            axis.set_yticklabels([])
        
        axes = [axis10, axis11, axis12, axis13]
        axescbar = [axis00, axis01, axis02, axis03]
        thresholds = [2.5, 0.4, 0.5, 0.55]
        x_attributes = [concentration_indices, kappas_corotation, disc_fractions, rotationals_over_dispersions]
        labels = [r'$\mathrm{Concentration\;index}$', r'$\mathrm{\kappa_{co}}$', r'$\mathrm{D/T_{\vec{J}_{b}=0}}$', r'$\mathrm{V_{rot}/\sigma}$']
        for axis, axiscbar, x_attribute, label, threshold in zip(axes, axescbar, x_attributes, labels, thresholds):
            # Plot attributes #
            hb = axis.hexbin(x_attribute, disc_fractions_IT20, gridsize=100, label=r'$D/T_{\vec{J}_{b} = 0}$', cmap=cmap)
            plot_tools.create_colorbar(axiscbar, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
            
            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.median_1sigma(x_attribute, disc_fractions_IT20, 0.09, log=False)
            axis.plot(x_value, median, color='silver', linewidth=3, zorder=5)
            axis.fill_between(x_value, shigh, slow, color='silver', alpha='0.3', zorder=5)
            
            axis.axvline(x=threshold, c='tab:red')  # Plot threshold lines.
            
            axis.set_xlabel(label, size=16)
        # Save the figure #
        plt.savefig(plots_path + 'DTT_MP' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsMorphologicalParameters(simulation_path, tag)
