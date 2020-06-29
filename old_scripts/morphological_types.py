import os
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


class MorphologicalTypes:
    """
    For all galaxies create: a fractional breakdown into different morphological types plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift directory
        """
        start_local_time = time.time()  # Start the local time.
        
        stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions_IT20, stellar_masses)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MorphologicalTypes for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions_IT20, stellar_masses):
        """
        Plot the a fractional breakdown into different morphological types.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        
        plt.grid(True, which='both', axis='both')
        # plt.xscale('log')
        # ax10.set_yscale('log')
        # ax10.set_ylim(1e0, 1e5)
        # ax10.set_xlim(1e9, 1e12)
        # ax10.set_xlabel(r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', size=16)
        # ax10.set_ylabel(r'$\mathrm{(|\vec{J}_{\odot}|/M_{\odot})/(kpc\;km\;s^{-1})}$', size=16)
        # ax10.tick_params(direction='out', which='both', top='on', right='on',  labelsize=16)
        
        bulge_fractions_IT20 = 1 - disc_fractions_IT20
        stellar_masses = np.log10(stellar_masses) + np.log10(0.673)
        bulge_masses = bulge_fractions_IT20 * stellar_masses
        
        # Divisions between bulge classes #
        logRatio1 = -0.154902
        logRatio2 = -2
        
        # Bins for histogram and plotting #
        binwidth = 0.25
        xrange = np.array([8.67, 11.8])
        bins = np.arange(xrange[0], xrange[1] + 0.001, binwidth)
        
        # Put galaxies into bins #
        indBin = np.digitize(stellar_masses, bins)
        nBin = len(bins) - 1
        x = np.empty(nBin)
        yIrr = np.empty(nBin)
        yDisk = np.empty(nBin)
        yBulge = np.empty(nBin)
        
        # Loop over bins and count fractions in each class #
        for iBin in range(nBin):
            x[iBin] = 0.5 * (bins[iBin] + bins[iBin + 1])
            indThisBin = np.where(indBin == iBin + 1)[0]
            allBin = len(indThisBin)
            
            # Bulges
            yBulge[iBin] = len(np.where((bulge_masses[indThisBin] - stellar_masses[indThisBin]) > logRatio1)[0]) / float(allBin)
            # Disks
            yIrr[iBin] = len(np.where((bulge_masses[indThisBin] - stellar_masses[indThisBin]) < logRatio2)[0]) / float(allBin)
            # Intermediates
            yDisk[iBin] = 1. - yBulge[iBin] - yIrr[iBin]
        
        # Plot fractions #
        plt.plot(x, yIrr, color='blue', lw=2)
        plt.plot(x, yBulge, color='red', lw=2)
        plt.plot(x, yDisk, color='green', lw=2)
        
        # Read observational data from C06 and ﻿KDR14 #
        obsBulge = np.loadtxt('./Obs_Data/Conselice06_Bulge_Frac.txt')
        obsDisk = np.loadtxt('./Obs_Data/Conselice06_Disk_Frac.txt')
        obsIrr = np.loadtxt('./Obs_Data/Conselice06_Irr_Frac.txt')
        
        x = [9, 9.5, 10, 10.5, 11, 11.5]
        yE = [0.23, 0.25, 0.24, 0.31, 0.72, 1.00]
        yS0Scd = [0.46, 0.64, 0.74, 0.68, 0.28, 0.00]
        ySd = [0.31, 0.11, 0.02, 0.00, 0.00, 0.00]
        
        # Plot observational data from C06 and ﻿KDR14 #
        obsHubble = 0.7
        obsMass = obsBulge[:, 0] + 2 * np.log10(obsHubble)
        plt.errorbar(obsMass, obsBulge[:, 1], yerr=obsBulge[:, 2], marker='o', color='red', linestyle='None', elinewidth=1, capsize=3, capthick=1)
        
        obsMass = obsDisk[:, 0] + 2 * np.log10(obsHubble)
        plt.errorbar(obsMass, obsDisk[:, 1], yerr=obsDisk[:, 2], marker='o', color='green', linestyle='None', elinewidth=1, capsize=3, capthick=1)
        
        obsMass = obsIrr[:, 0] + 2 * np.log10(obsHubble)
        plt.errorbar(obsMass, obsIrr[:, 1], yerr=obsIrr[:, 2], marker='o', color='blue', linestyle='None', elinewidth=1, capsize=3, capthick=1)
        
        plt.errorbar(x, yE, xerr=0.1, color='red', marker='s', linestyle='None', elinewidth=1, capsize=3, capthick=1)
        plt.errorbar(x, yS0Scd, xerr=0.1, color='green', marker='s', linestyle='None', elinewidth=1, capsize=3, capthick=1)
        plt.errorbar(x, ySd, xerr=0.1, color='blue', marker='s', linestyle='None', elinewidth=1, capsize=3, capthick=1)
        
        # Save the plot #
        plt.savefig(plots_path + 'MT' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/MT/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = MorphologicalTypes(simulation_path, tag)
