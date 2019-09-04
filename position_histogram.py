import re
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

import eagle_IO.eagle_IO.eagle_IO as E

sns.set()
sns.set_style('ticks', {'axes.grid': True})
sns.set_context('notebook', font_scale=1.6)
date = time.strftime("%d_%m_%y_%H%M")
outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/'


class PositionHistogram:

    def __init__(self, sim, tag):
        start_global_time = time.time()  # Start the global time.

        start_local_time = time.time()  # Start the local time.
        # Load data #
        self.stellar_data = self.read_galaxy(sim, tag)
        print("--- Finished reading the data in %.5s seconds ---" % (time.time() - start_local_time))  # Print reading time.

        # Plot #
        self.plot()
        print("--- Finished PositionHistogram.py in %.5s seconds ---" % (time.time() - start_global_time))  # Print plotting time.


    def read_galaxy(self, sim, tag):
        """
         For a given galaxy (defined by its GroupNumber and SubGroupNumber) extract the selected particle and subhalo properties.

        :param sim: simulation directory
        :param tag: redshift folder
        :return: subhalo_data, stellar_data
        """

        # Load subhalo data #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for att in ['GroupNumber', 'SubGroupNumber', 'CentreOfPotential']:
            subhalo_data[att] = E.read_array(file_type, sim, tag, '/Subhalo/' + att, noH=True, physicalUnits=True, CGS=True, numThreads=4,
                                             verbose=False)

        # Load particle data #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for att in ['GroupNumber', 'SubGroupNumber', 'Coordinates']:
            stellar_data[att] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + att, noH=True, physicalUnits=True, CGS=True,
                                             numThreads=4, verbose=False)

        # Normalise coordinates wrt the centre of mass of the subhalo #
        stellar_data['Coordinates'] = np.subtract(stellar_data['Coordinates'], subhalo_data['CentreOfPotential'][0])

        # Mask to selected GroupNumber and SubGroupNumber #
        cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', numThreads=4)
        cd = E.read_array('PARTDATA', sim, tag, '/PartType' + particle_type + '/Coordinates', numThreads=4)
        mask = np.where((stellar_data['SubGroupNumber'] == subhalo_data['SubGroupNumber'][0]) & (
                stellar_data['GroupNumber'] == subhalo_data['GroupNumber'][0]) & (np.sqrt(np.sum((cd - cop[0]) ** 2, axis=1)) <= 0.03))

        for att in stellar_data.keys():
            stellar_data[att] = stellar_data[att][mask]

        # Convert to astronomical units #
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)

        return stellar_data


    def plot(self):
        """

        :return: None
        """
        # Generate the figures #
        plt.close()
        figure = plt.figure(0, figsize=(10, 10))

        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=(20, 1))
        gs.update(hspace=0.2)
        axtop = plt.subplot(gs[0, 0])
        axbot = plt.subplot(gs[1, 0])
        axcbar = plt.subplot(gs[:, 1])

        # Generate the XY projection #
        axtop.set_xlabel(r'$\mathrm{x/kpc}$')
        axtop.set_ylabel(r'$\mathrm{y/kpc}$')
        axtop.tick_params(direction='in', which='both', top='on', right='on')

        pltop = axtop.hexbin(list(zip(*self.stellar_data['Coordinates']))[0], list(zip(*self.stellar_data['Coordinates']))[1], bins='log',
                             cmap="Greys", gridsize=150, edgecolor='none')

        # Generate the XZ projection #
        axbot.set_xlabel(r'$\mathrm{x/kpc}$')
        axbot.set_ylabel(r'$\mathrm{z/kpc}$')

        plbot = axbot.hexbin(list(zip(*self.stellar_data['Coordinates']))[0], list(zip(*self.stellar_data['Coordinates']))[2], bins='log',
                             cmap="Greys", gridsize=150, edgecolor='none')

        # Generate the color bar #
        cbar = plt.colorbar(pltop, cax=axcbar)
        cbar.set_label('$\mathrm{log_{10}(Particles\; per\; hexbin)}$')

        # Save plot #
        plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + 'PH' + '-' + date + '.png', bbox_inches='tight')

        return None


if __name__ == '__main__':
    tag = '010_z005p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    x = PositionHistogram(sim, tag)