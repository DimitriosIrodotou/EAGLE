import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import eagle_IO.eagle_IO.eagle_IO as E

date = time.strftime("%d_%m_%y_%H%M")


class PositionHistogram:

    def __init__(self, tag, sim):
        start_time = time.time()  # Start the time.

        # Load data #
        self.stellar_data = self.read_galaxy(tag, sim)
        print("--- Finished reading the data in %.5s seconds ---" % (time.time() - start_time))  # Print reading time.
        start_time = time.time()  # Start the time.

        # Plot #
        self.plot()
        print("--- Finished plotting the data in %.5s seconds ---" % (time.time() - start_time))  # Print plotting time.


    def read_galaxy(self, tag, sim):
        """
         For a given galaxy (defined by its GroupNumber and SubGroupNumber) extract the selected particle and subhalo properties.

        :param tag: redshift
        :param sim: simulation directory
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

        # Mask to selected GroupNumber and SubGroupNumber #
        cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', numThreads=4)
        cd = E.read_array('PARTDATA', sim, tag, '/PartType' + particle_type + '/Coordinates', numThreads=4)
        mask = np.where((stellar_data['SubGroupNumber'] == subhalo_data['SubGroupNumber'][0]) & (
                stellar_data['GroupNumber'] == subhalo_data['GroupNumber'][0]) & (np.sqrt(np.sum((cd - cop[0]) ** 2, axis=1)) <= 0.03))

        for att in stellar_data.keys():
            stellar_data[att] = stellar_data[att][mask]

        # Convert to astronomical units #
        stellar_data['Coordinates'] *= u.cm.to(u.Mpc)

        return stellar_data


    def plot(self):
        """

        :return: None
        """

        plt.close()
        figure = plt.figure(0, figsize=(10, 7.5))

        plt.xlabel(r'$\mathrm{x/Mpc}$')
        plt.ylabel(r'$\mathrm{y/Mpc}$')
        plt.tick_params(direction='in', which='both', top='on', right='on')

        # Plot.
        hexbin = plt.hexbin(list(zip(*self.stellar_data['Coordinates']))[0], list(zip(*self.stellar_data['Coordinates']))[1], bins='log',
                            xscale='log', yscale='log', cmap="Greys", gridsize=150, edgecolor='none')
        # Adjust the color bar #
        cbaxes = figure.add_axes([0.9, 0.11, 0.02, 0.77])
        cb = plt.colorbar(hexbin, cax=cbaxes)
        cb.set_label('$\mathrm{Counts\; per\; hexbin}$')

        # Save plot.
        plt.savefig('PositionHistogram' + '-' + date + '.png', bbox_inches='tight')

        return None


if __name__ == '__main__':
    tag = '005_z010p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    x = PositionHistogram(tag, sim)