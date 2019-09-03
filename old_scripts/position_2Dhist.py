import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import read_dataset
from read_header import read_header

date = time.strftime("%d\%m\%y\%H%M")


class PositionHistogram:

    def __init__(self, gn, sgn):

        start_time = time.time()  # Start the time.

        # Load data #
        self.a, self.h, self.boxsize = read_header()
        self.stars = self.read_galaxy(4, gn, sgn)

        print("--- Finished reading the data in %.5s seconds ---" % (time.time() - start_time))  # Print reading time.
        start_time = time.time()  # Start the time.

        # Plot #
        # self.plot()
        print("--- Finished plotting the data in %.5s seconds ---" % (time.time() - start_time))  # Print plotting time.


    def read_galaxy(self, itype, gn, sgn):
        """
         For a given galaxy (defined by its GroupNumber and SubGroupNumber) extract the selected particle and subahlo properties.

        :param itype: particle type
        :param gn: selected GroupNumber
        :param sgn: selected SubGroupNumber
        :return: particle_data
        """

        # Empty arrays to hold the data #
        particle_data = {}
        subhalo_data = {}

        # Load subhalo data #
        for att in ['GroupNumber', 'SubGroupNumber', 'CentreOfPotential']:
            subhalo_data[att] = read_dataset.read_subhaloes(att, 127)
        subhalo_data['GroupNumber'] = subhalo_data['GroupNumber'].astype('int32')
        subhalo_data['SubGroupNumber'] = subhalo_data['SubGroupNumber'].astype('int32')

        # Load particle data #
        for att in ['GroupNumber', 'SubGroupNumber', 'Coordinates']:
            particle_data[att] = read_dataset.read_particles(itype, att, 128)

        # Mask to selected GroupNumber and SubGroupNumber #
        mask = np.where(particle_data['GroupNumber'] == subhalo_data['GroupNumber'][gn],
                        particle_data['SubGroupNumber'] == subhalo_data['SubGroupNumber'][sgn])

        print(mask)

        for att in particle_data.keys():
            particle_data[att] = particle_data[att][mask]

        # Convert to astronomical units #
        particle_data['Coordinates'] *= u.cm.to(u.Mpc)

        # Periodic wrap coordinates around centre #
        # boxsize = self.boxsize / self.h
        # particle_data['Coordinates'] = np.mod(particle_data['Coordinates'] - subhalo_data['CentreOfMass'] + 0.5 * boxsize, boxsize) + subhalo_data[
        #     'CentreOfMass'] - 0.5 * boxsize

        return particle_data


    def plot(self):
        """

        :return: None
        """

        plt.close()
        plt.figure()

        # Plot.
        plt.hist(self.stars['Coordinates'], c='red', s=3, edgecolor='none')

        # Save plot.
        plt.xlabel('GasMass]')
        plt.ylabel('StarsMass')
        plt.tight_layout()
        plt.savefig('PositionHistogram' + '-' + date + '.png')

        return None


if __name__ == '__main__':
    x = PositionHistogram(1, 0)