import re
import time
import glob
import warnings
import argparse
import matplotlib

matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from astropy_healpix import HEALPix
from morpho_kinematics import MorphoKinematics

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create D/T vs kappa_rot plot.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DiscToTotalVsKappaRot:
    """
    Create a disc to total ratio as a function of kappa_rot plot.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise empty arrays to hold the data #
        kappas = []
        disc_fractions = []
        disc_fractions_IT20 = []
        
        if not args.l:
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
            
            # Check if the data already exists, if not then read and save it #
            # names = glob.glob(SavePath + 'kappa_*' + '.npy')
            # names = [re.split('_|.npy', name)[1] for name in names]
            # if not glob.glob(SavePath + 'kappas.npy'):
            for group_number in np.sort(list(set(self.subhalo_data_tmp['GroupNumber']))):  # Loop over all the accepted haloes
                # for group_number in names:  # Loop over all masked haloes.
                for subgroup_number in range(0, 1):
                    if args.rs:  # Read and save data.
                        start_local_time = time.time()  # Start the local time.
                        
                        kappa, disc_fraction, disc_fraction_IT20 = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                        
                        # Save data in numpy arrays every 10 galaxies to make it faster #
                        np.save(SavePath + 'kappas/' + 'kappa_' + str(group_number), kappa)
                        np.save(SavePath + 'disc_feactions/' + 'disc_fraction_' + str(group_number), disc_fraction)
                        np.save(SavePath + 'disc_fractions_IT20/' + 'disc_fraction_IT20_' + str(group_number), disc_fraction_IT20)
                        print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                            round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                        print('–––––––––––––––––––––––––––––––––––––––––––––')
                        p += 1
                    elif args.r:  # Read data.
                        start_local_time = time.time()  # Start the local time.
                        
                        kappa, disc_fraction, disc_fraction_IT20 = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                        kappas.append(kappa)
                        disc_fractions.append(disc_fraction)
                        disc_fractions_IT20.append(disc_fraction_IT20)
                        print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                            round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                        print('–––––––––––––––––––––––––––––––––––––––––––––')
                        p += 1  # Increase the count by one.
                    
                    if args.l or args.rs:  # Load data.
                        start_local_time = time.time()  # Start the local time.
                        kappa = np.load(SavePath + 'kappas/' + 'kappa_' + str(group_number) + '.npy')
                        disc_fraction = np.load(SavePath + 'disc_fractions/' + 'disc_fraction_' + str(group_number) + '.npy')
                        disc_fraction_IT20 = np.load(SavePath + 'disc_fractions_IT20/' + 'disc_fraction_IT20_' + str(group_number) + '.npy')
                        kappas.append(kappa)
                        disc_fractions.append(disc_fraction)
                        disc_fractions_IT20.append(disc_fraction_IT20)
                        print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                        # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                        print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            if args.l or args.rs:  # Load data.
                np.save(SavePath + 'kappas/' + 'kappas', kappas)
                np.save(SavePath + 'disc_fractions/' + 'disc_fractions', disc_fractions)
                np.save(SavePath + 'disc_fractions_IT20/' + 'disc_fractions_IT20', disc_fractions_IT20)
        else:
            start_local_time = time.time()  # Start the local time.
            
            kappas = np.load(SavePath + 'kappas/' + 'kappas.npy')
            disc_fractions = np.load(SavePath + 'disc_fractions/' + 'disc_fractions.npy')
            disc_fractions_IT20 = np.load(SavePath + 'disc_fractions_IT20/' + 'disc_fractions_IT20.npy')
            print('Loaded data for ' + re.split('EAGLE/|/data', sim)[0] + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(kappas, disc_fractions, disc_fractions_IT20)
        print('Plotted data for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished DiscToTotalVsKappaRot for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(sim, tag):
        """
         A method to extract particle and subhalo attributes.
        :param sim: simulation directory
        :param tag: redshift folder
        :return: stellar_data, subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, sim, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'ParticleBindingEnergy', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        A method to mask haloes.
        :return: subhalo_data_tmp
        """
        
        # Mask the data to select haloes more #
        mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 2.5e8)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        A method to mask galaxies.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, prc_unit_vector
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data_tmp['GroupNumber'] == group_number)[0][subgroup_number]
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(np.subtract(self.stellar_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index]), axis=1) <= 30.0))  # kpc
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass'], axis=0))  # km s-1
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)
        
        # Compute the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # Msun kpc km s-1
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        # Calculate kinematic diagnostics #
        kappa, disc_fraction, orbital, vrotsig, vrots, delta, zaxis, momentum = MorphoKinematics.kinematics_diagnostics(
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'], stellar_data_tmp['ParticleBindingEnergy'])
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of points in each HEALPix pixel.
        
        # Find location of density maximum and plot its positions and the ra and dec of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        index = np.where(angular_theta_from_densest < np.divide(np.pi, 6.0))
        disc_fraction_IT20 = np.divide(np.sum(stellar_data_tmp['Mass'][index]), np.sum(stellar_data_tmp['Mass']))
        
        return kappa, disc_fraction, disc_fraction_IT20
    
    
    @staticmethod
    def plot(kappas, disc_fractions, disc_fractions_IT20):
        """
        A method to plot a HEALPix histogram.
        :param kappas: from mask_galaxies
        :param disc_fractions: from mask_galaxies
        :param disc_fractions_IT20: from mask_galaxies
        :return: None
        """
        
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.5)
        
        # Generate initial figure #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        grid = plt.GridSpec(1, 2, width_ratios=(1, 0.25), wspace=0.0)
        main_plot = figure.add_subplot(grid[0, 0])
        y_hist = figure.add_subplot(grid[0, 1])
        for a in [main_plot, y_hist]:
            a.grid(True)
            a.set_ylabel('D/T')
            a.set_ylim(-0.2, 1)
        
        main_plot.set_xlim(0, 1)
        main_plot.set_xlabel('$\kappa_{rot}$')
        y_hist.xaxis.set_ticks_position("top")
        y_hist.yaxis.set_ticks_position("right")
        
        # Calculate median and 1-sigma #
        nbin = int((max(kappas) - min(kappas)) / 0.02)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(kappas)
        for i in range(nbin):
            index = np.where((kappas >= x_low) & (kappas < x_low + 0.02))[0]
            x_value[i] = np.mean(np.absolute(kappas)[index])
            if len(index) > 0:
                median[i] = np.nanmedian(disc_fractions_IT20[index])
                slow[i] = np.nanpercentile(disc_fractions_IT20[index], 15.87)
                shigh[i] = np.nanpercentile(disc_fractions_IT20[index], 84.13)
            x_low += 0.02
        
        # Plot median and 1-sigma lines #
        median_IT20, = main_plot.plot(x_value, median, color='black', zorder=5)
        main_plot.fill_between(x_value, shigh, slow, color='black', alpha='0.5', zorder=5)
        fill_IT20, = plt.fill(np.NaN, np.NaN, color='black', alpha=0.5, zorder=5)
        
        # Calculate median and 1-sigma #
        nbin = int((max(kappas) - min(kappas)) / 0.02)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(kappas)
        for i in range(nbin):
            index = np.where((kappas >= x_low) & (kappas < x_low + 0.02))[0]
            x_value[i] = np.mean(np.absolute(kappas)[index])
            if len(index) > 0:
                median[i] = np.nanmedian(disc_fractions[index])
                slow[i] = np.nanpercentile(disc_fractions[index], 15.87)
                shigh[i] = np.nanpercentile(disc_fractions[index], 84.13)
            x_low += 0.02
        
        # Plot median and 1-sigma lines #
        median, = main_plot.plot(x_value, median, color='blue', zorder=5)
        main_plot.fill_between(x_value, shigh, slow, color='blue', alpha='0.5', zorder=5)
        fill, = plt.fill(np.NaN, np.NaN, color='black', alpha=0.5, zorder=5)
        
        main_plot.scatter(kappas, disc_fractions_IT20, s=1, label='$D/T_{30\degree}$', color='black')
        main_plot.scatter(kappas, disc_fractions, s=1, label=r'$D/T_{\vec{J}_{b} = 0}$', color='brown')
        main_plot.legend(loc='upper left', frameon=False, markerscale=5)
        # main_plot.legend([median_IT20, fill_IT20, median, fill],
        #                  [r'$\mathrm{This\; work: Median}$', r'$\mathrm{This\; work: 16^{th}-84^{th}\,\%ile}$'], frameon=False, loc=2)
        
        y_hist.hist(disc_fractions_IT20, bins=np.linspace(-1, 1, 50), histtype='step', orientation='horizontal', color='black')
        y_hist.hist(disc_fractions, bins=np.linspace(-1, 1, 50), histtype='step', orientation='horizontal', color='brown')
        
        # Save the plot #
        plt.savefig(outdir + 'DTTK' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    sim = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/DTTK/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsKappaRot(sim, tag)