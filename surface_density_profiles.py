import re
import time
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

from matplotlib import gridspec
from scipy.special import gamma
from astropy_healpix import HEALPix
from scipy.optimize import curve_fit
from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematics

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Fit profiles.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SurfaceDensityProfiles:
    """
    Fit surface density profiles to the components produced by RA_Dec_surface_density.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        stellar_data_tmp = {}  # Initialise a dictionary to store the data.
        
        if not args.l:
            # Extract particle and subhalo attributes and convert them to astronomical units #
            self.stellar_data, self.subhalo_data = self.read_galaxies(simulation_path, tag)
            print('Read data for ' + re.split('EAGLE/|/data', simulation_path)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher that 1e8
        
        # for group_number in np.sort(list(set(self.subhalo_data_tmp['GroupNumber']))):  # Loop over all masked haloes.
        for group_number in range(25, 26):
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    
                    # Save data in numpy arrays #
                    np.save(data_path + 'group_numbers/' + 'group_number_' + str(group_number), group_number)
                    np.save(data_path + 'subgroup_numbers/' + 'subgroup_number_' + str(group_number), subgroup_number)
                    np.save(data_path + 'stellar_data_tmps/' + 'stellar_data_tmp_' + str(group_number), stellar_data_tmp)
                    print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    group_number = np.load(data_path + 'group_numbers/' + 'group_number_' + str(group_number) + '.npy')
                    subgroup_number = np.load(data_path + 'subgroup_numbers/' + 'subgroup_number_' + str(group_number) + '.npy')
                    stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/' + 'stellar_data_tmp_' + str(group_number) + '.npy', allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished SurfaceDensityProfiles in %.4s s' % (time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        :return: stellar_data, subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e8 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the data to select haloes more #
        mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e8)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp
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
        
        return stellar_data_tmp
    
    
    def plot(self, stellar_data_tmp, group_number, subgroup_number):
        """
        Plot surface density profiles.
        :param stellar_data_tmp: from mask_galaxies
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: None
        """
        
        f = plt.figure(0, figsize=(10, 7.5))
        plt.ylim(1e-1, 1e6)
        plt.xlim(0.0, 30.0)
        plt.grid(True)
        plt.xlabel("$\mathrm{R [kpc]}$", size=16)
        plt.ylabel("$\mathrm{\Sigma [M_{\odot} pc^{-2}]}$", size=16)
        plt.tick_params(direction='out', which='both', top='on', right='on')
        
        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(
            stellar_data_tmp)
        
        # prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        #
        # # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        # ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        # dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        #
        # # Plot a HEALPix histogram #
        # nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        # hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        # indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        # density = np.bincount(indices, minlength=hp.npix)  # Count number of points in each HEALPix pixel.
        #
        # # Find location of density maximum and plot its positions and the ra and dec of the galactic angular momentum #
        # index_densest = np.argmax(density)
        # lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        # lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        #
        # # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        # angular_theta_from_densest = np.arccos(
        #     np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:,
        #     2])) * np.cos(
        #         lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        # index = np.where(angular_theta_from_densest > np.divide(np.pi, 6.0))[0]
        #
        # pos = np.fliplr(stellar_data_tmp['Coordinates'][index])
        pos = stellar_data_tmp['Coordinates'] * 1e-3
        
        zcut = 0.001  # vertical cut in Mpc
        Rcut = 0.030
        bins = 60
        rad = np.sqrt((pos[:, :2] ** 2).sum(axis=1))
        
        ii, = np.where((abs(pos[:, 2]) < zcut))
        
        sden, edges = np.histogram(rad[ii], bins=bins, range=(0., Rcut), weights=stellar_data_tmp['Mass'][ii])
        sa = np.zeros(len(edges) - 1)
        sa[:] = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sden /= sa
        
        x = np.zeros(len(edges) - 1)
        x[:] = 0.5 * (edges[1:] + edges[:-1])
        sden *= 1e-6
        r = x * 1e3
        
        sdlim = 1.
        indy = self.find_nearest(sden * 1e4, [sdlim]).astype('int64')
        
        sdfit = sden[:indy]
        r = r[:indy][sdfit > 0.]
        sdfit = sdfit[sdfit > 0.]
        
        
        def total_profile(x, da, h, ba, b, n):
            y = exponential_profile(x, da, h) + sersic_profile(x, ba, b, n)
            return (y)
        
        
        def sersic_profile(r, I_0, r_e, n):
            return I_0 * np.exp(-1 * (r / r_e) ** (1 / n))
        
        
        def exponential_profile(x, da, h):
            return da * np.exp(-x / h)
        
        
        allowonecompfit = True
        flag = 0
        try:
            guess = (0.1, 2., 0.4, 0.6, 1.)  # guess = (exp amp, Rd, sersic amp, sersic b, 1/n)
            # guess = (1., 4., 1., 1., 1.)
            
            bounds = ([0.01, 0., 0.01, 0.5, 0.25], [1., 6., 10., 2., 10.])
            sigma = 0.1 * sdfit
            (popt, pcov) = curve_fit(total_profile, r, sdfit, guess, sigma=sigma, bounds=bounds)
            
            if allowonecompfit is True:
                perr = np.sqrt(np.diag(pcov)).mean()
                try:
                    bguess = guess[2:]
                    bounds = ([0., 0., 0.5], [10., 10., 4.])
                    (popt2, pcov2) = curve_fit(p.sersic_prof1, r, sdfit, bguess, sigma=sigma, bounds=bounds)
                    perr2 = np.sqrt(np.diag(pcov2)).mean()
                    if perr2 < perr:
                        print("**sersic profile preferred")
                        popt = popt2
                        flag = 1
                except:
                    pass
        
        except:
            popt = np.zeros(5)
            print("fitting failed")
        
        disk_mass = 2.0 * np.pi * popt[0] * popt[1] * popt[1]
        bulge_mass = np.pi * popt[2] * popt[3] * popt[3] * gamma(2.0 / popt[4] + 1)
        disk_to_total = disk_mass / (bulge_mass + disk_mass)
        print(disk_to_total)
        
        plt.semilogy(r, sdfit * 1e-6, 'o', markersize=1.5, color='k', linewidth=0.)
        if flag == 1:
            plt.semilogy(r, 1e10 * sersic_profile(r, popt[0], popt[1], popt[2]) * 1e-6, 'r--', label=r'n=%.2f' % (1. / popt[2]))
        if flag == 0 or flag == 2:
            plt.semilogy(r, 1e10 * exponential_profile(r, popt[0], popt[1]) * 1e-6, 'b-', label=r'$R_d = %.2f$' % (popt[1]))
        if flag == 0:
            plt.semilogy(r, 1e10 * sersic_profile(r, popt[2], popt[3], popt[4]) * 1e-6, 'r--', label=r'n=%.2f' % (1. / popt[4]))
            plt.semilogy(r, 1e10 * total_profile(r, popt[0], popt[1], popt[2], popt[3], popt[4]) * 1e-6, 'k-')
        
        plt.legend(loc='center right')
        
        # Save the plot #
        plt.savefig(plots_path + str(group_number) + str(subgroup_number) + '-' + 'SDP' + '-' + date + '.png', bbox_inches='tight')
        return None
    
    
    def find_nearest(self, array, value):
        if len(value) == 1:
            idx = (np.abs(array - value)).argmin()
        else:
            idx = np.zeros(len(value))
            for i in range(len(value)):
                idx[i] = (np.abs(array - value[i])).argmin()
        
        return idx


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/SDP/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = SurfaceDensityProfiles(simulation_path, tag)