import os
import re
import time
import warnings
import argparse
import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from scipy.special import gamma
from astropy_healpix import HEALPix
from scipy.optimize import curve_fit
from rotate_galaxies import RotateCoordinates

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
            self.stellar_data, self.subhalo_data = self.read_attributes(simulation_path, tag)
            print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher that 1e9
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all masked haloes.
        for group_number in range(1, 26):
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    
                    # Save data in numpy arrays #
                    np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
                    print('Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                        time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    print('Masked data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                        time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp = np.load(
                        data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                        allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print(
            'Finished SurfaceDensityProfiles for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
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
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e9)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        halo_mask, = np.where((self.subhalo_data_tmp['GroupNumber'] == group_number) & (self.subhalo_data_tmp['SubGroupNumber'] == subgroup_number))
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        galaxy_mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask], axis=1) <= 30.0))  # In kpc.
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[galaxy_mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = stellar_data_tmp['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask]
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass'], axis=0))  # In km s-1.
        stellar_data_tmp['Velocity'] = stellar_data_tmp['Velocity'] - CoM_velocity
        
        return stellar_data_tmp
    
    
    def plot(self, stellar_data_tmp, group_number, subgroup_number):
        """
        Plot surface density profiles.
        :param stellar_data_tmp: from mask_galaxies
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: None
        """
        
        
        # Declare the Sersic, exponential and total profiles #
        def sersic_profile(r, I_0b, b, n):
            return I_0b * np.exp(-(r / b) ** (1 / n))  # b = R_eff / b_n ^ n
        
        
        def exponential_profile(r, I_0d, R_d):
            return I_0d * np.exp(-r / R_d)
        
        
        def total_profile(r, I_0d, R_d, I_0b, b, n):
            y = exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
            return (y)
        
        
        def sersic_b_n(n):
            if n <= 0.36:
                b_n = 0.01945 + n * (- 0.8902 + n * (10.95 + n * (- 19.67 + n * 13.43)))
            else:
                x = 1.0 / n
                b_n = -1.0 / 3.0 + 2. * n + x * (4.0 / 405. + x * (46. / 25515. + x * (131. / 1148175 - x * 2194697. / 30690717750.)))
            return b_n
        
        
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(0, figsize=(10, 7.5))
        plt.grid(True)
        plt.yscale('log')
        plt.axis([0.0, 30.0, 1e6, 1e10])
        plt.xlabel("$\mathrm{R\;[kpc]}$", size=16)
        plt.ylabel("$\mathrm{\Sigma\;[M_{\odot}\;kpc^{-2}]}$", size=16)
        plt.tick_params(direction='out', which='both', top='on', right='on', labelsize=16)
        
        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(
            stellar_data_tmp)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Plot a HEALPix histogram #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.
        
        # Find location of density maximum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate and plot the disc (bulge) mass surface density as the mass within (outside) 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask, = np.where(angular_theta_from_densest < np.divide(np.pi, 6.0))
        bulge_mask, = np.where(angular_theta_from_densest > np.divide(np.pi, 6.0))
        
        colors = ['blue', 'red']
        labels = ['Disc', 'Bulge']
        labels2 = ['Exponential', 'Sersic']
        masks = [disc_mask, bulge_mask]
        profiles = [exponential_profile, sersic_profile]
        for mask, color, profile, label, label2 in zip(masks, colors, profiles, labels, labels2):
            cylindrical_distance = np.sqrt(
                stellar_data_tmp['Coordinates'][mask, 0] ** 2 + stellar_data_tmp['Coordinates'][mask, 1] ** 2)  # Radius of each particle.
            vertical_mask, = np.where(abs(stellar_data_tmp['Coordinates'][:, 2][mask]) < 5)  # Vertical cut in kpc.
            component_mass = stellar_data_tmp['Mass'][mask]
            
            mass, edges = np.histogram(cylindrical_distance[vertical_mask], bins=50, range=(0, 30), weights=component_mass[vertical_mask])
            centers = 0.5 * (edges[1:] + edges[:-1])
            surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            sden = np.divide(mass, surface)
            
            plt.errorbar(centers, sden, yerr=0.1 * sden, c=color, marker='.', linestyle="None", elinewidth=1, capsize=2, capthick=1, zorder=3,
                         label=label)
            
            try:
                if mask is disc_mask:
                    popt, pcov = curve_fit(profile, centers, sden, sigma=0.1 * sden, p0=[sden[0], 2])  # p0 = [I_0d, R_d]
                    plt.plot(centers, profile(centers, popt[0], popt[1]), c=color, label=label2)
                    sden_tmp = sden
                    
                    # Calculate disc attributes #
                    I_0d, R_d = popt[0], popt[1]
                    disk_mass = 2.0 * np.pi * I_0d * R_d ** 2
                
                elif mask is bulge_mask:
                    popt, pcov = curve_fit(profile, centers, sden, sigma=0.1 * sden, p0=[sden[0], 2, 4])  # p0 = [I_0b, b, n]
                    plt.plot(centers, profile(centers, popt[0], popt[1], popt[2]), c=color, label=label2)
                    
                    # Calculate bulge attributes #
                    I_0b, b, n = popt[0], popt[1], popt[2]
                    R_eff = b * sersic_b_n(n) ** n
                    bulge_mass = np.pi * I_0b * R_eff ** 2 * gamma(2.0 / n + 1)
            
            except RuntimeError:
                print('WARNING: Could not fit a profile')
        
        cylindrical_distance = np.sqrt(
            stellar_data_tmp['Coordinates'][:, 0] ** 2 + stellar_data_tmp['Coordinates'][:, 1] ** 2)  # Radius of each particle.
        vertical_mask, = np.where(abs(stellar_data_tmp['Coordinates'][:, 2]) < 5)  # Vertical cut in kpc.
        
        mass, edges = np.histogram(cylindrical_distance[vertical_mask], bins=50, range=(0, 30), weights=stellar_data_tmp['Mass'][vertical_mask])
        centers = 0.5 * (edges[1:] + edges[:-1])
        surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sden = np.divide(mass, surface)
        
        plt.errorbar(centers, sden, yerr=0.1 * sden, c='k', marker='.', linestyle="None", elinewidth=1, capsize=2, capthick=1, zorder=3,
                     label='Total')
        
        try:
            popt, pcov = curve_fit(total_profile, centers, sden, sigma=0.1 * sden, p0=[sden_tmp[0], 2, sden[0], 2, 4])  # p0 = [I_0d, R_d, I_0b, b, n]
            plt.plot(centers, total_profile(centers, popt[0], popt[1], popt[2], popt[3], popt[4]), c='k', label=r'$\mathrm{Total}$')
            
            # Calculate galactic attributes #
            I_0d, R_d, I_0b, b, n = popt[0], popt[1], popt[2], popt[3], popt[4]
            R_eff = b * sersic_b_n(n) ** n
            disk_mass = 2.0 * np.pi * I_0d * R_d ** 2
            bulge_mass = np.pi * I_0b * R_eff ** 2 * gamma(2.0 / n + 1)
            disk_fraction = disk_mass / (bulge_mass + disk_mass)
        
        except RuntimeError:
            print('WARNING: Could not fit a profile')
        
        figure.text(0.5, 0.72, '\n' r'$\mathrm{n}=%.2f$' '\n' r'$\mathrm{R_{d}}=%.2f$ kpc' '\n' r'$\mathrm{R_{eff}}=%.2f$ kpc' '\n' % (n, R_d, R_eff),
                    size=16)
        
        # Create the legend and save the figure #
        plt.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'SDP' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/SDP/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SurfaceDensityProfiles(simulation_path, tag)
