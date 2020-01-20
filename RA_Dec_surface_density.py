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
from astropy_healpix import HEALPix
from rotate_galaxies import RotateGalaxies
from morpho_kinematics import MorphoKinematics

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create ra and dec.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADecSurfaceDensity:
    """
    Create a ra and dec plot with the angular momentum of particles for each galaxy.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise arrays and a dictionary to store the data #
        prc_unit_vector = []
        glx_unit_vector = []
        stellar_data_tmp = {}
        
        if not args.l:
            self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all masked haloes.
        for group_number in range(8, 9):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, prc_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    
                    # Save data in numpy arrays #
                    np.save(SavePath + 'unit_vector_' + str(group_number), prc_unit_vector)
                    np.save(SavePath + 'group_number_' + str(group_number), group_number)
                    np.save(SavePath + 'subgroup_number_' + str(group_number), subgroup_number)
                    np.save(SavePath + 'stellar_data_tmp_' + str(group_number), stellar_data_tmp)
                    np.save(SavePath + 'glx_unit_vector_' + str(group_number), glx_unit_vector)
                    print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, prc_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    group_number = np.load(SavePath + 'group_number_' + str(group_number) + '.npy')
                    subgroup_number = np.load(SavePath + 'subgroup_number_' + str(group_number) + '.npy')
                    prc_unit_vector = np.load(SavePath + 'unit_vector_' + str(group_number) + '.npy')
                    glx_unit_vector = np.load(SavePath + 'glx_unit_vector_' + str(group_number) + '.npy')
                    stellar_data_tmp = np.load(SavePath + 'stellar_data_tmp_' + str(group_number) + '.npy', allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, glx_unit_vector, prc_unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished RADecSurfaceDensity for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (
            time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(sim, tag):
        """
         A static method to extract particle and subhalo attributes.
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
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # Msun kpc km s-1
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        # Rotate galaxies #
        xdir, ydir, zdir = RotateGalaxies.get_principal_axis(stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], glx_unit_vector)
        stellar_data_tmp['Coordinates'] = RotateGalaxies.rotate(stellar_data_tmp['Coordinates'], xdir, ydir, zdir)
        stellar_data_tmp['Velocity'] = RotateGalaxies.rotate(stellar_data_tmp['Velocity'], xdir, ydir, zdir)
        
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # Msun kpc km s-1
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # Msun kpc km s-1
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        return stellar_data_tmp, glx_unit_vector, prc_unit_vector
    
    
    @staticmethod
    def plot(stellar_data_tmp, glx_unit_vector, prc_unit_vector, group_number, subgroup_number):
        """
        A method to plot a HEALPix histogram.
        :param stellar_data_tmp: from mask_galaxies
        :param glx_unit_vector: from mask_galaxies
        :param prc_unit_vector: from mask_galaxies
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: None
        """
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)
        
        # Generate the figure #
        plt.close()
        figure = plt.figure(0, figsize=(20, 22.5))
        
        gs = gridspec.GridSpec(3, 2)
        axupperleft = plt.subplot(gs[0, 0], projection='mollweide')
        axupperright = plt.subplot(gs[0, 1])
        axmiddleleft = plt.subplot(gs[1, 0])
        axmiddleright = plt.subplot(gs[1, 1])
        axlowerleft = plt.subplot(gs[2, 0])
        axlowerright = plt.subplot(gs[2, 1])
        
        axupperright.grid(True)
        axmiddleleft.grid(True)
        axmiddleright.grid(True)
        axlowerleft.grid(True)
        axlowerright.grid(True)
        axupperleft.set_xlabel('RA ($\degree$)')
        axupperleft.set_ylabel('Dec ($\degree$)')
        axmiddleleft.set_ylabel('Particles per grid cell')
        axmiddleleft.set_xlabel('Angular distance from X ($\degree$)')
        axupperright.set_ylabel('Particles per grid cell')
        axupperright.set_xlabel('Angular distance from densest grid cell ($\degree$)')
        axlowerleft.set_ylabel('$\mathrm{A_{2}}$')
        axlowerleft.set_xlabel('R [kpc]')
        
        axupperright.set_xlim(-10, 190)
        axmiddleleft.set_xlim(-10, 190)
        axlowerleft.set_ylim(-0.2, 1.2)
        axlowerleft.set_xlim(0.0, 10.0)
        
        y_tick_labels = np.array(['', '-60', '', '-30', '', '0', '', '30', '', 60])
        x_tick_labels = np.array(['', '-120', '', '-60', '', '0', '', '60', '', 120])
        axupperleft.set_xticklabels(x_tick_labels)
        axupperleft.set_yticklabels(y_tick_labels)
        
        axupperleft.annotate(r'0', xy=(0, 0), xycoords='data', size=18)  # Position of 0 degrees.
        axupperleft.annotate(r'60', xy=(np.pi / 3, 0), xycoords='data', size=18)  # Position of 60 degrees.
        axupperleft.annotate(r'-60', xy=(-np.pi / 3, 0), xycoords='data', size=18)  # Position of -60 degrees.
        axupperleft.annotate(r'120', xy=(2 * np.pi / 3, 0), xycoords='data', size=18)  # Position of 120 degrees.
        axupperleft.annotate(r'-120', xy=(-2 * np.pi / 3, 0), xycoords='data', size=18)  # Position of -120 degrees.
        
        axmiddleleft.set_xticks(np.arange(0, 181, 20))
        axupperright.set_xticks(np.arange(0, 181, 20))
        
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
        axupperleft.annotate(r'Density maximum', xy=(lon_densest, lat_densest), xycoords='data', xytext=(0.78, 1.00), textcoords='axes fraction',
                             arrowprops=dict(arrowstyle="-", color='black', connectionstyle="arc3,rad=0"))  # Position of the denset pixel.
        axupperleft.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                            zorder=5)  # Position of the galactic angular momentum.
        
        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)
        
        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = density[coordinate_index]
        
        # Display data on a 2D regular raster and create a pseudo-color plot #
        im = axupperleft.imshow(density_map, cmap='nipy_spectral_r', aspect='auto', norm=matplotlib.colors.LogNorm(vmin=1))
        cbar = plt.colorbar(im, ax=axupperleft, orientation='horizontal')
        cbar.set_label('$\mathrm{Particles\; per\; grid\; cell}$')
        axupperleft.pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        
        # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        index = np.where(angular_theta_from_densest < np.divide(np.pi, 6.0))
        disc_fraction_IT20 = np.divide(np.sum(stellar_data_tmp['Mass'][index]), np.sum(stellar_data_tmp['Mass']))
        
        # Calculate and plot the angular distance between the densest and all the other grid cells - all methods are identical #
        # 1) Spherical law of cosines https://en.wikipedia.org/wiki/Spherical_law_of_cosines
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.radians(dec_grid.value)) + np.cos(lat_densest) * np.cos(np.radians(dec_grid.value)) * np.cos(
                lon_densest - np.radians(ra_grid.value)))  # In radians.
        
        # # 2) Haversine formula https://en.wikipedia.org/wiki/Haversine_formula
        # # delt_lat = (np.subtract(position_densest[0, 1], position_other[:, 1]))
        # # delta_lon = (np.subtract(position_densest[0, 0], position_other[:, 0]))
        # # angular_theta_from_densest = 2.0 * np.arcsin(
        # #     np.sqrt(np.sin(delt_lat / 2.0) ** 2 + np.cos(position_densest[0, 1]) * np.cos(position_other[:, 1]) * np.sin(delta_lon / 2.0) ** 2))
        #
        # # 3) Vincenty's formula https://en.wikipedia.org/wiki/Vincenty%27s_formulae
        # # sdlon = np.sin(position_densest[0, 0] - position_other[:, 0])
        # # cdlon = np.cos(position_densest[0, 0] - position_other[:, 0])
        # # slat1 = np.sin(position_densest[0, 1])
        # # slat2 = np.sin(position_other[:, 1])
        # # clat1 = np.cos(position_densest[0, 1])
        # # clat2 = np.cos(position_other[:, 1])
        # # num1 = clat2 * sdlon
        # # num2 = clat1 * slat2 - slat1 * clat2 * cdlon
        # # denominator = slat1 * slat2 + clat1 * clat2 * cdlon
        # # angular_theta_from_densest = np.arctan2(np.hypot(num1, num2), denominator)
        #
        # # 4) Chord length
        # # deltax = np.cos(position_densest[0, 1]) * np.cos(position_densest[0, 0]) - np.cos(position_other[:, 1]) * np.cos(position_other[:, 0])
        # # deltay = np.cos(position_densest[0, 1]) * np.sin(position_densest[0, 0]) - np.cos(position_other[:, 1]) * np.sin(position_other[:, 0])
        # # deltaz = np.sin(position_densest[0, 1]) - np.sin(position_other[:, 1])
        # # c = np.sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz)
        # # angular_theta_from_densest = 2 * np.arcsin(c / 2)
        
        axupperright.scatter(angular_theta_from_densest[density_map.nonzero()] * np.divide(180.0, np.pi), density_map[density_map.nonzero()],
                             c='black', s=10)  # In degrees.
        axupperright.axvline(x=30, c='blue', lw=3, linestyle='dashed', label='D/T= %.3f ' % disc_fraction_IT20)  # Vertical line at 30 degrees.
        axupperright.axvspan(0, 30, facecolor='0.2', alpha=0.5)  # Draw a vertical span.
        axupperright.legend(loc='upper center', fontsize=16, frameon=False, numpoints=1)
        
        # Calculate and plot the angular distance between the (unit vector of) the galactic angular momentum and all the other grid cells #
        position_of_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T
        
        angular_theta_from_X = np.arccos(np.sin(position_of_X[0, 1]) * np.sin(np.radians(dec_grid.value)) + np.cos(position_of_X[0, 1]) * np.cos(
            np.radians(dec_grid.value)) * np.cos(position_of_X[0, 0] - np.radians(ra_grid.value)))  # In radians.
        axmiddleleft.scatter(angular_theta_from_X[density_map.nonzero()] * np.divide(180.0, np.pi), density_map[density_map.nonzero()], c='black',
                             s=10)  # In degrees.
        axmiddleleft.axvline(x=90, c='red', lw=3, linestyle='dashed')  # Vertical line at 30 degrees.
        axmiddleleft.axvspan(90, 180, facecolor='0.2', alpha=0.5)  # Draw a vertical span.
        
        # Calculate kinematic diagnostics #
        kappa, discfrac, orbital, vrotsig, vrots, delta, zaxis, momentum = MorphoKinematics.kinematics_diagnostics(stellar_data_tmp['Coordinates'],
                                                                                                                   stellar_data_tmp['Mass'],
                                                                                                                   stellar_data_tmp['Velocity'],
                                                                                                                   stellar_data_tmp[
                                                                                                                       'ParticleBindingEnergy'])
        
        # Calculate the distribution of orbital circularity #
        j, = np.where((orbital < 0.0))
        k, = np.where((orbital > 0.7) & (orbital < 1.7))
        l, = np.where((orbital > -1.7) & (orbital < 1.7))
        disc_fraction_00 = 1 - 2 * np.sum(stellar_data_tmp['Mass'][j]) / np.sum(stellar_data_tmp['Mass'][l])
        disc_fraction_07 = np.sum(stellar_data_tmp['Mass'][k]) / np.sum(stellar_data_tmp['Mass'][l])
        
        ydata, edges = np.histogram(orbital, bins=100, range=[-1.7, 1.7], weights=stellar_data_tmp['Mass'] / np.sum(stellar_data_tmp['Mass']))
        ydata /= edges[1:] - edges[:-1]
        axmiddleright.plot(0.5 * (edges[1:] + edges[:-1]), ydata, label='D/T = %.3f [%.3f]' % (disc_fraction_07, disc_fraction_00))
        axmiddleright.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
        
        # Calculate bar strength from Fourier modes of surface density as a function of radius plot #
        nbins = 40  # Number of radial bins.
        r = np.sqrt(stellar_data_tmp['Coordinates'][:, 0] ** 2 + stellar_data_tmp['Coordinates'][:, 1] ** 2)  # Radius of each particle.
        
        # Initialise Fourier components #
        r_m = np.zeros(nbins)
        beta_2 = np.zeros(nbins)
        alpha_0 = np.zeros(nbins)
        alpha_2 = np.zeros(nbins)
        
        # Split up galaxy in radius bins and calculate Fourier components #
        for i in range(0, nbins):
            r_s = float(i) * 0.25
            r_b = float(i) * 0.25 + 0.25
            r_m[i] = float(i) * 0.25 + 0.125
            xfit = stellar_data_tmp['Coordinates'][:, 0][(r < r_b) & (r > r_s)]
            yfit = stellar_data_tmp['Coordinates'][:, 1][(r < r_b) & (r > r_s)]
            for k in range(0, len(xfit)):
                th_i = np.arctan2(yfit[k], xfit[k])
                alpha_0[i] = alpha_0[i] + 1
                alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                beta_2[i] = beta_2[i] + np.sin(2 * th_i)
        
        # Calculate bar strength A_2 #
        a2 = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])
        
        # Plot bar strength #
        axlowerleft.plot(r_m, a2, label="bar strength: %.2f" % max(a2))
        axlowerleft.legend(loc='upper left', fontsize=12, frameon=False, numpoints=1)
        
        # Save the plot #
        plt.savefig(outdir + str(group_number) + str(subgroup_number) + '-' + 'RDSD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    # tag = '010_z005p000'
    # sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    # outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RDSD/G-EAGLE/'  # Path to save plots.
    # SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/RDSD/G-EAGLE/'  # Path to save/load data.
    tag = '027_z000p101'
    sim = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
    outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RDSD/EAGLE/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/RDSD/EAGLE/'  # Path to save/load data.
    x = RADecSurfaceDensity(sim, tag)