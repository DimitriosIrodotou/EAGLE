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
from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematics

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create ra and dec plot.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SurfaceDensity:
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
        stellar_data_tmp = {}
        prc_unit_vector, glx_unit_vector = [], []
        
        if not args.l:
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
        
        # for group_number in np.sort(list(set(self.subhalo_data_tmp['GroupNumber']))):  # Loop over all masked haloes.
        for group_number in range(8, 9):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, prc_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    
                    # Save data in numpy arrays #
                    np.save(SavePath + 'group_numbers/' + 'group_number_' + str(group_number), group_number)
                    np.save(SavePath + 'unit_vectors/' + 'unit_vector_' + str(group_number), prc_unit_vector)
                    np.save(SavePath + 'subgroup_numbers/' + 'subgroup_number_' + str(group_number), subgroup_number)
                    np.save(SavePath + 'glx_unit_vectors/' + 'glx_unit_vector_' + str(group_number), glx_unit_vector)
                    np.save(SavePath + 'stellar_data_tmps/' + 'stellar_data_tmp_' + str(group_number), stellar_data_tmp)
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
                    
                    group_number = np.load(SavePath + 'group_numbers/' + 'group_number_' + str(group_number) + '.npy')
                    prc_unit_vector = np.load(SavePath + 'unit_vectors/' + 'unit_vector_' + str(group_number) + '.npy')
                    subgroup_number = np.load(SavePath + 'subgroup_numbers/' + 'subgroup_number_' + str(group_number) + '.npy')
                    glx_unit_vector = np.load(SavePath + 'glx_unit_vectors/' + 'glx_unit_vector_' + str(group_number) + '.npy')
                    stellar_data_tmp = np.load(SavePath + 'stellar_data_tmps/' + 'stellar_data_tmp_' + str(group_number) + '.npy', allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, glx_unit_vector, prc_unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')
        
        print(
            'Finished SurfaceDensity for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')
    
    
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
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
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
        
        # Isolate the desired Group
        radii = np.linalg.norm(stellar_data_tmp['Coordinates'], axis=1)
        radsort = np.argsort(radii)
        stellar_data_tmp['Coordinates'] = stellar_data_tmp['Coordinates'][radsort]
        cum_mass = np.cumsum(stellar_data_tmp['Mass'])
        starinnermass = cum_mass
        starmass = stellar_data_tmp['Mass']
        starpos = stellar_data_tmp['Coordinates']
        starvel = stellar_data_tmp['Velocity']
        massvel = np.array([starvel[i] * starmass[i] for i in range(0, len(starvel))])
        starj = np.array([np.cross(starp, starv) for starp, starv in zip(starpos, massvel)])
        r200j = starj
        tot_ang_mom = np.sum(r200j, axis=0)
        a = np.matrix([tot_ang_mom[0], tot_ang_mom[1], tot_ang_mom[2]]) / np.linalg.norm([tot_ang_mom[0], tot_ang_mom[1], tot_ang_mom[2]])
        b = np.matrix([0, 0, 1])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b.T)
        vx = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
        transform = np.eye(3, 3) + vx + (vx * vx) * ((1 - c[0, 0]) / s ** 2)
        starpos = np.array([np.matmul(transform, starpos[i].T) for i in range(0, len(starpos))])[:, 0]
        starvel = np.array([np.matmul(transform, starvel[i].T) for i in range(0, len(starvel))])[:, 0]
        starr_xy = np.linalg.norm(np.dstack((starpos[:, 0], starpos[:, 1]))[0], axis=1)
        G = 4.302e-9  # Units?
        starv_c = np.sqrt((G * starinnermass) / starr_xy)
        massvel = np.array([starvel[i] * starmass[i] for i in range(0, len(starvel))])
        starj = np.array([np.cross(starp, starv) for starp, starv in zip(starpos, massvel)])
        starjspec = np.array([np.cross(starp, starv) for starp, starv in zip(starpos, starvel)])
        starradii = np.linalg.norm(starpos, axis=1)
        r200j = starj
        tot_ang_mom = np.sum([r200j[:, 0], r200j[:, 1], r200j[:, 2]], axis=1)
        tot_ang_mom = tot_ang_mom / np.linalg.norm(tot_ang_mom)
        print('aligned! angular momentum:' + str(tot_ang_mom))
        starj_z = starjspec[:, 2]
        starj_c = starv_c * starr_xy
        starjz_jc = (starj_z / starj_c)
        
        # rad_low = 0.003
        # rad_high = 0.015
        # rad_bins = np.linspace(rad_low, rad_high, num=15, endpoint=True)
        # radial_fe_h = []
        # radial_mass = []
        # bincenters = []
        # for i in range(0, len(rad_bins) - 1):
        #     bin_mask = [(np.abs(starr_xy) > rad_bins[i]) & (np.abs(starr_xy) < rad_bins[i + 1])]
        #     bincenter = rad_bins[i] + ((rad_bins[i + 1] - rad_bins[i]) / 2)
        #     fe_h_bin = fe_h[bin_mask]
        #     mass_bin = starmass[bin_mask]
        #     bincenters.append(bincenter)
        #     radial_fe_h.append(fe_h_bin)
        #     radial_mass.append(mass_bin)
        # fe_h_av = []
        # for i in range(0, len(radial_fe_h)):
        #     av_fe_h = sum(np.array(radial_fe_h[i] * radial_mass[i])[radial_fe_h[i] != -np.inf]) / sum(radial_mass[i])
        #     fe_h_av.append(av_fe_h)
        #
        # hist, bins = np.histogram(np.abs(starpos[:, 0]), bins=100, range=(0, 0.06))
        # centers = (bins[:-1] + bins[1:]) / 2
        # hist = hist
        #
        #
        # def sersic(r, I_0, r_e, n):
        #     return I_0 * np.exp(-1 * (r / r_e) ** (1 / n))
        #
        #
        # try:
        #     popt, pcov = curve_fit(sersic, centers, hist, p0=[hist[0], 0.01, 1])
        # except RuntimeError:
        #     print
        #     'WARNING: could not fit sersic profile... returning NaN'
        #     popt = [np.nan, np.nan, np.nan]
        #
        # r_e = popt[1]
        # sersic_index = popt[2]
        # disc_mask = (starjz_jc > 0.7) & (starjz_jc < 5) & (starradii > 0.002) & (starradii < 0.02) & (starpos[:, 2] < 0.05)  # Check
        
        # Save the plot #
        plt.savefig(outdir + str(group_number) + str(subgroup_number) + '-' + 'RDSD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    sim = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/SD/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/SD/'  # Path to save/load data.
    x = SurfaceDensity(sim, tag)