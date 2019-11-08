import re
import time
import warnings
import argparse

import numpy as np
import seaborn as sns
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec
from scipy import interpolate, linalg

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create RA and Dec.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADec:
    """
    Create a RA and Dec plot with the angular momemntum of particles for each galaxy.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise arrays and a dictionary to store the data #
        unit_vector = []
        stellar_data_tmp = {}
        glx_unit_vector = []
        
        if not args.l:
            self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all the accepted haloes
        for group_number in range(1, 20):  # Loop over all the accepted haloes
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    
                    # Save data in nampy arrays #
                    np.save(SavePath + 'unit_vector_' + str(group_number), unit_vector)
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
                    
                    stellar_data_tmp, glx_unit_vector, unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    group_number = np.load(SavePath + 'group_number_' + str(group_number) + '.npy')
                    subgroup_number = np.load(SavePath + 'subgroup_number_' + str(group_number) + '.npy')
                    unit_vector = np.load(SavePath + 'unit_vector_' + str(group_number) + '.npy')
                    glx_unit_vector = np.load(SavePath + 'glx_unit_vector' + str(group_number) + '.npy')
                    stellar_data_tmp = np.load(SavePath + 'stellar_data_tmp_' + str(group_number) + '.npy', allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, glx_unit_vector, unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished RADec for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))  # Print total time.
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
        for attribute in ['BirthDensity', 'Coordinates', 'GroupNumber', 'Mass', 'ParticleBindingEnergy', 'StellarFormationTime', 'SubGroupNumber',
                          'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        stellar_data['BirthDensity'] /= 6.769911178294543e-31  # Convert back to physical units.
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        A method to mask haloes.
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
        A method to mask galaxies.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, unit_vector
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
        
        # Normalise the coordinates and velocities wrt the centre of mass of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass']))  # km s-1
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)
        
        # Compute the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # Msun kpc km s-1
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # Msun kpc km s-1
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        return stellar_data_tmp, glx_unit_vector, unit_vector
    
    
    @staticmethod
    def kinematics_diagnostics(XYZ, mass, Vxyz, PBE, aperture=0.03, CoMvelocity=True):
        """
                Compute the various kinematics diagnostics.
        XYZ : array_like of dtype float, shape (n, 3)
            Particles coordinates (in unit of length L) such that XYZ[:,0] = X,
            XYZ[:,1] = Y & XYZ[:,2] = Z
        mass : array_like of dtype float, shape (n, )
            Particles masses (in unit of mass M)
        Vxyz : array_like of dtype float, shape (n, 3)
            Particles coordinates (in unit of velocity V) such that Vxyz[:,0] = Vx,
            Vxyz[:,1] = Vy & Vxyz[:,2] = Vz
        PBE : array_like of dtype float, shape (n, )
            Particles specific binding energies
        aperture : float, optional
            Aperture (in unit of length L) for the computation. Default is 0.03 L
        CoMvelocity : bool, optional
            Boolean to allow the centering of velocities by the considered particles
            centre-of-mass velocity. Default to True

        Returns
        -------
        kappa : float
            The kinetic energy fraction invested in co-rotation.
        discfrac : float
            The disc-to-total mass fraction estimated from the counter-rotating
            bulge.
        orbi : float
            The median orbital circularity of the particles values.
        vrotsig : float
            The rotation-to-dispersion ratio .
        delta : float
            The dispersion anisotropy.
        zaxis : array of dtype float, shape (3, )
            The unit vector of the momentum axis (pointing along the momentum direction).
        Momentum : float
            The momentum magnitude (in unit M.L.V).
        :param mass:
        :param Vxyz:
        :param PBE:
        :param aperture:
        :param CoMvelocity:
        :return:
        """
        particlesall = np.vstack([XYZ.T, mass, Vxyz.T, PBE]).T
        # Compute distances
        distancesall = np.linalg.norm(particlesall[:, :3], axis=1)
        # Restrict particles
        extract = (distancesall < aperture)
        particles = particlesall[extract].copy()
        distances = distancesall[extract].copy()
        Mass = np.sum(particles[:, 3])
        if CoMvelocity:
            # Compute CoM velocty & correct
            dvVmass = np.nan_to_num(np.sum(particles[:, 3][:, np.newaxis] * particles[:, 4:7], axis=0) / Mass)
            particlesall[:, 4:7] -= dvVmass
            particles[:, 4:7] -= dvVmass
        # Compute momentum
        smomentums = np.cross(particles[:, :3], particles[:, 4:7])
        momentum = np.sum(particles[:, 3][:, np.newaxis] * smomentums, axis=0)
        Momentum = np.linalg.norm(momentum)
        # Compute cylindrical quantities
        zaxis = (momentum / Momentum)
        zheight = np.sum(zaxis * particles[:, :3], axis=1)
        cylposition = particles[:, :3] - zheight[:, np.newaxis] * [zaxis]
        cyldistances = np.sqrt(distances ** 2 - zheight ** 2)
        smomentumz = np.sum(zaxis * smomentums, axis=1)
        vrots = smomentumz / cyldistances
        vrads = np.sum(cylposition * particles[:, 4:7] / cyldistances[:, np.newaxis], axis=1)
        vheis = np.sum(zaxis * particles[:, 4:7], axis=1)
        # Compute co-rotational kinetic energy fraction
        Mvrot2 = np.sum((particles[:, 3] * vrots ** 2)[vrots > 0])
        kappa = Mvrot2 / np.sum(particles[:, 3] * (np.linalg.norm(particles[:, 4:7], axis=1)) ** 2)
        # Compute disc-to-total ratio
        discfrac = 1 - 2 * np.sum(particles[vrots <= 0, 3]) / Mass
        # Compute orbital circularity
        sbindingenergy = particles[:, 7]
        sortE = np.argsort(sbindingenergy)
        unsortE = np.argsort(sortE)
        jzE = np.vstack([sbindingenergy, smomentumz]).T[sortE]
        orbital = (jzE[:, 1] / np.maximum.accumulate(np.abs(jzE[:, 1])))[unsortE]
        orbi = np.median(orbital)
        # Compute rotation-to-dispersion and dispersion anisotropy
        Vrot = np.abs(RADec.cumsummedian(vrots, weights=particles[:, 3]))
        SigmaXY = np.sqrt(np.average(np.sum(particles[:, [3]] * np.vstack([vrads, vrots]).T ** 2, axis=0) / Mass))  #
        SigmaO = np.sqrt(SigmaXY ** 2 - .5 * Vrot ** 2)
        SigmaZ = np.sqrt(np.average(vheis ** 2, weights=particles[:, 3]))
        vrotsig = Vrot / SigmaO
        delta = 1 - (SigmaZ / SigmaO) ** 2
        # Return
        return kappa, discfrac, orbi, vrotsig, delta, zaxis, Momentum
    
    
    @staticmethod
    def cumsummedian(a, weights=None):
        """
        Compute the weighted median.

        Returns the median of the array elements.

        Parameters
        ----------
        a : array_like, shape (n, )
            Input array or object that can be converted to an array.
        weights : {array_like, shape (n, ), None}, optional
            Input array or object that can be converted to an array.

        Returns
        -------
        median : float

        """
        if weights is None:
            weights = np.ones(np.array(a).shape)
        A = np.array(a).astype('float')
        W = np.array(weights).astype('float')
        if not (np.product(np.isnan(A))):
            I = np.argsort(A)
            cumweight = np.hstack([0, np.cumsum(W[I])])
            X = np.hstack([0, (cumweight[:-1] + cumweight[1:]) / (2 * cumweight[-1]), 1])
            Y = np.hstack([np.min(A), A[I], np.max(A)])
            P = interpolate.interp1d(X, Y)(0.5)
            return float(P)
        else:
            return np.nan
    
    
    @staticmethod
    def plot(stellar_data_tmp, glx_unit_vector, unit_vector, group_number, subgroup_number):
        """
        A method to plot a hexbin histogram.
        :param stellar_data_tmp: from mask_galaxies
        :param glx_unit_vector: from mask_galaxies
        :param unit_vector: from mask_galaxies
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
        figure = plt.figure(0, figsize=(20, 15))
        
        gs = gridspec.GridSpec(2, 2)
        axupperleft = plt.subplot(gs[0, 0], projection="mollweide")
        axlowerleft = plt.subplot(gs[1, 0], projection="mollweide")
        axupperright = plt.subplot(gs[0, 1], projection="mollweide")
        axlowerright = plt.subplot(gs[1, 1], projection="mollweide")
        
        axupperleft.grid(True, color='black')
        axlowerleft.grid(True, color='black')
        axupperright.grid(True, color='black')
        axlowerright.grid(True, color='black')
        axupperleft.set_xlabel('RA ($\degree$)')
        axlowerleft.set_xlabel('RA ($\degree$)')
        axupperright.set_xlabel('RA ($\degree$)')
        axlowerright.set_xlabel('RA ($\degree$)')
        axupperleft.set_ylabel('Dec ($\degree$)')
        axlowerleft.set_ylabel('Dec ($\degree$)')
        axupperright.set_ylabel('Dec ($\degree$)')
        axlowerright.set_ylabel('Dec ($\degree$)')
        
        y_tick_labels = np.array(['', '-60', '', '-30', '', 0, '', '30', '', 60])
        x_tick_labels = np.array(['', '-120', '', '-60', '', 0, '', '60', '', 120])
        axupperleft.set_xticklabels(x_tick_labels)
        axupperleft.set_yticklabels(y_tick_labels)
        axlowerleft.set_xticklabels(x_tick_labels)
        axlowerleft.set_yticklabels(y_tick_labels)
        axlowerright.set_xticklabels(x_tick_labels)
        axlowerright.set_yticklabels(y_tick_labels)
        axupperright.set_xticklabels(x_tick_labels)
        axupperright.set_yticklabels(y_tick_labels)
        
        # Generate the RA and Dec projection #
        hexbin = axupperleft.hexbin(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]), bins='log', cmap='PuRd',
                                    gridsize=100, edgecolor='none', mincnt=1, zorder=-1)  # Element-wise arctan of x1/x2.
        axupperleft.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                            zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(hexbin, ax=axupperleft, orientation='horizontal')
        cbar.set_label('$\mathrm{Particles\; per\; hexbin}$')
        
        # Generate the RA and Dec projection colour-coded by StellarFormationTime #
        scatter = axupperright.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]),
                                       c=stellar_data_tmp['StellarFormationTime'], cmap='jet_r', s=1, zorder=-1)
        
        axupperright.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                             zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axupperright, orientation='horizontal')
        cbar.set_label('$\mathrm{StellarFormationTime}$')
        
        # Generate the RA and Dec projection colour-coded by e^beta-1 #
        velocity_r_sqred = np.divide(np.sum(np.multiply(stellar_data_tmp['Velocity'], stellar_data_tmp['Coordinates']), axis=1) ** 2,
                                     np.sum(np.multiply(stellar_data_tmp['Coordinates'], stellar_data_tmp['Coordinates']), axis=1))
        beta = np.subtract(1, np.divide(
            np.subtract(np.sum(np.multiply(stellar_data_tmp['Velocity'], stellar_data_tmp['Velocity']), axis=1), velocity_r_sqred),
            2 * velocity_r_sqred))
        
        scatter = axlowerleft.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]),
                                      c=np.log10(np.divide(beta, np.mean(beta))), cmap='tab20', s=1, zorder=-1)
        
        axlowerleft.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                            zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axlowerleft, orientation='horizontal')
        cbar.set_label(r'$\mathrm{log_{10}({\beta/\bar{\beta}})}$')
        
        # Generate the RA and Dec projection colour-coded by BirthDensity #
        scatter = axlowerright.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]),
                                       c=stellar_data_tmp['BirthDensity'], cmap='jet', s=1, zorder=-1)
        axlowerright.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                             zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axlowerright, orientation='horizontal')
        cbar.set_label('$\mathrm{BirthDensity}$')
        
        # Save the plot #
        # plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + str(group_number) + str(subgroup_number) + '-' + 'RD' + '-' + date + '.png', bbox_inches='tight')
        
        kappa, discfrac, orbi, vrotsig, delta, zaxis, Momentum = RADec.kinematics_diagnostics(stellar_data_tmp['Coordinates'],
                                                                                              stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'],
                                                                                              stellar_data_tmp['ParticleBindingEnergy'],
                                                                                              aperture=0.03, CoMvelocity=False)
        return None


if __name__ == '__main__':
    # tag = '010_z005p000'
    # sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    # outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RD/G-EAGLE/'  # Path to save plots.
    # SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/RD/G-EAGLE/'  # Path to save data.
    tag = '027_z000p101'
    sim = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
    outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RD/EAGLE/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/RD/EAGLE/'  # Path to save data.
    x = RADec(sim, tag)