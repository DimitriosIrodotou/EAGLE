import re
import sys
import time
import h5py

import numpy as np
import astropy.units as u
import eagle_IO.eagle_IO.eagle_IO as E

from scipy.special import gamma
from astropy_healpix import HEALPix
from scipy.optimize import curve_fit
from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematic

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.


class ReadAttributes:
    """
    For each galaxy: read in and save its attribute(s).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        p = 1  # Counter.
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.box_data, self.stellar_data, self.gaseous_data, self.dark_matter_data, self.subhalo_data, self.FOF_data = self.read_attributes(
            simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 5e9 Msun.
        
        job_number = int(sys.argv[2]) - 1
        group_numbers = np.array_split(list(self.subhalo_data_tmp['GroupNumber']), 30)
        subgroup_numbers = np.array_split(list(self.subhalo_data_tmp['SubGroupNumber']), 30)
        
        for group_number, subgroup_number in zip(group_numbers[job_number],
                                                 subgroup_numbers[job_number]):  # Loop over all masked haloes and sub-haloes.
            start_local_time = time.time()  # Start the local time.
            
            # Mask galaxies and normalise data #
            stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp, subhalo_data_tmp = self.mask_galaxies(group_number, subgroup_number)
            
            # Save data in numpy array #
            np.save(data_path + 'box_data', self.box_data)
            np.save(data_path + 'FOF_data_tmps/FOF_data_tmp_' + str(group_number), self.FOF_data)
            np.save(data_path + 'subhalo_data_tmps/subhalo_data_tmp_' + str(group_number) + '_' + str(subgroup_number), subhalo_data_tmp)
            np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
            np.save(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number), gaseous_data_tmp)
            np.save(data_path + 'dark_matter_data_tmps/dark_matter_data_tmp_' + str(group_number) + '_' + str(subgroup_number), dark_matter_data_tmp)
            
            print('Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            p += 1
        
        print('Finished ReadAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def read_attributes(self, simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        :return: stellar_data, gaseous_data, dark_matter_data, subhalo_data, FOF_data
        """
        
        # Load particle data in h-free physical CGS units #
        box_data, stellar_data, gaseous_data, dark_matter_data = {}, {}, {}, {}
        
        file_type = 'SUBFIND'
        for attribute in ['HubbleParam', 'BoxSize']:
            box_data[attribute] = E.read_header(file_type, simulation_path, tag, attribute)
        
        file_type = 'PARTDATA'
        particle_type = '4'
        for attribute in ['BirthDensity', 'Coordinates', 'GroupNumber', 'InitialMass', 'Mass', 'Metallicity', 'ParticleBindingEnergy', 'ParticleIDs',
                          'StellarFormationTime', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        particle_type = '0'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'ParticleIDs', 'StarFormationRate', 'SubGroupNumber', 'Velocity']:
            gaseous_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        particle_type = '1'
        for attribute in ['Coordinates', 'GroupNumber', 'ParticleIDs', 'SubGroupNumber', 'Velocity']:
            dark_matter_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'IDMostBound', 'InitialMassWeightedStellarAge',
                          'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load FOF data in h-free physical CGS units #
        FOF_data = {}
        file_type = 'SUBFIND'
        for attribute in ['Group_M_Crit200', 'Group_R_Crit200', 'FirstSubhaloID']:
            FOF_data[attribute] = E.read_array(file_type, simulation_path, tag, '/FOF/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['InitialMass'] *= u.g.to(u.Msun)
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        stellar_data['BirthDensity'] *= u.g.to(u.Msun) / u.cm.to(u.kpc) ** 3
        
        gaseous_data['Mass'] *= u.g.to(u.Msun)
        gaseous_data['Velocity'] *= u.cm.to(u.km)  # per second.
        gaseous_data['Coordinates'] *= u.cm.to(u.kpc)
        gaseous_data['StarFormationRate'] *= u.g.to(u.Msun) / u.second.to(u.year)
        
        dark_matter_data['Coordinates'] *= u.cm.to(u.kpc)
        dark_matter_data['Velocity'] *= u.cm.to(u.km)  # per second.
        dark_matter_data['Mass'] = self.dark_matter_mass(simulation_path, tag) * u.g.to(u.Msun)
        
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        FOF_data['Group_M_Crit200'] *= u.g.to(u.Msun)
        FOF_data['Group_R_Crit200'] *= u.cm.to(u.kpc)
        
        return box_data, stellar_data, gaseous_data, dark_matter_data, subhalo_data, FOF_data
    
    
    def mask_haloes(self):
        """
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 5e9 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 5e9)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: stellar_data_tmp, glx_unit_vector
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        halo_mask, = np.where((self.subhalo_data_tmp['GroupNumber'] == group_number) & (self.subhalo_data_tmp['SubGroupNumber'] == subgroup_number))
        
        # Periodically wrap coordinates around centre #
        box_side = self.box_data['BoxSize'] * 1e3 / self.box_data['HubbleParam']
        self.stellar_data['Coordinates'] = np.mod(
            self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask] + 0.5 * box_side, box_side) + \
                                           self.subhalo_data_tmp['CentreOfPotential'][halo_mask] - 0.5 * box_side
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        stellar_mask = np.where(
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask], axis=1) <= 30.0)
        
        gaseous_mask = np.where(
            np.linalg.norm(self.gaseous_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask], axis=1) <= 30.0)
        
        dark_matter_mask = np.where(
            np.linalg.norm(self.dark_matter_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask], axis=1) <= 30.0)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp, stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp = {}, {}, {}, {}
        
        for attribute in self.subhalo_data_tmp.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data_tmp[attribute])[halo_mask]
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[stellar_mask]
        for attribute in self.gaseous_data.keys():
            gaseous_data_tmp[attribute] = np.copy(self.gaseous_data[attribute])[gaseous_mask]
        for attribute in self.dark_matter_data.keys():
            dark_matter_data_tmp[attribute] = np.copy(self.dark_matter_data[attribute])[dark_matter_mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        for data in [stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp]:
            data['Coordinates'] = data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask]
        
        prc_masses = np.hstack([stellar_data_tmp['Mass'], gaseous_data_tmp['Mass'], dark_matter_data_tmp['Mass']])
        prc_velocities = np.vstack([stellar_data_tmp['Velocity'], gaseous_data_tmp['Velocity'], dark_matter_data_tmp['Velocity']])
        CoM_velocity = np.divide(np.sum(prc_masses[:, np.newaxis] * prc_velocities, axis=0), np.sum(prc_masses, axis=0))  # In km s^-1.
        # CoM_velocity2 = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0) + np.sum(
        #     gaseous_data_tmp['Mass'][:, np.newaxis] * gaseous_data_tmp['Velocity'], axis=0) + np.sum(
        #     dark_matter_data_tmp['Mass'][:, np.newaxis] * dark_matter_data_tmp['Velocity'], axis=0),
        #                           np.sum(stellar_data_tmp['Mass'], axis=0) + np.sum(gaseous_data_tmp['Mass'], axis=0) + np.sum(
        #                               dark_matter_data_tmp['Mass'], axis=0))  # In km s^-1.
        
        for data in [stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp]:
            data['Velocity'] = data['Velocity'] - CoM_velocity
        
        return stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp, subhalo_data_tmp
    
    
    @staticmethod
    def dark_matter_mass(simulation_path, tag):
        """
        Create a mass array for dark matter particles. As all dark matter particles share the same mass, there exists no PartType1/Mass dataset in
        the snapshot files.
        :return: particle_mass
        """
        
        # Read the required attributes from the header and get conversion factors from gaseous particles #
        f = h5py.File(simulation_path + 'snapshot_' + tag + '/snap_027_z000p101.0.hdf5', 'r')
        a = f['Header'].attrs.get('Time')
        h = f['Header'].attrs.get('HubbleParam')
        n_particles = f['Header'].attrs.get('NumPart_Total')[1]
        dark_matter_mass = f['Header'].attrs.get('MassTable')[1]
        
        cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
        aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
        hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
        f.close()
        
        # Create an array of length equal to the number of dark matter particles and convert to h-free physical CGS units #
        particle_masses = np.ones(n_particles, dtype='f8') * dark_matter_mass
        particle_masses *= cgs * a ** aexp * h ** hexp
        
        return particle_masses


class AddAttributes:
    """
    For each galaxy: load its stellar_data_tmp dictionary and add the new attribute(s).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.subhalo_data = self.read_attributes(simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 5e9 Msun.
        
        job_number = int(sys.argv[2]) - 1
        group_numbers = np.array_split(list(self.subhalo_data_tmp['GroupNumber']), 30)
        subgroup_numbers = np.array_split(list(self.subhalo_data_tmp['SubGroupNumber']), 30)
        
        for group_number, subgroup_number in zip(group_numbers[job_number],
                                                 subgroup_numbers[job_number]):  # Loop over all masked haloes and sub-haloes.
            start_local_time = time.time()  # Start the local time.
            
            # Load data #
            stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            stellar_data_tmp = stellar_data_tmp.item()
            
            gaseous_data_tmp = np.load(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            gaseous_data_tmp = gaseous_data_tmp.item()
            
            dark_matter_data_tmp = np.load(
                data_path + 'dark_matter_data_tmps/dark_matter_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                allow_pickle=True)
            dark_matter_data_tmp = dark_matter_data_tmp.item()
            
            # Add attributes #
            stellar_data_tmp['c'] = self.concentration_index(stellar_data_tmp)
            stellar_data_tmp['kappa_corotation'] = self.kappa_corotation(stellar_data_tmp)
            stellar_data_tmp['velocity_sqred'], stellar_data_tmp['velocity_r_sqred'] = self.beta_components(stellar_data_tmp)
            stellar_data_tmp['disc_mask_IT20'], stellar_data_tmp['bulge_mask_IT20'], stellar_data_tmp['delta_theta'] = self.decomposition_IT20(
                stellar_data_tmp)
            stellar_data_tmp['disc_fraction'], stellar_data_tmp['rotational_over_dispersion'], stellar_data_tmp['rotational_velocity'], \
            stellar_data_tmp['sigma_0'], stellar_data_tmp['delta'] = self.kinematic_diagnostics(stellar_data_tmp)
            stellar_data_tmp['n'], stellar_data_tmp['R_d'], stellar_data_tmp['R_eff'], stellar_data_tmp['disk_fraction_profile'], stellar_data_tmp[
                'fitting_flag'] = self.profile_fitting(stellar_data_tmp)
            stellar_data_tmp['delta_r'] = self.delta_r(stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp)
            
            gaseous_data_tmp['star_forming_mask'] = np.where(gaseous_data_tmp['StarFormationRate'] > 0.0)
            gaseous_data_tmp['non_star_forming_mask'] = np.where(gaseous_data_tmp['StarFormationRate'] == 0.0)
            
            # Save data in numpy array #
            np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
            np.save(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number), gaseous_data_tmp)
            
            print(
                'Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished AddAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        :return: subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return subhalo_data
    
    
    def mask_haloes(self):
        """
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 5e9 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 5e9)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp
    
    
    @staticmethod
    def decomposition_IT20(stellar_data_tmp):
        """
        Find the particles that belong to the disc and bulge based on the IT20 method.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: disc_mask_IT20, bulge_mask_IT20
        """
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        glx_stellar_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        glx_unit_vector = glx_stellar_angular_momentum / np.linalg.norm(glx_stellar_angular_momentum)
        
        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        coordinates, velocity, prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(stellar_data_tmp, glx_unit_vector)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Plot a HEALPix histogram #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
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
        disc_mask_IT20, = np.where(angular_theta_from_densest < (np.pi / 6.0))
        bulge_mask_IT20, = np.where(angular_theta_from_densest > (np.pi / 6.0))
        
        # Calculate the angle between the densest pixel and the angular momentum vector of the galaxy #
        position_of_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T
        angular_theta_from_X = np.arccos(
            np.sin(position_of_X[0, 1]) * np.sin(lat_densest) + np.cos(position_of_X[0, 1]) * np.cos(lat_densest) * np.cos(
                position_of_X[0, 0] - lon_densest))  # In radians.
        delta_theta = np.degrees(angular_theta_from_X)  # In degrees.
        
        return disc_mask_IT20, bulge_mask_IT20, delta_theta
    
    
    @staticmethod
    def concentration_index(stellar_data_tmp):
        """
        Calculate the concentration index R90/R50 or 5*log10(R80/R20).
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: c
        """
        
        c = MorphoKinematic.r_mass(stellar_data_tmp, 0.9) / MorphoKinematic.r_mass(stellar_data_tmp, 0.5)
        # c = 5 * np.log10(MorphoKinematic.r_mass(stellar_data_tmp, 0.8) / MorphoKinematic.r_mass(stellar_data_tmp, 0.2))
        return c
    
    
    @staticmethod
    def kappa_corotation(stellar_data_tmp):
        """
        Calculate the fraction of a particle's kinetic energy that's invested in corotation.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: kappa_corotation
        """
        
        # Rotate the galaxy and calculate the unit vector pointing along the glx_stellar_angular_momentum direction.
        coordinates, velocity, prc_angular_momentum, glx_stellar_angular_momentum = RotateCoordinates.rotate_Jz(stellar_data_tmp)
        prc_spc_angular_momentum = np.cross(coordinates, velocity)  # In kpc km s^-1.
        glx_unit_vector = glx_stellar_angular_momentum / np.linalg.norm(glx_stellar_angular_momentum)
        spc_angular_momentum_z = np.sum(glx_unit_vector * prc_spc_angular_momentum, axis=1)  # In kpc km s^-1.
        prc_cylindrical_distance = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)  # In kpc.
        
        corotation_mask, = np.where(spc_angular_momentum_z > 0)
        specific_angular_velocity = spc_angular_momentum_z[corotation_mask] / prc_cylindrical_distance[corotation_mask]  # In km s^-1.
        kinetic_energy = np.sum(stellar_data_tmp['Mass'] * np.linalg.norm(velocity, axis=1) ** 2)  # In Msun km^2 s^-2.
        angular_kinetic_energy = np.sum(stellar_data_tmp['Mass'][corotation_mask] * specific_angular_velocity ** 2)  # In Msun km^2 s^-2.
        kappa_corotation = angular_kinetic_energy / kinetic_energy
        
        return kappa_corotation
    
    
    @staticmethod
    def kinematic_diagnostics(stellar_data_tmp):
        """
        Calculate the disc fraction and the rotational over dispersion velocity ratio.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: disc_fraction, rotational_over_dispersion, rotational_velocity, sigma_0, delta
        """
        
        kappa, disc_fraction, circularity, rotational_over_dispersion, vrots, rotational_velocity, sigma_0, \
        delta = MorphoKinematic.kinematic_diagnostics(
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'], stellar_data_tmp['ParticleBindingEnergy'])
        
        return disc_fraction, rotational_over_dispersion, rotational_velocity, sigma_0, delta
    
    
    @staticmethod
    def beta_components(stellar_data_tmp):
        """
        Calculate the components of the anisotropy parameter beta.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: velocity_sqred, velocity_r_sqred
        """
        
        # Calculate u^2 and u_r^2 #
        velocity_sqred = np.sum(stellar_data_tmp['Velocity'] * stellar_data_tmp['Velocity'], axis=1)
        velocity_r_sqred = np.divide(np.sum(stellar_data_tmp['Velocity'] * stellar_data_tmp['Coordinates'], axis=1) ** 2,
                                     np.sum(stellar_data_tmp['Coordinates'] * stellar_data_tmp['Coordinates'], axis=1))
        
        return velocity_sqred, velocity_r_sqred
    
    
    @staticmethod
    def profile_fitting(stellar_data_tmp):
        """
        Calculate the Sersic index.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: n, R_d, R_eff, disk_fraction_profile
        """
        
        
        # Declare the Sersic, exponential and total profiles and Sersic parameter b #
        def sersic_profile(r, I_0b, b, n):
            """
            Calculate a Sersic profile.
            :param r: radius.
            :param I_0b: Bulge central intensity.
            :param b: Sersic b parameter
            :param n: Sersic index
            :return: I_0b * np.exp(-(r / b) ** (1 / n))
            """
            return I_0b * np.exp(-(r / b) ** (1 / n))  # b = R_eff / b_n ^ n
        
        
        def exponential_profile(r, I_0d, R_d):
            """
            Calculate an exponential profile.
            :param r: radius
            :param I_0d: Disc central intensity.
            :param R_d: Disc scale length.
            :return: I_0d * np.exp(-r / R_d)
            """
            return I_0d * np.exp(-r / R_d)
        
        
        def total_profile(r, I_0d, R_d, I_0b, b, n):
            """
            Calculate a total (Sersic + exponential) profile.
            :param r: radius.
            :param I_0d: Disc central intensity.
            :param R_d: Disc scale length.
            :param I_0b: Bulge central intensity.
            :param b: Sersic b parameter.
            :param n: Sersic index.
            :return: exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
            """
            y = exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
            return y
        
        
        def sersic_b_n(n):
            """
            Calculate the Sersic b parameter.
            :param n: Sersic index.
            :return: b_n
            """
            if n <= 0.36:
                b_n = 0.01945 + n * (- 0.8902 + n * (10.95 + n * (- 19.67 + n * 13.43)))
            else:
                x = 1.0 / n
                b_n = -1.0 / 3.0 + 2. * n + x * (4.0 / 405. + x * (46. / 25515. + x * (131. / 1148175 - x * 2194697. / 30690717750.)))
            return b_n
        
        
        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        coordinates, velocity, prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(stellar_data_tmp)
        
        cylindrical_distance = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)  # Radius of each particle.
        vertical_mask, = np.where(abs(coordinates[:, 2]) < 5)  # Vertical cut in kpc.
        
        mass, edges = np.histogram(cylindrical_distance[vertical_mask], bins=50, range=(0, 30), weights=stellar_data_tmp['Mass'][vertical_mask])
        centers = 0.5 * (edges[1:] + edges[:-1])
        surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sden = mass / surface
        
        try:
            fitting_flag = 1
            popt, pcov = curve_fit(total_profile, centers, sden, sigma=0.1 * sden, p0=[sden[0], 2, sden[0], 2, 4])  # p0 = [I_0d, R_d, I_0b, b, n]
            
            # Calculate galactic attributes #
            I_0d, R_d, I_0b, b, n = popt[0], popt[1], popt[2], popt[3], popt[4]
            R_eff = b * sersic_b_n(n) ** n
            disk_mass = 2.0 * np.pi * I_0d * R_d ** 2
            bulge_mass = np.pi * I_0b * R_eff ** 2 * gamma(2.0 / n + 1)
            disk_fraction_profile = np.divide(disk_mass, bulge_mass + disk_mass)
        
        except RuntimeError:
            fitting_flag = 0
            n, R_d, R_eff, disk_fraction_profile = 0, 0, 0, 0
            print('Could not fit a Sersic+exponential profile')
        
        # If a Sersic+exponential fit fails try fitting a single Sersic profile #
        if fitting_flag == 0:
            try:
                popt, pcov = curve_fit(sersic_profile, centers, sden, sigma=0.1 * sden, p0=[sden[0], 2, 4])  # p0 = [I_0b, b, n]
                
                # Calculate bulge attributes #
                I_0b, b, n = popt[0], popt[1], popt[2]
                R_eff = b * sersic_b_n(n) ** n
            except RuntimeError:
                print('Could not fit neither a Sersic+exponential nor a Sersic profile')
        
        return n, R_d, R_eff, disk_fraction_profile, fitting_flag
    
    
    @staticmethod
    def delta_r(stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp):
        """
        Calculate the distance between the centre of mass and the centre of potential (accounted for in mask_galaxies) normalised wrt the half-mass
        radius.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param gaseous_data_tmp: from read_add_attributes.py.
        :param dark_matter_data_tmp: from read_add_attributes.py.
        :return: delta_r
        """
        prc_masses = np.hstack([stellar_data_tmp['Mass'], gaseous_data_tmp['Mass'], dark_matter_data_tmp['Mass']])
        prc_coordinates = np.vstack([stellar_data_tmp['Coordinates'], gaseous_data_tmp['Coordinates'], dark_matter_data_tmp['Coordinates']])
        CoM = np.divide(np.sum(prc_masses[:, np.newaxis] * prc_coordinates, axis=0), np.sum(prc_masses, axis=0))
        
        delta_r = np.linalg.norm(CoM) / MorphoKinematic.r_mass(stellar_data_tmp, 0.5)
        
        return delta_r


class AppendAttributes:
    """
    For each galaxy: load its stellar_data_tmp dictionary and append the attribute(s).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        # Initialise arrays to store the data #
        group_numbers, subgroup_numbers, CoPs = [], [], []
        
        glx_stellar_angular_momenta, glx_stellar_masses, glx_concentration_indices, glx_kappas_corotation, glx_disc_fractions_IT20, \
        glx_disc_fractions, glx_rotationals_over_dispersions, glx_Sersic_indices, glx_scale_lengths, glx_effective_radii, \
        glx_disk_fraction_profiles, glx_n_particles, glx_fitting_flags, glx_delta_thetas, glx_delta_rs, glx_deltas, glx_sigma_0s, glx_rotationals, \
        glx_as = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        glx_gaseous_angular_momenta, glx_gaseous_masses, glx_star_formings, glx_non_star_formings, glx_star_formation_rates = [], [], [], [], []
        
        dark_matter_masses = []
        
        disc_metallicities, bulge_metallicities, disc_betas, bulge_betas, disc_rotationals, bulge_rotationals, disc_sigma_0s, bulge_sigma_0s, \
        disc_birth_densities, bulge_birth_densities, disc_deltas, bulge_deltas, disc_as, bulge_as,disc_stellar_angular_momenta,bulge_stellar_angular_momenta = [], [], [], [], [], [], [], [], [], [], [], [],\
                                                                                                    [], [],[],[]
        
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.subhalo_data = self.read_attributes(simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 5e9 Msun.
        for group_number, subgroup_number in zip(list(self.subhalo_data_tmp['GroupNumber']),
                                                 list(self.subhalo_data_tmp['SubGroupNumber'])):  # Loop over all masked haloes and sub-haloes.
            start_local_time = time.time()  # Start the local time.
            
            # Load data #
            # subhalo_data_tmp = np.load(data_path + 'subhalo_data_tmps/subhalo_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            #                            allow_pickle=True)
            # subhalo_data_tmp = subhalo_data_tmp.item()
            
            stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            stellar_data_tmp = stellar_data_tmp.item()
            
            # gaseous_data_tmp = np.load(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            #                            allow_pickle=True)
            # gaseous_data_tmp = gaseous_data_tmp.item()
            
            # dark_matter_data_tmp = np.load(
            #     data_path + 'dark_matter_data_tmps/dark_matter_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            #     allow_pickle=True)
            # dark_matter_data_tmp = dark_matter_data_tmp.item()
            
            # Calculate galactic attributes #
            prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                      stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
            # glx_stellar_angular_momentum = np.sum(prc_angular_momentum, axis=0)
            
            # prc_angular_momentum = gaseous_data_tmp['Mass'][:, np.newaxis] * np.cross(gaseous_data_tmp['Coordinates'],
            #                                                                           gaseous_data_tmp['Velocity'])  # In Msun kpc km s^-1.
            
            # disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][stellar_data_tmp['disc_mask_IT20']]) / np.sum(stellar_data_tmp['Mass'])
            
            # glx_gaseous_angular_momentum = np.sum(prc_angular_momentum, axis=0)
            
            # Calculate component attributes #
            for i, mask in enumerate([stellar_data_tmp['disc_mask_IT20'], stellar_data_tmp['bulge_mask_IT20']]):
                component_stellar_angular_momentum = np.sum(prc_angular_momentum[mask], axis=0)
                # component_mass = np.sum(stellar_data_tmp['Mass'][mask])
                # metals = np.divide(stellar_data_tmp['Metallicity'][mask] * stellar_data_tmp['Mass'][mask], component_mass)
                # velocity_sqred = stellar_data_tmp['velocity_sqred'][mask]
                # velocity_r_sqred = stellar_data_tmp['velocity_r_sqred'][mask]
                # kappa, disc_fraction, circularity, rotational_over_dispersion, vrots, rotational_velocity, sigma_0, \
                # delta = MorphoKinematic.kinematic_diagnostics(
                #     stellar_data_tmp['Coordinates'][mask], stellar_data_tmp['Mass'][mask], stellar_data_tmp['Velocity'][mask],
                #     stellar_data_tmp['ParticleBindingEnergy'][mask])
                # birth_density = stellar_data_tmp['BirthDensity'][mask]
                # birth_stellar_formation_time = stellar_data_tmp['StellarFormationTime'][mask]
                # birth_mass = stellar_data_tmp['InitialMass'][mask]
                
                if i == 0:
                    # stellar_data_tmp['disc_beta'] = 1 - np.divide(np.mean(velocity_sqred) - np.mean(velocity_r_sqred), 2 * np.mean(velocity_r_sqred))
                    # stellar_data_tmp['disc_metallicity'] = np.sum(metals) / 0.0134  # In solar metallicity.
                    # stellar_data_tmp['disc_rotational'] = rotational_velocity  # In km s^-1.
                    # stellar_data_tmp['disc_sigma_0'] = sigma_0  # In km s^-1.
                    # stellar_data_tmp['disc_delta'] = delta  # In km s^-1.
                    # stellar_data_tmp['disc_birth_density'] = birth_density  # In Msun kpc^-3.
                    # stellar_data_tmp['disc_a'] = np.mean(birth_stellar_formation_time)  # Expansion factor at birth.
                    # stellar_data_tmp['disc_a'] = np.average(birth_stellar_formation_time, weights=birth_mass,axis=0)  # Expansion factor at birth.
                    stellar_data_tmp['disc_stellar_angular_momentum'] = component_stellar_angular_momentum  # In Msun kpc km s^-1.
                else:
                    # stellar_data_tmp['bulge_beta'] = 1 - np.divide(np.mean(velocity_sqred) - np.mean(velocity_r_sqred), 2 * np.mean(velocity_r_sqred))
                    # stellar_data_tmp['bulge_metallicity'] = np.sum(metals) / 0.0134  # In solar metallicity.
                    # stellar_data_tmp['bulge_rotational'] = rotational_velocity  # In km s^-1.
                    # stellar_data_tmp['bulge_sigma_0'] = sigma_0  # In km s^-1.
                    # stellar_data_tmp['bulge_delta'] = delta  # In km s^-1.
                    # stellar_data_tmp['bulge_birth_density'] = birth_density  # In Msun kpc^-3.
                    # stellar_data_tmp['bulge_a'] = np.mean(birth_stellar_formation_time)  # Expansion factor at birth.
                    # stellar_data_tmp['bulge_a'] = np.average(birth_stellar_formation_time, weights=birth_mass,axis=0)  # Expansion factor at birth.
                    stellar_data_tmp['bulge_stellar_angular_momentum'] = component_stellar_angular_momentum  # In Msun kpc km s^-1.
            
            # Append halo attributes into single arrays #
            # group_numbers.append(group_number)
            # subgroup_numbers.append(subgroup_number)
            # CoPs.append(subhalo_data_tmp['CentreOfPotential'])
            
            # Append galactic attributes into single arrays #
            # glx_deltas.append(stellar_data_tmp['delta'])
            # glx_delta_rs.append(stellar_data_tmp['delta_r'])
            # glx_sigma_0s.append(stellar_data_tmp['sigma_0'])
            # glx_Sersic_indices.append(stellar_data_tmp['n'])
            # glx_scale_lengths.append(stellar_data_tmp['R_d'])
            # glx_disc_fractions_IT20.append(disc_fraction_IT20)
            # glx_n_particles.append(len(stellar_data_tmp['Mass']))
            # glx_effective_radii.append(stellar_data_tmp['R_eff'])
            # glx_as.append(np.mean(stellar_data_tmp['StellarFormationTime']))
            # glx_concentration_indices.append(stellar_data_tmp['c'])
            # glx_delta_thetas.append(stellar_data_tmp['delta_theta'])
            # glx_fitting_flags.append(stellar_data_tmp['fitting_flag'])
            # glx_stellar_masses.append(np.sum(stellar_data_tmp['Mass']))
            # glx_disc_fractions.append(stellar_data_tmp['disc_fraction'])
            # glx_rotationals.append(stellar_data_tmp['rotational_velocity'])
            # glx_stellar_angular_momenta.append(glx_stellar_angular_momentum)
            # glx_kappas_corotation.append(stellar_data_tmp['kappa_corotation'])
            # glx_disk_fraction_profiles.append(stellar_data_tmp['disk_fraction_profile'])
            # glx_rotationals_over_dispersions.append(stellar_data_tmp['rotational_over_dispersion'])
            
            # glx_gaseous_masses.append(np.sum(gaseous_data_tmp['Mass']))
            # glx_gaseous_angular_momenta.append(glx_gaseous_angular_momentum)
            # glx_star_formation_rates.append(np.sum(gaseous_data_tmp['StarFormationRate']))
            # glx_star_formings.append(np.sum(gaseous_data_tmp['Mass'][gaseous_data_tmp['star_forming_mask']]))
            # glx_non_star_formings.append(np.sum(gaseous_data_tmp['Mass'][gaseous_data_tmp['non_star_forming_mask']]))
            
            # dark_matter_masses.append(np.sum(dark_matter_data_tmp['Mass']))
            
            # Append component attributes into single arrays #
            # disc_as.append(stellar_data_tmp['disc_a'])
            disc_stellar_angular_momenta.append(stellar_data_tmp['disc_stellar_angular_momentum'])
            bulge_stellar_angular_momenta.append(stellar_data_tmp['bulge_stellar_angular_momentum'])
            # bulge_as.append(stellar_data_tmp['bulge_a'])
            # disc_betas.append(stellar_data_tmp['disc_beta'])
            # bulge_betas.append(stellar_data_tmp['bulge_beta'])
            # disc_deltas.append(stellar_data_tmp['disc_delta'])
            # bulge_deltas.append(stellar_data_tmp['bulge_delta'])
            # disc_sigma_0s.append(stellar_data_tmp['disc_sigma_0'])
            # bulge_sigma_0s.append(stellar_data_tmp['bulge_sigma_0'])
            # disc_rotationals.append(stellar_data_tmp['disc_rotational'])
            # bulge_rotationals.append(stellar_data_tmp['bulge_rotational'])
            # disc_metallicities.append(stellar_data_tmp['disc_metallicity'])
            # bulge_metallicities.append(stellar_data_tmp['bulge_metallicity'])
            # disc_birth_densities.append(stellar_data_tmp['disc_birth_density'])
            # bulge_birth_densities.append(stellar_data_tmp['bulge_birth_density'])
            
            print(
                'Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Save data in numpy array #
        # np.save(data_path + 'CoPs', CoPs)
        # np.save(data_path + 'group_numbers', group_numbers)
        # np.save(data_path + 'subgroup_numbers', subgroup_numbers)
        
        # np.save(data_path + 'glx_as', glx_as)
        # np.save(data_path + 'glx_deltas', glx_deltas)
        # np.save(data_path + 'glx_delta_rs', glx_delta_rs)
        # np.save(data_path + 'glx_sigma_0s', glx_sigma_0s)
        # np.save(data_path + 'glx_rotationals', glx_rotationals)
        # np.save(data_path + 'glx_n_particles', glx_n_particles)
        # np.save(data_path + 'glx_delta_thetas', glx_delta_thetas)
        # np.save(data_path + 'glx_fitting_flags', glx_fitting_flags)
        # np.save(data_path + 'glx_scale_lengths', glx_scale_lengths)
        # np.save(data_path + 'glx_stellar_masses', glx_stellar_masses)
        # np.save(data_path + 'glx_disc_fractions', glx_disc_fractions)
        # np.save(data_path + 'glx_Sersic_indices', glx_Sersic_indices)
        # np.save(data_path + 'glx_effective_radii', glx_effective_radii)
        # np.save(data_path + 'glx_kappas_corotation', glx_kappas_corotation)
        # np.save(data_path + 'glx_disc_fractions_IT20', glx_disc_fractions_IT20)
        # np.save(data_path + 'glx_concentration_indices', glx_concentration_indices)
        # np.save(data_path + 'glx_disk_fraction_profiles', glx_disk_fraction_profiles)
        # np.save(data_path + 'glx_stellar_angular_momenta', glx_stellar_angular_momenta)
        # np.save(data_path + 'glx_rotationals_over_dispersions', glx_rotationals_over_dispersions)
        #
        # np.save(data_path + 'glx_star_forming', glx_star_formings)
        # np.save(data_path + 'glx_gaseous_masses', glx_gaseous_masses)
        # np.save(data_path + 'glx_non_star_forming', glx_non_star_formings)
        # np.save(data_path + 'glx_star_formation_rates', glx_star_formation_rates)
        # np.save(data_path + 'glx_gaseous_angular_momenta', glx_gaseous_angular_momenta)
        #
        # np.save(data_path + 'dark_matter_masses', dark_matter_masses)
        
        # np.save(data_path + 'disc_as', disc_as)
        # np.save(data_path + 'bulge_as', bulge_as)
        # np.save(data_path + 'disc_betas', disc_betas)
        # np.save(data_path + 'bulge_betas', bulge_betas)
        # np.save(data_path + 'disc_deltas', disc_deltas)
        # np.save(data_path + 'bulge_deltas', bulge_deltas)
        # np.save(data_path + 'disc_sigma_0s', disc_sigma_0s)
        # np.save(data_path + 'bulge_sigma_0s', bulge_sigma_0s)
        # np.save(data_path + 'disc_rotationals', disc_rotationals)
        # np.save(data_path + 'bulge_rotationals', bulge_rotationals)
        # np.save(data_path + 'disc_metallicities', disc_metallicities)
        # np.save(data_path + 'bulge_metallicities', bulge_metallicities)
        # np.save(data_path + 'disc_birth_densities', disc_birth_densities)
        # np.save(data_path + 'bulge_birth_densities', bulge_birth_densities)
        np.save(data_path + 'disc_stellar_angular_momenta', disc_stellar_angular_momenta)
        np.save(data_path + 'bulge_stellar_angular_momenta', bulge_stellar_angular_momenta)
        
        print('Finished AppendAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        :return: subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return subhalo_data
    
    
    def mask_haloes(self):
        """
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 5e9 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 5e9)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if str(sys.argv[1]) == '-rd':
        print('Reading attributes')
        x = ReadAttributes(simulation_path, tag)
    elif str(sys.argv[1]) == '-ad':
        print('Adding attributes')
        x = AddAttributes(simulation_path, tag)
    elif str(sys.argv[1]) == '-ap':
        print('Appending attributes')
        x = AppendAttributes(simulation_path, tag)
