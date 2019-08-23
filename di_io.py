# Import required python libraries #
import re

import h5py
import numpy as np

last_file = 255
first_file = 0


def show_particle_properties(par_type):
	"""
	Function to show particle properties to the user.
	
	:param par_type: particle type (input from user)
	:return: None

	"""

	# Define the number of files you want to read and their name #

	file_prefix = 'snap_'
	file_postfix = '.hdf5'
	output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/particledata_012_z004p688/'
	run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/snapshot_|_', output_path)[1:4]]
	file = output_path + file_prefix + file_number + '_' + redshift + '.'

	# Determine the size and declare arrays to hold the data #
	f = h5py.File(file + str(last_file) + file_postfix, 'r')
	print('Found ' + str(len(f['PartType' + par_type].keys())) + ' properties for PartType' + par_type + ': ' + str(f['PartType' + par_type].keys())[
																												15:-2])
	f.close()

	return None


def show_galactic_properties():
	"""
	Function to show galactic properties to the user.

	:return: None

	"""

	# Define the number of files you want to read and their name #
	file_prefix = 'eagle_subfind_tab_'
	file_postfix = '.hdf5'
	output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
	run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/groups_|_', output_path)[1:4]]
	file = output_path + file_prefix + file_number + '_' + redshift + '.'

	# Determine the size and declare arrays to hold the data #
	f = h5py.File(file + str(last_file) + file_postfix, 'r')
	print('Found ' + str(len(f['Subhalo'].keys())) + ' properties for galaxies' + ': ' + str(f['Subhalo'].keys())[15:-2])
	f.close()

	return None


class AskUser:

	def __init__(self):
		pass

	def io(self):
		"""
		Ask the user what do they want to do.
		
		:return: properties
		"""
		axes = ('x', 'y', 'z')

		shortcut = input('Do you want to take a shortcut? y/n: ')

		# Create a list with the particle properties you want to read #
		if shortcut is 'y':
			condition = 'n'
		else:
			condition = input('Do you want to read new data? y/n: ')

		if condition == 'y':
			condition = input('Do you want to read groups or snapshots? g/s: ')
			if condition == 'g':
				show_galactic_properties()

				prop_num = input("How many properties do you want to read?: ")
				properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

				eat_groups(properties)
			elif condition == 's':
				par_type = input('What type of particles do you want to use? 0-5: ')
				show_particle_properties(par_type)

				prop_num = input("How many properties do you want to read?: ")
				properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

				eat_snapshots(properties, par_type)
		elif condition == 'n':

			if shortcut is 'y':
				condition = 'g'
			else:
				condition = input('Do you want to use groups or snapshots? g/s: ')

			if condition == 'g':
				if shortcut is 'y':
					prop_num = '2'
					properties = ['Mass', 'HalfMassRad']
				else:
					show_galactic_properties()

					prop_num = input("How many properties do you want to use?: ")
					properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

			elif condition == 's':
				par_type = input('What type of particles do you want to use? 0-5: ')
				show_particle_properties(par_type)

				prop_num = input("How many properties do you want to use?: ")
				properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

		return properties, prop_num

	def plot(self):
		"""
		Ask the user what type of plot do they want to do.

		:return:
		"""
		style = input('Do you want a scatter or hexbin plot? s or h: ')
		xscale = input('What scale do you want the x axis to be? log/linear: ')
		yscale = input('What scale do you want the y axis to be? log/linear: ')

		return style, xscale, yscale


def eat_snapshots(properties, par_type):
	"""
	Function to eat (i.e., open-read-save) snapshots.
	
	:param properties: particle properties (input from user)
	:param par_type: particle type (input from user)
	:return: None
	
	"""

	# Define the number of files you want to read and their name #
	file_prefix = 'snap_'
	file_postfix = '.hdf5'
	output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/snapshot_012_z004p688/'
	run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/snapshot_|_', output_path)[1:4]]
	file = output_path + file_prefix + file_number + '_' + redshift + '.'

	# Determine the size and declare arrays to hold the data #
	f = h5py.File(file + str(last_file) + file_postfix, 'r')
	part_number = f['Header'].attrs.get('NumPart_ThisFile')[int(par_type)]
	f.close()

	for i in range(len(properties)):
		prop = np.empty(part_number)

		# Read in the data #
		i_file = 0
		for iFile in range(first_file, last_file + 1):
			with h5py.File(file + '%i' % iFile + file_postfix, 'r') as f:
				n_file = part_number

				# Properties #
				prop[i_file:i_file + n_file] = f['PartType' + str(par_type)][properties[i]]

				i_file += n_file

		# Save the arrays so you can load them multiple times in different scripts #
		np.save('./data/snapshots/' + properties[i], prop)

	return None


def eat_groups(properties):
	"""
	Function to eat (i.e., open-read-save) groups.

	:param properties: galactic properties (input from user)
	:return: None

	"""

	# Define the number of files you want to read and their name #
	ga_number = 0
	file_prefix = 'eagle_subfind_tab_'
	file_postfix = '.hdf5'
	output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
	run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/groups_|_', output_path)[1:4]]
	file = output_path + file_prefix + file_number + '_' + redshift + '.'

	# Determine the size and declare arrays to hold the data #
	for iFile in range(first_file, last_file + 1):
		with h5py.File(file + '%i' % iFile + file_postfix, 'r') as f:
			shape = [len(f['Subhalo'][properties[i]].shape) for i in range(len(properties))]
			ga_number += f['Header'].attrs.get('Nsubgroups')
			print('Found ' + str(f['Header'].attrs.get('Nsubgroups')) + ' galaxies in file ' + str(iFile))
	print('Found ' + str(ga_number) + ' galaxies in total')

	for i in range(len(properties)):
		i_file = 0
		prop = np.empty(ga_number)

		if shape[i] == 1:
			# Read in the data #
			for iFile in range(first_file, last_file + 1):
				with h5py.File(file + '%i' % iFile + file_postfix, 'r') as f:
					n_file = f['Header'].attrs.get('Nsubgroups')
					# Properties #
					prop[i_file:i_file + n_file] = f['Subhalo'][properties[i]]

					i_file += n_file
		else:
			# Read in the data #
			print('Property ' + properties[i] + ' is a 2D array')
			par_type = input('What type of particles do you want to use? 0-5: ')

			for iFile in range(first_file, last_file + 1):
				with h5py.File(file + '%i' % iFile + file_postfix, 'r') as f:
					n_file = f['Header'].attrs.get('Nsubgroups')

					# Properties #
					prop[i_file:i_file + n_file] = list(zip(*f['Subhalo']['HalfMassRad']))[int(par_type)]

					i_file += n_file

		# Save the arrays so you can load them multiple times in different scripts #
		np.save('./data/groups/' + properties[i], prop)

	return None


def convert_to_physical_units(property):
	"""
	Convert a property to physical units.

	:param property: galactic property (input from user)
	:return:

	"""
	file_prefix = 'eagle_subfind_tab_'
	file_postfix = '.hdf5'
	output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
	run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/groups_|_', output_path)[1:4]]
	file = output_path + file_prefix + file_number + '_' + redshift + '.'

	f = h5py.File(file + str(first_file) + file_postfix, 'r')
	a = f['Header'].attrs.get('Time')
	h = f['Header'].attrs.get('HubbleParam')
	cgs = f['Subhalo'][property].attrs.get('CGSConversionFactor')
	aexp = f['Subhalo'][property].attrs.get('aexp-scale-exponent')
	hexp = f['Subhalo'][property].attrs.get('h-scale-exponent')
	f.close()

	prop = np.load('./data/groups/' + property + '.npy')
	prop = np.multiply(prop, cgs * a ** aexp * h ** hexp, dtype='f8')

	return prop