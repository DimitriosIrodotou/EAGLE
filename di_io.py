# Import required python libraries #
import re

import h5py
import numpy as np


def show_properties(par_type):
    """
    Function to show particle properties.
    
    :param par_type: particle type (input from user)
    :return: None

    """
    
    # Define the number of files you want to read and their name #
    last_file = 0
    file_prefix = 'snap_'
    file_postfix = '.hdf5'
    output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_Data/0021/particledata_012_z004p688/'
    run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/snapshot_|_', output_path)[1:4]]
    file = output_path + file_prefix + file_number + '_' + redshift + '.'
    
    # Determine the size and declare arrays to hold the data #
    f = h5py.File(file + str(last_file) + file_postfix, 'r')
    print('Found ' + str(len(f['PartType' + par_type].keys())) + ' properties for PartType' + par_type + ': ' + str(f['PartType' + par_type].keys())[
                                                                                                                15:-2])
    f.close()
    
    return None


def ask_user():
    """
    Ask the user what does he/she wants to do
    
    :return: properties
    """
    
    axes = ('x', 'y', 'z')
    
    # Create a list with the particle properties you want to read #
    condition = input('Do you want to read new data? y/n: ')
    if condition == 'y':
        par_type = input('Which type of particles do you want to use? 0-5: ')
        show_properties(par_type)
        
        prop_num = input("How many properties do you want to read?: ")
        properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]
        
        eat_snapshots(properties, par_type)
    elif condition == 'n':
        par_type = input('Which type of particles do you want to use? 0-5: ')
        show_properties(par_type)
        
        prop_num = input("How many properties do you want to use?: ")
        properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]
    
    return properties, prop_num


def eat_snapshots(properties, par_type):
    """
    Function to eat (i.e., open-read-save) snapshots.
    
    :param properties: particle properties (user input)
    :param par_type: particle type (input from user)
    :return: None
    
    """
    
    # Define the number of files you want to read and their name #
    first_file = 0
    last_file = 0
    file_prefix = 'snap_'
    file_postfix = '.hdf5'
    output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_Data/0021/snapshot_012_z004p688/'
    run, file_number, redshift = [item.replace('/', '') for item in re.split('_Data|/snapshot_|_', output_path)[1:4]]
    file = output_path + file_prefix + file_number + '_' + redshift + '.'
    
    # Determine the size and declare arrays to hold the data #
    f = h5py.File(file + str(last_file) + file_postfix, 'r')
    part_number = f['Header'].attrs.get('NumPart_ThisFile')[int(par_type)]
    f.close()
    
    for i in range(len(properties)):
        x = np.empty(part_number)
        
        # Read in the data #
        i_file = 0
        for iFile in range(first_file, last_file + 1):
            with h5py.File(file + '%i' % iFile + file_postfix, 'r') as f:
                n_file = part_number
                
                # Properties #
                x[i_file:i_file + n_file] = f['PartType' + str(par_type)][properties[i]]
                
                i_file += n_file
        
        # Save the arrays so you can load them multiple times in different scripts #
        np.save('./Data/' + properties[i], x)
        
        return None