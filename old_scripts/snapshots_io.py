# Import required python libraries #
import glob
import re
import sys
from pprint import pprint

import h5py
import numpy as np

# First and last files of the G-EAGLE data #
last_file = 255
first_file = 0


class Show:
    """
    A class that contains all the methods (static) that show properties to the user.

    """


    @staticmethod
    def show_particle_properties(par_type):
        """
        A function to show particle properties to the user.

        :param par_type: particle type (input from user from di_io.Ask.read).
        :return: None

        """

        # Define the name/path of the files you want to read #
        file_prefix = 'snap_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/snapshot_012_z004p688/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/snapshot_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Show the available properties to the user #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Found ' + str(len(f['PartType' + par_type].keys())) + ' properties for PartType' + par_type + ':')
            pprint(str(f['PartType' + par_type].keys())[15:-2])
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        return None


    @staticmethod
    def show_subhalo_properties():
        """
        A function to show subhalo properties to the user.

        :return: None

        """

        # Define the name/path of the files you want to read #
        file_prefix = 'eagle_subfind_tab_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/groups_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Show the available properties to the user #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Found ' + str(len(f['Subhalo'].keys())) + ' properties for this subhalo:')
            pprint(str(f['Subhalo'].keys())[15:-2])
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        return None


    @staticmethod
    def show_stellar_properties():
        """
        A function to show stellar properties to the user.

        :return: None

        """

        # Define the name/path of the files you want to read #
        file_prefix = 'eagle_subfind_tab_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/groups_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Show the available properties to the user #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Found ' + str(len(f['Subhalo']['Stars'].keys())) + ' properties for stars' + ':')
            pprint(str(f['Subhalo']['Stars'].keys())[15:-2])
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        return None


class Ask:
    """
    A class that contains all the methods (static) that ask the user for input.

    """


    @staticmethod
    def read():
        """
        A function to ask the user which plots/files/properties they want to make/read/use.

        :return: properties

        """

        axes = ('x', 'y', 'z')  # Plot axes.

        # Ask if the user wants to reproduce a new or an existing plot #
        plot = input('Do you want to make a new or an existing plot? n/e: ')
        rules = [plot is not 'n', plot is not 'e']
        # Warn the user that the answer was wrong #
        while all(rules) is True:
            plot = input('Wrong input! Do you want to make a new or an existing plot? n/e: ')
            rules = [plot is not 'n', plot is not 'e']

        if plot is 'e':  # Existing.
            # Show the available plots to the user #
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Available plots:')
            with open("avail_plots.txt") as f:
                print(f.read())
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

            # Ask the user to pick one of the available plots #
            reproduce = input('Which plot do you want to reproduce? [1-]: ')
            rules = [reproduce is not '1']
            # Warn the user that the answer was wrong #
            while all(rules) is True:
                reproduce = input('Wrong input! Which plot do you want to reproduce? [1-]: ')
                rules = [reproduce is not '1']

            # Produce the chosen plot #
            if reproduce is '1':
                exec(open("groups_mass_size.py").read())

            sys.exit()  # Finish the script.

        elif plot is 'n':  # New.
            # Ask if the user wants to read new data #
            read = input('Do you want to read new data? y/n: ')
            rules = [read is not 'y', read is not 'n']
            # Warn the user that the answer was wrong #
            while all(rules) is True:
                read = input('Wrong input! Do you want to read new data? y/n: ')
                rules = [read is not 'y', read is not 'n']

            if read is 'y':  # Read
                # Ask if the read wants to read group or snapshot data #
                file = input('Do you want to read groups or snapshots? g/s: ')
                rules = [file is not 'g', file is not 's']
                # Warn the user that the answer was wrong #
                while all(rules) is True:
                    file = input('Wrong input! Do you want to read groups or snapshots? g/s: ')
                    rules = [file is not 'g', file is not 's']

                if file is 'g':  # Groups
                    # Ask if the user wants to read subhalo or stellar properties #
                    component = input('Do you want to read subhalo groups or stars? g/s: ')
                    rules = [component is not 'g', component is not 's']
                    # Warn the user that the answer was wrong #
                    while all(rules) is True:
                        component = input('Wrong input! Do you want to read subhalo groups or stars? g/s: ')
                        rules = [component is not 'g', component is not 's']

                    if component is 'g':  # Subhalo groups
                        Show.show_subhalo_properties()  # Show them the properties.

                        # Ask the user how many properties they want to read #
                        prop_num = input("How many properties do you want to read?: ")
                        properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

                        Eat.eat_groups(properties)  # Open-read-save the properties chosen by the user.
                    elif component is 's':  # Stars
                        Show.show_stellar_properties()  # Show them the properties.
                        # Ask the user how many properties they want to read #
                        prop_num = input("How many properties do you want to read?: ")
                        properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

                        Eat.eat_stars(properties)  # Open-read-save the properties chosen by the user.

                elif file is 's':  # Snapshots
                    # Ask the user what type of particles they want to read #
                    par_type = input('What type of particles do you want to use? 0-5: ')
                    rules = [par_type is not '0', par_type is not '1', par_type is not '2', par_type is not '3', par_type is not '4',
                             par_type is not '5']
                    # Warn the user that the answer was wrong #
                    while all(rules) is True:
                        par_type = input('Wrong input! What type of particles do you want to use? 0-5: ')
                        rules = [par_type is not '0', par_type is not '1', par_type is not '2', par_type is not '3', par_type is not '4',
                                 par_type is not '5']

                    Show.show_particle_properties(par_type)  # Show them the properties.

                    # Ask the user how many properties they want to read #
                    prop_num = input("How many properties do you want to read?: ")
                    properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

                    Eat.eat_snapshots(properties, par_type)  # Open-read-save the properties chosen by the user.

            elif read is 'n':  # Don't read.
                # Ask if the read wants to read group or snapshot data #
                file = input('Do you want to use groups or snapshots? g/s: ')
                rules = [file is not 'g', file is not 's']
                # Warn the user that the answer was wrong #
                while all(rules) is True:
                    file = input('Wrong input! Do you want to use groups or snapshots? g/s: ')
                    rules = [file is not 'g', file is not 's']

                if file is 'g':  # Groups.
                    # Ask if the user wants to read subhalo or stellar properties #
                    component = input('Do you want to read subhalo groups or stars? g/s: ')
                    rules = [component is not 'g', component is not 's']
                    # Warn the user that the answer was wrong #
                    while all(rules) is True:
                        component = input('Wrong input! Do you want to read subhalo groups or stars? g/s: ')
                        rules = [component is not 'g', component is not 's']

                    if component is 'g':  # Subhalo groups
                        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
                        print('Found ' + str(len(glob.glob('./data/groups/subhalo/*.npy'))) + ' available properties:')  # Show them the available
                        # properties.
                        for name in glob.glob('./data/groups/subhalo/*.npy'):
                            print(name)
                        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

                    elif component is 's':  # Stars
                        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
                        print('Found ' + str(len(glob.glob('./data/groups/stars/*.npy'))) + ' available properties:')  # Show them the available
                        # properties.
                        for name in glob.glob('./data/groups/stars/*.npy'):
                            print(name)
                        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

                    # Ask the user how many properties they want to read #
                    prop_num = input("How many properties do you want to use?: ")
                    properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

                elif file is 's':  # Snapshots
                    par_type = input('What type of particles do you want to use? 0-5: ')
                    rules = [par_type is not '0', par_type is not '1', par_type is not '2', par_type is not '3', par_type is not '4',
                             par_type is not '5']
                    # Warn the user that the answer was wrong #
                    while all(rules) is True:
                        par_type = input('Wrong input! What type of particles do you want to use? 0-5: ')
                        rules = [par_type is not '0', par_type is not '1', par_type is not '2', par_type is not '3', par_type is not '4',
                                 par_type is not '5']

                    print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
                    print('Found ' + str(len(glob.glob('./data/snapshots/*.npy'))) + ' available properties:')  # Show them the available
                    # properties.
                    for name in glob.glob('./data/snapshots/*.npy'):
                        print(name)
                    print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

                    # Ask the user how many properties they want to read #
                    prop_num = input("How many properties do you want to use?: ")
                    properties = [input("Property " + axes[i] + " : ") for i in range(int(prop_num))]

        return properties, prop_num, file, component


    @staticmethod
    def plot():
        """
        A function to ask the user what type of plot do they want to do.

        :return: n_plots, style, xscale, yscale

        """

        # Ask if the user wants a hexbin or a scatter plot and what scale they want the axes to be #
        style = input('Do you want a scatter or hexbin plot? s/h: ')
        rules = [style is not 's', style is not 'h']
        # Warn the user that the answer was wrong #
        while all(rules) is True:
            style = input('Wrong input! Do you want a scatter or hexbin plot? s/h: ')
            rules = [style is not 's', style is not 'h']

        # Ask the user to define the number of panels the plot will have #
        n_plots = input('Do you want a one or two panels? [1,2]: ')
        rules = [n_plots is not '1', n_plots is not '2']
        # Warn the user that the answer was wrong #
        while all(rules) is True:
            n_plots = input('Wrong input! Do you want a one or two panels? [1,2]: ')
            rules = [n_plots is not '1', n_plots is not '2']

        xscale = input('What scale do you want the x axis to be? log/linear: ')
        rules = [xscale is not 'log', xscale is not 'linear']
        # Warn the user that the answer was wrong #
        # while all(rules) is True:
        #     xscale = input('Wrong input! What scale do you want the x axis to be? log/linear: ')
        #     rules = [xscale is not 'log', xscale is not 'linear']

        yscale = input('What scale do you want the y axis to be? log/linear: ')
        rules = [yscale is not 'log', yscale is not 'linear']
        # Warn the user that the answer was wrong #
        # while all(rules) is True:
        #     yscale = input('Wrong input! What scale do you want the y axis to be? log/linear: ')
        #     rules = [yscale is not 'log', yscale is not 'linear']

        return n_plots, style, xscale, yscale


class Eat:
    """
    A class that contains all the methods (static) to eat data.

    """


    @staticmethod
    def eat_snapshots(properties, par_type):
        """
        A function to eat (i.e., open-read-save) snapshots.

        :param properties: particle properties (input from user).
        :param par_type: particle type (input from user).
        :return: None

        """

        # Define the name/path of the files you want to read #
        file_prefix = 'snap_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/snapshot_012_z004p688/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/snapshot_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Determine the size of the arrays to hold the data #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            part_number = f['Header'].attrs.get('NumPart_ThisFile')[int(par_type)]

        # Eat the snapshots #
        for i in range(len(properties)):
            prop = np.empty(part_number)

            i_file = 0
            for iFile in range(first_file, last_file + 1):
                # Open and read the data #
                with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                    n_file = part_number
                    prop[i_file:i_file + n_file] = f['PartType' + str(par_type)][properties[i]]
                    i_file += n_file

            np.save('./data/snapshots/' + properties[i], prop)  # Save the arrays so you can load them multiple times in different scripts #

        return None


    @staticmethod
    def eat_groups(properties):
        """
        A function to eat (i.e., open-read-save) groups.

        :param properties: galactic properties (input from user from di_io.Ask.read).
        :return: None

        """

        # Define the name/path of the files you want to read #
        ga_number = 0
        file_prefix = 'eagle_subfind_tab_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/groups_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Determine the size of the arrays to hold the data #
        for iFile in range(first_file, last_file + 1):
            with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                shape = [len(f['Subhalo'][properties[i]].shape) for i in range(len(properties))]
                ga_number += f['Header'].attrs.get('Nsubgroups')
                print('Found ' + str(f['Header'].attrs.get('Nsubgroups')) + ' galaxies in file ' + str(iFile))
        print('Found ' + str(ga_number) + ' galaxies in total')
        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        # Eat the groups #
        for i in range(len(properties)):
            i_file = 0
            prop = np.empty(ga_number)

            # For 1D arrays #
            if shape[i] == 1:
                # Read in the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = f['Subhalo'][properties[i]]
                        i_file += n_file

            else:  # For 2D arrays.
                # Ask the user what particles they want to read #
                print('Property ' + properties[i] + ' is a 2D array')
                par_type = input('What type of particles do you want to use? 0-5: ')

                # Open and read the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = list(zip(*f['Subhalo'][properties[i]]))[int(par_type)]
                        i_file += n_file

            np.save('./data/groups/subhalo/' + properties[i], prop)

        return None


    @staticmethod
    def eat_stars(properties):
        """
        A function to eat (i.e., open-read-save) stars.

        :param properties: stellar properties (input from user read).
        :return: None

        """

        # Define the name/path of the files you want to read #
        ga_number = 0
        file_prefix = 'eagle_subfind_tab_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/groups_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Determine the size of the arrays to hold the data #
        for iFile in range(first_file, last_file + 1):
            with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                shape = [len(f['Subhalo']['Stars'][properties[i]].shape) for i in range(len(properties))]
                ga_number += f['Header'].attrs.get('Nsubgroups')
                print('Found ' + str(f['Header'].attrs.get('Nsubgroups')) + ' galaxies in file ' + str(iFile))
        print('Found ' + str(ga_number) + ' galaxies in total')
        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        # Eat the groups #
        for i in range(len(properties)):
            i_file = 0
            prop = np.empty(ga_number)

            # For 1D arrays #
            if shape[i] == 1:
                # Read in the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = f['Subhalo']['Stars'][properties[i]]
                        i_file += n_file

            else:  # For 2D arrays.
                # Ask the user what particles they want to read #
                print('Property ' + properties[i] + ' is a 2D array')
                par_type = input('What type of particles do you want to use? 0-5: ')

                # Open and read the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = list(zip(*f['Subhalo']['Stars'][properties[i]]))[int(par_type)]
                        i_file += n_file

            np.save('./data/groups/stars/' + properties[i], prop)  # Save the arrays so you can load them multiple times in different scripts #

        return None


class Convert:
    """
    A class that contains all the methods (static) to convert properties to physical units.

    """


    @staticmethod
    def convert_to_physical_units(properties, file, component):

        # Ask the user if they want to convert the property to physical units #
        convert = input('Do you want to convert the property to physical units? y/n: ')
        rules = [convert is not 'y', convert is not 'n']  # Warn the user that the answer was wrong.
        while all(rules) is True:
            convert = input('Wrong input! Do you want to convert the property to physical units? y/n: ')
            rules = [convert is not 'y', convert is not 'n']
        if convert is 'y':
            if file is 'g':
                if component is 'g':
                    x, y = Convert.convert_groups_to_physical_units(properties)
                else:
                    x, y = Convert.convert_stars_to_physical_units(properties)
            else:
                x, y = Convert.convert_snapshots_to_physical_units(properties)

        return x, y


    @staticmethod
    def convert_groups_to_physical_units(properties):
        """
        A function to convert a property to physical units.

        :param file: path to the files you want to convert
        :param prop: galactic property (input from user from di_io.Ask.read).
        :return:

        """
        # Define the name/path of the files you want to read #
        file_prefix = 'eagle_subfind_tab_'
        output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
        run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/groups_|_', output_path)[1:4]]
        file = output_path + file_prefix + file_number + '_' + redshift + '.'

        # Get the conversion factors #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            a = f['Header'].attrs.get('Time')
            h = f['Header'].attrs.get('HubbleParam')
            cgs = f['Subhalo'][properties[0]].attrs.get('CGSConversionFactor')
            aexp = f['Subhalo'][properties[0]].attrs.get('aexp-scale-exponent')
            hexp = f['Subhalo'][properties[0]].attrs.get('h-scale-exponent')

            # Convert to physical units #
            prop = np.load('./data/groups/subhalo/' + properties[0] + '.npy')
            x = np.multiply(prop, cgs * a ** aexp * h ** hexp, dtype='f8')

        # Get the conversion factors #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            a = f['Header'].attrs.get('Time')
            h = f['Header'].attrs.get('HubbleParam')
            cgs = f['Subhalo'][properties[1]].attrs.get('CGSConversionFactor')
            aexp = f['Subhalo'][properties[1]].attrs.get('aexp-scale-exponent')
            hexp = f['Subhalo'][properties[1]].attrs.get('h-scale-exponent')

            # Convert to physical units #
            prop = np.load('./data/groups/subhalo/' + properties[1] + '.npy')
            y = np.multiply(prop, cgs * a ** aexp * h ** hexp, dtype='f8')

        return x, y