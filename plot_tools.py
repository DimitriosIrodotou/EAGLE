import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy_healpix import HEALPix


class RotateCoordinates:
    """
    Rotate coordinates and velocities wrt different quantities.
    """


    @staticmethod
    def rotate_X(stellar_data_tmp, glx_unit_vector):
        """
        Rotate first about z-axis to set y=0 and then about the y-axis to set z=0
        :param stellar_data_tmp: from read_add_attributes.py.
        :param glx_unit_vector: from mask_galaxies
        :return: stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector
        """
        # Calculate the rotation matrices and combine them #
        ra = np.arctan2(glx_unit_vector[1], glx_unit_vector[0])
        el = np.arcsin(glx_unit_vector[2])

        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(el), 0, np.sin(el)], [0, 1, 0], [-np.sin(el), 0, np.cos(el)]])
        Ryz = np.matmul(Ry, Rz)

        # Rotate the coordinates and velocities of stellar particles #
        coordinates = np.matmul(Ryz, stellar_data_tmp['Coordinates'][..., None]).squeeze()
        velocities = np.matmul(Ryz, stellar_data_tmp['Velocity'][..., None]).squeeze()

        # Recalculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(coordinates, velocities)  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.
        glx_unit_vector = glx_angular_momentum / np.linalg.norm(glx_angular_momentum)
        prc_unit_vector = prc_angular_momentum / np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis]

        return coordinates, velocities, prc_unit_vector, glx_unit_vector


    @staticmethod
    def rotate_densest(prc_unit_vector, glx_unit_vector):
        """
        Rotate first about z-axis to set y=0and then about the y-axis to set z=0
        :param prc_unit_vector:
        :param glx_unit_vector:
        :return: prc_unit_vector, glx_unit_vector
        """
        # Calculate the ra and el of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        el = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, el * u.deg)  # Create list of HEALPix indices from particles' ra and el.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.

        # Find location of density maximum and plot its positions and the ra (lon) and el (lat) of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2

        # Calculate the rotation matrices and combine them #
        ra = np.float(lon_densest)
        el = np.float(lat_densest)
        print(density[index_densest])
        # ra = np.arctan2(glx_unit_vector[1], glx_unit_vector[0])
        # el = np.arcsin(glx_unit_vector[2])

        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(el), 0, np.sin(el)], [0, 1, 0], [-np.sin(el), 0, np.cos(el)]])
        Ryz = np.matmul(Ry, Rz)

        prc_unit_vector = np.matmul(Ryz, prc_unit_vector[..., None]).squeeze()
        glx_unit_vector = np.matmul(Ryz, glx_unit_vector)

        return prc_unit_vector, glx_unit_vector


    @staticmethod
    def rotate_Jz(stellar_data_tmp):
        """
        Rotate a galaxy such that its angular momentum is along the z axis.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_angular_momentum, glx_angular_momentum
        """

        # Calculate the angular momentum of the galaxy #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.

        # Define the rotation matrices #
        a = np.matrix([glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]]) / np.linalg.norm(
            [glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]])
        b = np.matrix([0, 0, 1])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b.T)
        vx = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
        transform = np.eye(3, 3) + vx + (vx * vx) * ((1 - c[0, 0]) / s ** 2)

        # Rotate the coordinates and velocities #
        coordinates = np.array([np.matmul(transform, stellar_data_tmp['Coordinates'][i].T) for i in range(0, len(stellar_data_tmp['Coordinates']))])[
                      :, 0]
        velocities = np.array([np.matmul(transform, stellar_data_tmp['Velocity'][i].T) for i in range(0, len(stellar_data_tmp['Velocity']))])[:, 0]

        # Calculate the rotated angular momentum of the galaxy #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(coordinates, velocities)  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.

        return coordinates, velocities, prc_angular_momentum, glx_angular_momentum


def median_1sigma(x_data, y_data, delta, log):
    """
    Calculate the median and 1-sigma lines.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param delta: step.
    :param log: boolean.
    :return: x_value, median, shigh, slow
    """
    # Initialise arrays #
    if log is True:
        x = np.log10(x_data)
    else:
        x = x_data
    n_bins = int((max(x) - min(x)) / delta)
    x_value = np.empty(n_bins)
    median = np.empty(n_bins)
    slow = np.empty(n_bins)
    shigh = np.empty(n_bins)
    x_low = min(x)

    # Loop over all bins and calculate the median and 1-sigma lines #
    for i in range(n_bins):
        index, = np.where((x >= x_low) & (x < x_low + delta))
        x_value[i] = np.mean(x_data[index])
        if len(index) > 0:
            median[i] = np.nanmedian(y_data[index])
        slow[i] = np.nanpercentile(y_data[index], 15.87)
        shigh[i] = np.nanpercentile(y_data[index], 84.13)
        x_low += delta

    return x_value, median, shigh, slow


def binned_median_1sigma(x_data, y_data, bin_type, n_bins, log=False):
    """
    Calculate the binned median and 1-sigma lines in either equal number of width bins.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param bin_type: equal number or width type of the bin.
    :param n_bins: number of the bin.
    :param log: boolean.
    :return: x_value, median, shigh, slow
    """
    if bin_type == 'equal_number':
        if log is True:
            x = np.log10(x_data)
        else:
            x = x_data

        # Declare arrays to store the data #
        n_bins = np.quantile(np.sort(x), np.linspace(0, 1, n_bins + 1))
        slow = np.zeros(len(n_bins))
        shigh = np.zeros(len(n_bins))
        median = np.zeros(len(n_bins))
        x_value = np.zeros(len(n_bins))

        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(len(n_bins) - 1):
            index, = np.where((x >= n_bins[i]) & (x < n_bins[i + 1]))
            x_value[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)

        return x_value, median, shigh, slow

    elif bin_type == 'equal_width':
        if log is True:
            x = np.log10(x_data)
        else:
            x = x_data
        x_low = min(x)

        # Declare arrays to store the data #
        bin_width = (max(x) - min(x)) / n_bins
        slow = np.zeros(n_bins)
        shigh = np.zeros(n_bins)
        median = np.zeros(n_bins)
        x_value = np.zeros(n_bins)

        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(n_bins):
            index, = np.where((x >= x_low) & (x < x_low + bin_width))
            x_value[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)
            x_low += bin_width

        return x_value, median, shigh, slow


def create_colorbar(axis, plot, label, orientation='vertical', top=True, ticks=None, size=16):
    """
    Generate a colorbar.
    :param axis: colorbar axis.
    :param plot: corresponding plot.
    :param label: colorbar label.
    :param top: move ticks and labels on top of the colorbar.
    :param ticks: array of ticks.
    :param orientation: colorbar orientation.
    :return: None
    """
    cbar = plt.colorbar(plot, cax=axis, ticks=ticks, orientation=orientation)
    cbar.set_label(label, size=size)
    axis.tick_params(direction='out', which='both', right='on', labelsize=size)

    if top is True:
        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")
        axis.tick_params(direction='out', which='both', top='on', labelsize=size)
    return None


def set_axis(axis, xlim=None, ylim=None, xscale=None, yscale=None, xlabel=None, ylabel=None, aspect='equal', which='both', size=16):
    """
    Set axis parameters.
    :param axis: name of the axis.
    :param xlim: x axis limits.
    :param ylim: y axis limits.
    :param xscale: x axis scale.
    :param yscale: y axis scale.
    :param xlabel: x axis label.
    :param ylabel: y axis label.
    :param aspect: aspect of the axis scaling.
    :param which: major, minor or both for grid and ticks.
    :param size: text size.
    :return:
    """
    # Set axis limits #
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)

    # Set axis labels #
    if xlabel:
        axis.set_xlabel(xlabel, size=size)
    if ylabel:
        axis.set_ylabel(ylabel, size=size)

    # Set axis scales #
    if xscale:
        axis.set_xscale(xscale)
    if yscale:
        axis.set_yscale(yscale)

    if not xlim and not xlabel:
        axis.set_xticks([])
        axis.set_xticklabels([])
    if not ylim and not ylabel:
        axis.set_yticks([])
        axis.set_yticklabels([])

    # Set grid and tick parameters #
    if aspect is not None:
        axis.set_aspect('equal')
    axis.grid(True, which=which, axis='both', color='gray', linestyle='-')
    axis.tick_params(direction='out', which=which, top='on', right='on', labelsize=size)

    return None
