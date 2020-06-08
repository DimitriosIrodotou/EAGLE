import numpy as np
import matplotlib.pyplot as plt


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


def create_colorbar(ax, plot, label, orientation='vertical'):
    """
    Generate a colorbar.
    :param ax: colorbar axis.
    :param plot: corresponding plot.
    :param label: colorbar label.
    :param orientation: colorbar orientation.
    :return: None
    """
    cbar = plt.colorbar(plot, cax=ax, orientation=orientation)
    cbar.set_label(label, size=16)
    
    if orientation == 'horizontal':
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(direction='out', which='both', top='on', labelsize=16)
    return None
