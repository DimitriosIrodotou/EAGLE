import numpy as np
import matplotlib.pyplot as plt


def median_1sigma(x, y, delta):
    """
    Calculate the median and 1-sigma lines.
    :param x: x-axis data.
    :param y: y-axis data.
    :param delta: step.
    :return: x_value, median, shigh, slow
    """
    # Initialise arrays #
    nbin = int((max(np.log10(x)) - min(np.log10(x))) / delta)
    x_value = np.empty(nbin)
    median = np.empty(nbin)
    slow = np.empty(nbin)
    shigh = np.empty(nbin)
    x_low = min(np.log10(x))
    
    # Loop over all bins and calculate the median and 1-sigma lines #
    for i in range(nbin):
        index, = np.where((np.log10(x) >= x_low) & (np.log10(x) < x_low + delta))
        x_value[i] = np.mean(x[index])
        if len(index) > 0:
            median[i] = np.nanmedian(y[index])
        slow[i] = np.nanpercentile(y[index], 15.87)
        shigh[i] = np.nanpercentile(y[index], 84.13)
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
