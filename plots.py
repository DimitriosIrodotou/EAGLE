# Import required python libraries #
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy import units as u

date = time.strftime("%d\%m\%y\%H%M")

# Define various plot parameters as constants#
lw = 3  # linewidth
mc = 1  # mincnt
sp = 3  # scatterpoints
ms = 5  # markersize
gs = 150  # gridsize
size = 50  # size
cmap = 'bone_r'  # colormap
dlog10 = 0.1  # bin step

# Set the general figure style #
sns.set()
sns.set_style('ticks', {'axes.grid': True})
sns.set_context('notebook', font_scale=1.6)


def generate_plot(properties, x, y, xscale, yscale, xlim=None, ylim=None, title=None):
    """
    A function to generate a figure with a plot.

    :param properties: properties (input from user from groups_io.Ask.read).
    :param x: x-axis property (defined in main.py).
    :param y: y-axis property (defined in main.py).
    :param xscale: scale of the x-axis (input from user from groups_io.Ask.style).
    :param yscale: scale of the y-axis (input from from groups_io.Ask.style).
    :param xlim: limit of the x-axis (defined in main.py).
    :param ylim: limit of the y-axis (defined in main.py).
    :param title: title of the plot (defined in main.py).
    :return: x, y, figure

    """
    # Mask the data #
    index = np.where((y > 0.0) & (x > 0.0))
    y = y[index]
    x = x[index]

    # Close previous figures and create a new one #
    plt.close()
    figure = plt.figure(0, figsize=(10, 7.5))
    plt.tick_params(direction='in', which='both', top='on', right='on')

    # Set the axes scales #
    plt.xscale(xscale)
    plt.yscale(yscale)

    # Set the axes limits #
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Set the axes labels and convert to astronomical units #
    if 'Mass' in properties[0]:
        if 'Rad' in properties[0]:
            plt.xlabel(r'$\mathrm{R_{HM} / kpc}$')
            x = ((x * u.cm).to(u.parsec) / 1e3).value  # Length units in kpc.
        else:
            plt.xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
            x = ((x * u.g).to(u.solMass)).value  # Mass units in Msun.

    elif 'Rad' in properties[0]:
        plt.xlabel(r'$\mathrm{R_{HM} / kpc}$')
        x = ((x * u.cm).to(u.parsec) / 1e3).value  # Length units in kpc.

    if 'Mass' in properties[1]:
        if 'Rad' in properties[1]:
            plt.ylabel(r'$\mathrm{R_{HM} / kpc}$')
            y = ((y * u.cm).to(u.parsec) / 1e3).value  # Length units in kpc.
        else:
            plt.ylabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
            y = ((y * u.g).to(u.solMass)).value  # Mass units in Msun.

    elif 'Rad' in properties[1]:
        plt.ylabel(r'$\mathrm{R_{HM} / kpc}$')
        y = ((y * u.cm).to(u.parsec) / 1e3).value  # Length units in kpc.

    # Set the title #
    plt.title(title)

    return x, y, figure


def generate_subplots(properties, x, y, xscale, yscale, xlim=None, ylim=None, title=None):
    """
    A function to generate a figure with subplots.

    :param properties: properties (input from user from groups_io.Ask.read).
    :param x: x-axis property (defined in main.py).
    :param y: y-axis property (defined in main.py).
    :param xscale: scale of the x-axis (input user from groups_io.Ask.style).
    :param yscale: scale of the y-axis (input user from groups_io.Ask.style).
    :param xlim: limit of the x-axis (defined in main.py).
    :param ylim: limit of the y-axis (defined in main.py).
    :param title: title of the plot (defined in main.py).
    :return: x, y, figure, ax1, ax2

    """
    # Mask the data #
    index = np.where((y > 0.0) & (x > 0.0))
    y = y[index]
    x = x[index]

    # Close previous figures and create a new one #
    plt.close()
    figure, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7.5))
    figure.subplots_adjust(hspace=0, wspace=0.3)
    ax1.tick_params(direction='in', which='both', top='on', right='on')
    ax2.tick_params(direction='in', which='both', top='on', right='on')

    # Set the axes scales #
    ax1.set_xscale(xscale)
    ax2.set_xscale(xscale)
    ax1.set_yscale(yscale)
    ax2.set_yscale(yscale)

    # Set the axes limits and convert to astronomical units #
    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)

    if 'Mass' in properties[0]:
        if 'Rad' in properties[0]:
            ax1.set_xlabel(r'$\mathrm{R_{HM} / kpc}$')
            ax2.set_xlabel(r'$\mathrm{R_{HM} / kpc}$')
            x = ((x * u.cm).to(u.parsec) / 1e3).value
        else:
            ax1.set_xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
            ax2.set_xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
            x = ((x * u.g).to(u.solMass)).value

    elif 'Rad' in properties[0]:
        ax1.set_xlabel(r'$\mathrm{R_{HM} / kpc}$')
        ax2.set_xlabel(r'$\mathrm{R_{HM} / kpc}$')
        x = ((x * u.cm).to(u.parsec) / 1e3).value

    if 'Mass' in properties[1]:
        if 'Rad' in properties[1]:
            ax1.set_ylabel(r'$\mathrm{R_{HM} / kpc}$')
            ax2.set_ylabel(r'$\mathrm{R_{HM} / kpc}$')
            y = ((y * u.cm).to(u.parsec) / 1e3).value
        else:
            ax1.set_ylabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
            ax2.set_ylabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
            y = ((y * u.g).to(u.solMass)).value

    elif 'Rad' in properties[1]:
        ax1.set_ylabel(r'$\mathrm{R_{HM} / kpc}$')
        ax2.set_ylabel(r'$\mathrm{R_{HM} / kpc}$')
        y = ((y * u.cm).to(u.parsec) / 1e3).value

    # Set the title #
    ax1.set_title(title)

    return x, y, figure, ax1, ax2


def make_plot(x, y, figure, style, xscale, yscale):
    """
    A function to make a plot.
    :param x: x-axis property (defined in main.py).
    :param y: y-axis property (defined in main.py).
    :param figure: figure (input from groups_plots.generate_plot).
    :param style: style of the plot (input user from groups_io.Ask.style).
    :param xscale: scale of the x-axis (input from user from groups_io.Ask.style).
    :param yscale: scale of the y-axis (input from user from groups_io.Ask.style).
    :return: None

    """

    if style is 's':  # Make a scatter plot.
        plt.scatter(x, y, c='black')

        # Calculate and plot median and 1-sigma lines #
        x_median, y_median, slow, shigh = median(x, y)
        plt.plot(x_median, y_median, color='grey', lw=lw)
        plt.fill_between(x_median, shigh, slow, color='grey', alpha='0.3', zorder=2)
        plt.fill(np.NaN, np.NaN, color='grey', alpha=0.3)

    if style is 'h':  # Make a hexbin plot.
        hexbin = plt.hexbin(x, y, xscale=xscale, yscale=yscale, bins=xscale, mincnt=mc, cmap=cmap, gridsize=gs)

        # Adjust the color bar #
        cbaxes = figure.add_axes([0.9, 0.11, 0.02, 0.77])
        cb = plt.colorbar(hexbin, cax=cbaxes)
        cb.set_label('$\mathrm{Counts\; per\; hexbin}$')

        # Calculate and plot median and 1-sigma lines #
        x_median, y_median, slow, shigh = median(x, y)
        plt.plot(x_median, y_median, color='maroon', lw=lw)
        plt.fill_between(x_median, shigh, slow, color='maroon', alpha='0.3', zorder=2)
        plt.fill(np.NaN, np.NaN, color='maroon', alpha=0.3)
    # Save the figure #
    plt.savefig('./plots/' + 'X_Vs_Y' + '-' + date, bbox_inches='tight')

    return None


def make_subplots(x, y, figure, ax1, ax2, style, xscale, yscale):
    """
    A function to make subplots.

    :param x: x-axis property (defined in main.py).
    :param y: y-axis property (defined in main.py).
    :param figure: figure (input from groups_plots.generate_subplots).
    :param ax1: first plot (input from groups_plots.generate_subplots).
    :param ax2: second plot (input from groups_plots.generate_subplots).
    :param style: style of the plot (input user from groups_io.Ask.style).
    :param xscale: scale of the x-axis (input from user from groups_io.Ask.style).
    :param yscale: scale of the y-axis (input from user from groups_io.Ask.style).
    :return: None

    """

    # Make a scatter plot #
    if style is 's':
        ax1.scatter(x, y, c='black')

        # Calculate and plot median and 1-sigma lines #
        x_median, y_median, slow, shigh = median(x, y)
        ax2.plot(x_median, y_median, color='grey', lw=lw)
        ax2.fill_between(x_median, shigh, slow, color='grey', alpha='0.3', zorder=2)
        ax2.fill(np.NaN, np.NaN, color='grey', alpha=0.3)

    # Make a hexbin plot #
    if style is 'h':
        hexbin = ax1.hexbin(x, y, xscale=xscale, yscale=yscale, bins=xscale, mincnt=mc, cmap=cmap, gridsize=gs)

        # Adjust the color bar #
        cbaxes = figure.add_axes([0.462, 0.11, 0.01, 0.77])
        cb = plt.colorbar(hexbin, cax=cbaxes)
        cb.set_label('$\mathrm{Counts\; per\; hexbin}$')

        # Calculate and plot median and 1-sigma lines #
        x_median, y_median, slow, shigh = median(x, y)
        ax2.plot(x_median, y_median, color='maroon', lw=lw)
        ax2.fill_between(x_median, shigh, slow, color='maroon', alpha='0.3', zorder=2)
        ax2.fill(np.NaN, np.NaN, color='maroon', alpha=0.3)

    # Save the figure #
    plt.savefig('./plots/' + 'X_Vs_Y' + '-' + date, bbox_inches='tight')

    return None


def median(x, y):
    """

    :param x: x-axis property (defined in main.py).
    :param y: y-axis property (defined in main.py).
    :return: x_median, y_median, slow, shigh
    """

    log10x = np.log10(x)
    log10xmax = np.log10(max(x))
    log10xmin = np.log10(min(x))
    nbin = int((log10xmax - log10xmin) / dlog10)
    x_median = np.empty(nbin)
    y_median = np.empty(nbin)
    slow = np.empty(nbin)
    shigh = np.empty(nbin)
    log10xlow = log10xmin
    for i in range(nbin):
        index = np.where((log10x >= log10xlow) & (log10x < log10xlow + dlog10))[0]
        x_median[i] = np.nanmean(x[index])
        if len(index) > 0:
            y_median[i] = np.nanmedian(y[index])
            slow[i] = np.nanpercentile(y[index], 15.87)
            shigh[i] = np.nanpercentile(y[index], 84.13)
        log10xlow += dlog10

    return x_median, y_median, slow, shigh


def plot():
    """
    A funtion to ask the user if they want to make a new plot or reproduce an existing plot.

    :return: None
    """

    # Ask if the user wants to make a new or an existing plot #
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
        return None