# Import required python libraries #
import time

import matplotlib.pyplot as plt
import seaborn as sns

date = time.strftime("%d\%m\%y\%H%M")


def generate_figure(nplots=1):
    """
    Function to generate a figure.
    
    :param nplots: number of plots
    :return: figure
    """
    plt.close()
    if nplots == 1:
        figure = plt.figure(0, figsize=(10, 7.5))
        return figure
    else:
        figure = plt.figure(0, figsize=(10, 7.5))
        return figure


def set_figure_param(nplots, figure, x, y, xscale=None, yscale=None, xlim=None, ylim=None, xlabel=None, ylabel=None):
    """
    Function to set figure parameters.
    
    :param nplots: number of plots
    :param figure: figure (input from create_figure)
    :param x: x-axis property
    :param y: y-axis property
    :param yscale: x-axis scale
    :param xscale: y-axis scale
    :param xlim: x-axis limit
    :param ylim: y-axis limit
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :return: None
    """
    # Declare the plotting style #
    sns.set()
    sns.set_style('ticks', {'axes.grid': True})
    sns.set_context('notebook', font_scale=1.6)
    
    if nplots == 1:
        
        # Set the axes scales #
        if xscale is not None:
            plt.xscale(yscale)
        if yscale is not None:
            plt.yscale(yscale)
        
        # Set the axes limits #
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        
        # Set the axes labels #
        if xlabel is None:
            plt.xticks([])
        else:
            plt.xlabel(xlabel)
        if ylabel is None:
            plt.yticks([])
        else:
            plt.ylabel(ylabel)
        
        plt.tick_params(direction='in', which='both', top='on', right='on')
    
    else:
        figure.set_yticks([])
    
    return None


def make_plot(style, X, Y):
    """
    Function to make a plot.
    
    :param style: style of the plot.
    :return:
    """
    lw = 3  # linewidth
    mc = 1  # mincnt
    sp = 3  # scatterpoints
    ms = 5  # markersize
    gs = 150  # gridsize
    size = 50  # size
    
    if style == 'scatter':
        plt.scatter(X, Y, c='lime', edgecolor='black')
    
    plt.savefig('X_Vs_Y' + '-' + date, bbox_inches='tight')
    return None