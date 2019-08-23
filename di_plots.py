# Import required python libraries #
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy import units as u

date = time.strftime("%d\%m\%y\%H%M")


def generate_figure(nplots, properties, xscale, yscale, xlim=None, ylim=None, title=None):
	"""
	Function to generate a figure.
	
	:param nplots: number of plots
	:return: figure
	"""
	plt.close()
	if nplots == 1:
		figure = plt.figure(0, figsize=(10, 7.5))

		sns.set()
		sns.set_style('ticks', {'axes.grid': True})
		sns.set_context('notebook', font_scale=1.6)

		# Set the axes scales #
		plt.xscale(xscale)
		plt.yscale(yscale)

		# Set the axes limits #
		if xlim is not None:
			plt.xlim(xlim)
		if ylim is not None:
			plt.ylim(ylim)

		# Set the axes labels #
		print(properties[0])
		plt.xlabel(properties[0])
		plt.ylabel(properties[1])

		plt.title(title)
		plt.tick_params(direction='in', which='both', top='on', right='on')
		return figure
	else:
		figure = plt.figure(0, figsize=(10, 7.5))
		return figure


def make_plot(figure, style, X, Y, xscale, yscale):
	"""
	Function to make a plot.

	:param figure:
	:param style:
	:param X:
	:param Y:
	:return:

	"""

	lw = 3  # linewidth
	mc = 1  # mincnt
	sp = 3  # scatterpoints
	ms = 5  # markersize
	gs = 150  # gridsize
	size = 50  # size
	dlog10 = 0.1  # bin step

	# Convert to astronomical units #
	X = ((X * u.g).to(u.solMass)).value
	Y = ((Y * u.cm).to(u.parsec) / 1e3).value

	# Calculate median and 1-sigma #
	log10x = np.log10(X)
	log10xmax = np.log10(max(X))
	log10xmin = np.log10(min(X))
	nbin = int((log10xmax - log10xmin) / dlog10)
	x = np.empty(nbin)
	median = np.empty(nbin)
	slow = np.empty(nbin)
	shigh = np.empty(nbin)
	log10xlow = log10xmin
	for i in range(nbin):
		index = np.where((log10x >= log10xlow) & (log10x < log10xlow + dlog10))[0]
		x[i] = np.mean(X[index])
		if len(index) > 0:
			median[i] = np.median(Y[index])
			slow[i] = np.percentile(Y[index], 15.87)
			shigh[i] = np.percentile(Y[index], 84.13)
		log10xlow += dlog10

	# Plot median and 1-sigma lines #
	plt.plot(x, median, color='red', lw=lw)
	plt.fill_between(x, shigh, slow, color='red', alpha='0.5', zorder=2)
	plt.fill(np.NaN, np.NaN, color='red', alpha=0.5)

	if style == 's':
		plt.scatter(X, Y, c='black')

	if style == 'h':
		y = Y[np.where(Y > 1e-1)]
		x = X[np.where(Y > 1e-1)]

		hexbin = plt.hexbin(x, y, xscale=xscale, yscale=yscale, bins=xscale, mincnt=mc, gridsize=gs)

		# Adjust the color bar #
		cbaxes = figure.add_axes([0.9, 0.11, 0.02, 0.77])
		cb = plt.colorbar(hexbin, cax=cbaxes)
		cb.set_label('$\mathrm{Counts\; per\; hexbin}$')

	plt.savefig('./plots/' + 'X_Vs_Y' + '-' + date, bbox_inches='tight')

	return None