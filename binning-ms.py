# Written by Brent Delano
# 4/6/2020
# Takes a .mgf file and creates a matrix of where m/z ratios from spectra fit into bins of different sizes
# The rows of the matrix are the bins (bin min is the lowest m/z value, truncated, and bin max at the greatest m/z value, rounded up) 
# The columns of the matrix are the spectra
# If an m/z ratio of a spectra (m) fits into a bin (n), then the value (n,m) on the matrix will be the intensity value of that m/z ratio;
#	if not, then it will be 0
# Able to add Gaussian noise to the dataset in order to visualize the effect on bins
# Plots histogram of m/z spectra in bins with relative frequency
# Uses https://www.python-course.eu/pandas_python_binning.php for binning functions
# Uses https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html for plots

import pyteomics
from pyteomics import mgf
import math
import numpy as np
import matplotlib.pyplot as plt

def main():
	# reads mgf file and initializes lists of m/z ratios and respective intensities
	mzs = read_mgf('HMBD.mgf')[0]
	intensities = read_mgf('HMBD.mgf')[1] 

	# adds gaussian noise to the m/z dataset (comment this line if you don't want noise)
	mzs = create_gaussian_noise(mzs)

	# creates bins
	minmz = 0
	maxmz = 0
	for i,mz in enumerate(mzs):
		if (i == 0):
			minmz = mz[0]
			maxmz = mz[0]
		for m in mz:
			if (m < minmz):
				minmz = m
			if (m > maxmz):
				maxmz = m
	bins = create_bins(lower_bound=math.floor(minmz), width=1, quantity=math.ceil(maxmz-minmz))

	# prints peaks matrix
	print(create_peak_matrix(mzs, intensities, bins))

	# graphs histogram
	graph(mzs, len(bins), [minmz,maxmz])


# takes in a .mgf file and reads it
# outputs a list holding the m/z ratios and a list holding the respective intensities
def read_mgf(mgfFile):
	mzs = []
	intensities = []
	with mgf.MGF('HMDB.mgf') as reader:
		for spectrum in reader:
			mzs.append(spectrum['m/z array'].tolist())
			intensities.append(spectrum['intensity array'].tolist())
	return mzs, intensities


# from https://www.python-course.eu/pandas_python_binning.php
# create bins of equal sizes based off of given parameters
def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins


# from https://www.python-course.eu/pandas_python_binning.php
# finds the bin that a given value fits into
def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1

	
# creates a matrix that finds the bins that the peaks in a spectra fall into
def create_peak_matrix(mzs, intensities, bins):
	peaks = []
	for i,mz in enumerate(mzs):
		temp = [0] * len(bins)
		for j,m in enumerate(mz):
			index = find_bin(m,bins)
			if (temp[index] == 0):
				temp[index] = intensities[i][j]
			else:
				if (isinstance(temp[index], list)):
					temp[index].append(intensities[i][j])
				else:
					temp[index] = [temp[index], intensities[i][j]]
		peaks.append(temp)
	return(peaks)


# adds gaussian noise to the m/z dataset
def create_gaussian_noise(mzs):
	mzs_od = []
	for mz in mzs:
		for m in mz:
			mzs_od.append(m)

	mu = np.mean(mzs_od)
	sigma = np.std(mzs_od)

	noise = np.random.normal(mu, sigma, len(mzs_od))
	noisy = mzs_od + noise

	index = 0
	noisy_shaped = []
	i = 0
	while i < len(mzs):
		temp = []
		j = 0
		while j < len(mzs[i]):
			temp.append(noisy[index])
			index += 1
			j += 1
		noisy_shaped.append(temp)
		i += 1

	return(noisy_shaped)

# Uses matplot lib and https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html to graph
# 	m/z data on a histogram; 
# Also plots the probability density function of the distribution (in red)
def graph(mzs, numBins, r):
	mzs_od = []
	for mz in mzs:
		for m in mz:
			mzs_od.append(m)

	mu = np.mean(mzs_od)
	sigma = np.std(mzs_od)

	count, bins, ignored = plt.hist(x=mzs_od, bins=numBins, density=True, range=r, histtype = 'bar', facecolor='blue')
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
	plt.ylabel('Frequency')
	plt.xlabel('M/Z Bins')
	plt.show()


if __name__ == "__main__":
    main()
