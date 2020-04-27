# Written by Brent Delano
# 4/6/2020
# Takes a .mgf file and creates a matrix of where m/z ratios from spectra fit into bins of different sizes
# The rows of the matrix are the bins (bin min is the lowest m/z value, truncated, and bin max at the greatest m/z value, rounded up) 
# The columns of the matrix are the spectra
# If an m/z ratio of a spectra (m) fits into a bin (n), then the value (n,m) on the matrix will be the intensity value of that m/z ratio;
#	if not, then it will be 0
# Able to add Gaussian noise to the dataset in order to visualize the effect on bins
# Plots histogram of m/z spectra in bins with relative frequency
# Compresses bins through pca
# Uses pyteomics api (https://pyteomics.readthedocs.io/en/latest/) for reading .mgf files
# Uses https://www.python-course.eu/pandas_python_binning.php for binning functions
# Uses https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html for plots
# Uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca for compression

import pyteomics
from pyteomics import mgf
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA

# takes in .mgf file file paths as strings (if more than one, then use a list of strings) and reads the .mgf file
# outputs 3 lists of lists: the first holding the m/z ratios, the second holding a list holding the respective intensities, 
#	the third a list of identifiers
def read_mgf_binning(mgfFile):
	mzs = []
	intensities = []
	identifiers = []
	if (isinstance(mgfFile, list)):
		for mgfs_n in mgfFile:
			with mgf.MGF(mgfs_n) as reader:
				for j, spectrum in enumerate(reader):
					mzs.append(spectrum['m/z array'].tolist())
					intensities.append(spectrum['intensity array'].tolist())
					identifiers.append(mgfs_n + '_' + str(j+1))
	else:
		with mgf.MGF(mgfFile) as reader:
			for j, spectrum in enumerate(reader):
				mzs.append(spectrum['m/z array'].tolist())
				intensities.append(spectrum['intensity array'].tolist())
				identifiers.append([mgfFile + '_' + str(j+1)])
	return mzs, intensities, identifiers


# finds the minimum bin size such that each m/z ratio within a spectra will fall into its own bin
def get_min_binsize(mzs):
	min = 0
	for spec in mzs:
		temp = copy.copy(spec)
		temp.sort()
		if (min == 0):
			min = temp[1]-temp[0]
		for n,mz in enumerate(temp):
			if (n < len(temp) - 1):
				diff = abs(mz - temp[n+1])
				if (diff < min):
					min = diff
	return min


# takes in m/z ratios as a list of lists (first dimension is lists representing the spectra, second dimension are the m/z values in each spectra list
# creates a list of lists of bins of size binsize
def create_bins(mzs, binsize):
	minMax = findMinMax(mzs)
	minmz = minMax[0] - 0.1
	maxmz = minMax[1] + 0.1
	bins = []
	quantity = math.ceil((maxmz-minmz)/binsize)
	i = 0;
	while (i < quantity):
		bins.append([i*binsize + minmz, (i+1)*binsize + minmz])
		i = i+1
	return bins


# from https://www.python-course.eu/pandas_python_binning.php
# finds the bin that a given value fits into
def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1

	
# creates a matrix that finds the bins that the m/z ratios in a spectra fall into
# if a m/z ratio of a spectra falls into a bin, then the intensity of that ratio will be placed into the bin (if not, the bin will have a 0)
# only accounts for m/z ratios with intensity values > 10
# rows (second dimension) are the bins that are either 0 or an intensity value, columns (first dimension) are the spectra
# if multiple m/z ratios of a spectra fall into a single bin, a list of intensities will be placed into the bin
# NOTE: the entry peaks[0] contains a list of identifiers for each column of the binned data (each column represents a spectra)
#	the list of identifiers is a list of strings, each string in the following format: filepath_scan#
# listIfMultMZ: optional - if true, then if there is >1 m/z in a bin, it will create a list in that bin; if false, it will add the intensities together
def create_peak_matrix(mzs, intensities, identifiers, bins, listIfMultMZ=False):
	# specBinTxt = open('binned_spectra', 'w')
	# specBinTxt.write('[[')
	# for n,i in enumerate(identifiers):
	# 	specBinTxt.write(i)
	# 	if (n < len(identifiers)-1):
	# 		specBinTxt.write(', ')
	# specBinTxt.write('], ')
	
	peaks = []
	peaks.append(identifiers)
	for i,mz in enumerate(mzs):
		temp = [0] * len(bins)
		for j,m in enumerate(mz):
			index = find_bin(m,bins)
			if (listIfMultMZ):
				if (intensities[i][j] > 10):
					if (temp[index] == 0):
						temp[index] = intensities[i][j]
					else:
						if (isinstance(temp[index], list)):
							temp[index].append(intensities[i][j])
						else:
							temp[index] = [temp[index], intensities[i][j]]
			else:
				if (intensities[i][j] > 10):
					temp[index] = temp[index] + intensities[i][j]
		peaks.append(temp)

	remove = []
	for i in range(len(peaks[1])):
		delete = True
		for j,p in enumerate(peaks):
			if (j != 0):
				if (p[i] != 0):
					delete = False
					break
		if (delete == True):
			remove.append(i)

	for j,p in enumerate(peaks):
		if (j != 0):
			for r in reversed(remove):
				if (j == 1):
					bins.pop(r)
				p.pop(r)

	# 	specBinTxt.write('[')
	# 	for n,t in enumerate(temp):
	# 		if(isinstance(t, list)):
	# 			specBinTxt.write('[')
	# 			for c,inst in enumerate(t):
	# 				specBinTxt.write(str(inst))
	# 				if (c < len(t)-1):
	# 					specBinTxt.write(', ')
	# 			specBinTxt.write(']')
	# 		else:
	# 			specBinTxt.write(str(t))
	# 		if (n < len(temp)-1):
	# 				specBinTxt.write(', ')
	# 	specBinTxt.write(']')
	# 	if (i < len(identifiers)-1):
	# 		specBinTxt.write(', ')

	# specBinTxt.write(']')
	return peaks


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

	return noisy_shaped

# Uses matplot lib and https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html to graph
# 	m/z data on a histogram; 
# Also plots the probability density function of the distribution (in red)
# Creates bins of similar dimension to create_bins()
def graph(mzs, numBins):
	mzs_od = []
	for mz in mzs:
		for m in mz:
			mzs_od.append(m)

	r = findMinMax(mzs)	
	mu = np.mean(mzs_od)
	sigma = np.std(mzs_od)

	count, bins, ignored = plt.hist(x=mzs_od, bins=numBins, density=True, range=r, histtype = 'bar', facecolor='blue')
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
	plt.ylabel('Frequency')
	plt.xlabel('M/Z Bins')
	plt.show()


# given a list of lists of m/z data, it will return the minimum and maximum m/z values in the lists
def findMinMax(mzs):
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
	return minmz, maxmz


# uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca
# reduces the number of bins by pca
def compress_Bins(filled_bins):
	filled_bins.pop(0)
	np_bins = np.array(filled_bins)
	pca = PCA(n_components=len(filled_bins[0])-1)
	compressed = pca.fit_transform(np_bins)
	print(compressed)


# for testing
def main():
	# reads mgf file and initializes lists of m/z ratios and respective intensities
	mgf_contents = read_mgf_binning(['./data/HMDB.mgf','./data/agp500.mgf','./data/agp3k.mgf'])
	mzs = mgf_contents[0]
	intensities = mgf_contents[1] 
	identifiers = mgf_contents[2]

	# adds gaussian noise to the m/z dataset (comment this line if you don't want noise)
	# mzs = create_gaussian_noise(mzs)

	# creates bins
	# min_binsize = get_min_binsize(mzs)
	bins = create_bins(mzs, 0.3)

	# prints peaks matrix
	peak_matrix = create_peak_matrix(mzs, intensities, identifiers, bins)
	print(peak_matrix)
	# compress_Bins(peak_matrix)

	# graphs histogram
	# graph(mzs, len(bins))

if __name__ == "__main__":
	main()
