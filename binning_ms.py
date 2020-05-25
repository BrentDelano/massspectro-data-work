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
from pyteomics import mgf, mzxml
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA
import pickle

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
				for j,spectrum in enumerate(reader):
					mzs.append(spectrum['m/z array'].tolist())
					intensities.append(spectrum['intensity array'].tolist())
					identifiers.append(mgfs_n + '_' + str(j+1))
	else:
		with mgf.MGF(mgfFile) as reader:
			for j,spectrum in enumerate(reader):
				mzs.append(spectrum['m/z array'].tolist())
				intensities.append(spectrum['intensity array'].tolist())
				identifiers.append(mgfFile + '_' + str(j+1))
	return mzs, intensities, identifiers

# takes in .mzxml file paths as a string
# outputs 3 lists of lists: the first holding the m/z ratios, the second holding a list holding the respective intensities, 
#	the third a list of identifiers
def read_mzxml(mzxmlFile):
	mzs = []
	intensities = []
	identifiers = []
	with mzxml.read(mzxmlFile) as reader: 
		for j,spectrum in enumerate(reader):
			mzs.append(spectrum['m/z array'].tolist())
			intensities.append(spectrum['intensity array'].tolist())
			identifiers.append(mzxmlFile + '_' + str(j+1))
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
# minIntens: optional - minimum intensity threshold level to add to peak matrix (default is 10)
# maxIntens: optional - maximum intensity threshold level to add to peak matrix (default is 0, which means that there is no max)
# also returns blockedIntens, which is the numeber of intensities that were filtered out by the threshold noise filtering
def create_peak_matrix(mzs, intensities, identifiers, bins, listIfMultMZ=False, minIntens=10, maxIntens=0):
	peaks = []
	peaks.append(identifiers)
	blockedIntens = 0;
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
				if (intensities[i][j] > minIntens):
					if (maxIntens == 0):
						temp[index] = temp[index] + intensities[i][j]
					else:
						if (intensities[i][j] < maxIntens):
							temp[index] = temp[index] + intensities[i][j]
						else:
							blockedIntens += 1
				else:
					blockedIntens += 1
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
	return peaks, blockedIntens


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
def compress_bins(filled_bins):
	filled_bins.pop(0)
	np_bins = np.array(filled_bins)
	pca = PCA(n_components=len(filled_bins[0])-1)
	compressed = pca.fit_transform(np_bins)
	components = pca.components_
	var_ratio = pca.explained_variance_ratio_
	return compressed, components, var_ratio


# uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca
# reduces the number of bins by pca (only keeps the componenets that explain 95% of the variance)
def compress_bins_sml(filled_bins):
	filled_bins.pop(0)
	np_bins = np.array(filled_bins)
	pca = PCA(n_components=0.95, svd_solver='full')
	compressed = pca.fit_transform(np_bins)
	components = pca.components_
	var_ratio = pca.explained_variance_ratio_
	return compressed, components, var_ratio	


# Uses matplot lib and https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html to graph
# 	m/z data on a histogram; 
# Also plots the probability density function of the distribution (in red)
# Creates bins of similar dimension to create_bins()
def graph_mzs(mzs, numBins):
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


# graphs the first two dimensions of the compressed dataset
def graph_compression(compressed):
	s1 = compressed[1]
	s2 = compressed[2]
	
	x = np.arange(len(s1))
	width = 0.45
	fig, ax = plt.subplots()
	rects1 = ax.bar(x - width/2, s1, width, label='Spectra 1')
	rects2 = ax.bar(x + width/2, s2, width, label='Spectra 2')
	
	ax.set_xlabel('Bins')
	ax.set_ylabel('Compressed Intensities')
	plt.title('Compressed Representation of First Two Spectra')
	ax.legend()
	fig.tight_layout()
	plt.show()


# graphs the directions of maximum variance
# Uses matplot lib and https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
def graph_components(components):
	comp_od = []
	for i in range(len(components[0])):
		temp_sum = 0
		for j in range(len(components)):
			temp_sum = temp_sum + components[j][i]
		comp_od.append(temp_sum)
	x = list(range(len(comp_od)))
	plt.bar(x, comp_od, 1, align='edge')
	plt.ylabel('Variance')
	plt.xlabel('Bins')
	plt.show()


# creates a scree plot of the variance for all the axis
def graph_scree_plot_variance(variance_ratio):
	x = list(range(1, len(variance_ratio) + 1))
	plt.plot(x, variance_ratio, '-o')
	plt.ylabel('Variance')
	plt.xlabel('Bins')
	plt.title('Variance Explained by Each Bin')
	plt.show()


# makes a bar chart and a histogram of abs(loadings)*variance for all axis
def graph_loadings_by_variance(components, variance_ratio):
	comp_sum = []
	for c in components:
		nsum = 0
		for j in c:
			nsum = nsum + abs(j)
		comp_sum.append(nsum)

	lbv = []
	for n,c in enumerate(comp_sum):
		lbv.append(c * variance_ratio[n])

	# x = list(range(len(lbv)))
	# plt.bar(x, lbv, 1, align='edge')
	# plt.ylabel('Loadings x Variance')
	# plt.xlabel('Bins')
	# plt.title('Loadings x Variance per Bin for First 27 Axes')
	# plt.show()

	r = [min(lbv), max(lbv)]
	plt.hist(x=lbv, bins=len(lbv), density=True, range=r, histtype='bar', facecolor='blue')
	plt.ylabel('Frequency')
	plt.xlabel('Loadings x Variance')
	plt.title('Histogram of Loadings x Variance for First 27 Axes')
	plt.show()


# makes a bar chart and a histogram of the sum of the intensities in each bin for all spectra
def graph_bins_vs_intens(binned_peaks):
	binned_peaks.pop(0)
	bin_pks_sum = []
	for i in range(len(binned_peaks[0])):
		sum = 0
		for j in binned_peaks:
			sum += j[i]
		bin_pks_sum.append(math.log10(sum))

	# x = list(range(len(bin_pks_sum)))
	# plt.bar(x, bin_pks_sum, 1, align='edge')
	# plt.ylabel('Sum of Intensities')
	# plt.xlabel('Bins/Axis')
	# plt.title('Sums of Intensities for Bins')
	# plt.show()

	r = [min(bin_pks_sum), max(bin_pks_sum)]
	n, bins, patches = plt.hist(x=bin_pks_sum, bins=50, range=r, histtype='bar', facecolor='blue')
	plt.ylabel('Frequency')
	plt.xlabel('Log10(Sum of Intensities)')
	plt.title('Histogram of Frequency of Sum of Intensities per Bin')
	plt.show()


# scatter plot of abs(loadings)*variance for the axis that explain 95% of the variance
def graph_loadsxvar_mostvar(components, variance_ratio):
	comp_sum = []
	for c in components:
		nsum = 0
		for j in c:
			nsum = nsum + abs(j)
		comp_sum.append(nsum)

	lbv = []
	for n,c in enumerate(comp_sum):
		lbv.append(c * variance_ratio[n])

	x = list(range(len(lbv)))
	plt.scatter(x, lbv)
	plt.xlabel('First 27 Axis')
	plt.ylabel('|Loadings| x Variance')
	plt.title('|Loadings| x Variance for Axis that Explain 95 Percent of Variance')
	plt.show()


# for testing
def main():
	# # reads mgf file and initializes lists of m/z ratios and respective intensities
	# mgf_contents = read_mgf_binning(['./data/HMDB.mgf','./data/agp500.mgf','./data/agp3k.mgf','./data/QUERT.mgf'])
	# mzs = mgf_contents[0]
	# intensities = mgf_contents[1] 
	# identifiers = mgf_contents[2]

	# # reads the mzxml file and initializes lists of m/z ratios and respective intensities
	# mzxml_contents = read_mzxml('./data/000020661_RG2_01_5517.mzXML')
	# mzs = mzxml_contents[0]
	# intensities = mzxml_contents[1]
	# identifiers = mzxml_contents[2]

	# # adds gaussian noise to the m/z dataset (comment this line if you don't want noise)
	# mzs = create_gaussian_noise(mzs)

	# # creates bins
	# bins = create_bins(mzs, 0.3)

	# # creates peaks matrix
	# peak_matrix = create_peak_matrix(mzs, intensities, identifiers, bins)[0]

	# # pickles the binned data
	# pkld_bins = open('binned_ms_mzxml.pkl', 'wb')
	# pickle.dump(peak_matrix, pkld_bins)
	# pkld_bins.close()

	# # opens the pickled .mgf uncompressed data
	# pkl_data = open('binned_ms.pkl', 'rb')
	# binned_peaks = pickle.load(pkl_data)
	# pkl_data.close()

	# opens the pickled .mzxml uncompressed data
	pkl_data = open('binned_ms_mzxml.pkl', 'rb')
	binned_peaks_mzxml = pickle.load(pkl_data)
	pkl_data.close()

	# uncompressed data plots
	graph_bins_vs_intens(binned_peaks_mzxml)

	# # compresses binned_peaks by only keeping 95% of explained variance
	# labels_sml = binned_peaks.pop(0)
	# compressed_sml = compress_bins_sml(binned_peaks)

	# # compresses binned_peaks with pca; graphs compression
	# labels = binned_peaks.pop(0)
	# compressed = compress_bins(binned_peaks)

	# # pickles the compressed data
	# pkld_bins2 = open('compressed_binned_ms.pkl', 'wb')
	# pickle.dump(compressed, pkld_bins2)
	# pkld_bins2.close()

	# # pickles the 95% of variance compressed data
	# pkld_bins3 = open('compressed_binned_ms_95var.pkl', 'wb')
	# pickle.dump(compressed_sml, pkld_bins3)
	# pkld_bins3.close()

	# # opens the pickled compressed data
	# pkl_data2 = open('compressed_binned_ms.pkl', 'rb')
	# compressed = pickle.load(pkl_data2)
	# pkl_data2.close()

	# opens the pickled compressed data (95% of variance)
	# pkl_data3 = open('compressed_binned_ms_95var.pkl', 'rb')
	# compressed_sml = pickle.load(pkl_data3)
	# pkl_data3.close()

	# # graphs of 95% of var compression
	# graph_loadsxvar_mostvar(compressed_sml[1], compressed_sml[2])

	# # graphs of compressed data
	# compr = compressed[0]
	# graph_compression(compr)
	# components = compressed[1]
	# graph_components([components[0], components[1], components[2]])
	# var_ratio = compressed[2]
	# graph_scree_plot_variance(var_ratio)
	# graph_loadings_by_variance(components[:27], var_ratio)
	# print(components)
	# print(var_ratio)
	# graphs histogram of m/z data
	# graph_mzs(mzs, len(bins))

if __name__ == "__main__":
	main()
