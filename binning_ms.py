# Written by Brent Delano
# 4/6/2020
# Takes a .mgf file and creates a matrix of where m/z ratios from spectra fit into bins of different sizes
# The columns of the matrix are the bins (bin min is the lowest m/z value, truncated, and bin max at the greatest m/z value, rounded up) 
# The rows of the matrix are the spectra
# If an m/z ratio of a spectra (m) fits into a bin (n), then the value (n,m) on the matrix will be the intensity value of that m/z ratio;
#	if not, then it will be 0
# Able to add Gaussian noise to the dataset in order to visualize the effect on bins
# Plots histogram of m/z spectra in bins with relative frequency
# Compresses bins through pca
# Uses pyteomics api (https://pyteomics.readthedocs.io/en/latest/) for reading .mgf files
# Uses https://www.python-course.eu/pandas_python_binning.php for binning functions
# Uses https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html for plots
# Uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca for compression
# Uses https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler for preprocessing scaling

import pyteomics
from pyteomics import mgf, mzxml
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA
import pickle
import argparse
import pandas as pd

def read_mgf_binning(mgfFile):
	""" reads .mgf files into executable data for these functions

		Args:
			mgfFile: list of (or just one) .mgf file paths (if more than one, then the data will be combined)

		Returns:
			mzs: a list of lists of the m/z ratios
			intensities: a list of lists (same size as mzs) of the intensities corresponding to each peak
			identifiers: a list of the identifiers for each spectra
			names: a list of the names of the spectra
			from_mgf: the mgf file each spectra comes from
			parent_masses: a list of the parent masses that each spectra correspond to
	"""
	mzs = []
	intensities = []
	identifiers = []
	from_mgf = []
	names = []
	parent_masses = []
	if isinstance(mgfFile, list):
		for count,mgf_n in enumerate(mgfFile):
			with mgf.MGF(mgf_n) as reader:
				for j,spectrum in enumerate(reader):
					mzs.append(spectrum['m/z array'].tolist())
					intensities.append(spectrum['intensity array'].tolist())
					identifiers.append(mgf_n + '_scan' + spectrum['params']['scans'])
					from_mgf.append(mgf_n)
					try:
						names.append(mgf_n + '_' + spectrum['params']['name'])
					except KeyError:
						names.append(mgf_n + '_' + 'unknown spectrum (%s spectrum #%s)' % (mgfFile[count], j))
					parent_masses.append(spectrum['params']['pepmass'][0])
			scale(mzs, intensities)
	else:
		with mgf.MGF(mgfFile) as reader:
			for j,spectrum in enumerate(reader):
				mzs.append(spectrum['m/z array'].tolist())
				intensities.append(spectrum['intensity array'].tolist())
				identifiers.append(mgfFile + '_scan' + spectrum['params']['scans'])
				from_mgf.append(mgfFile)
				try:
					names.append(mgfFile + '_' + spectrum['params']['name'])
				except KeyError:
					names.append(mgfFile + '_' + 'unknown spectrum (%s spectrum #%s)' % (mgfFile, j))
				parent_masses.append(spectrum['params']['pepmass'][0])
		scale(mzs, intensities)

	return mzs, intensities, identifiers, names, from_mgf, parent_masses


def read_mzxml(mzxmlFile):
	""" reads .mgf files into executable data for these functions

		Args:
			mzxmlFile: list of (or just one) .mzxml file paths (if more than one, then the data will be combined)

		Returns:
			mzs: a list of lists of the m/z ratios
			intensities: a list of lists (same size as mzs) of the intensities corresponding to each peak
			identifiers: a list of the identifiers for each spectra
			names: a list of the names of the spectra
			from_mgf: the mgf file each spectra comes from
			parent_masses: a list of the parent masses that each spectra correspond to
	"""
	mzs = []
	intensities = []
	identifiers = []
	from_mgf = []
	names = []
	parent_masses = []
	if isinstance(mzxmlFile, list):
		for count,mzxml_n in enumerate(mzxmlFile):
			with mzxml.read(mzxml_n) as reader:
				for j,spectrum in enumerate(reader):
					mzs.append(spectrum['m/z array'].tolist())
					intensities.append(spectrum['intensity array'].tolist())
					identifiers.append(mzxml_n + '_scan' + spectrum['params']['scans'])
					from_mgf.append(mzxml_n)
					try:
						names.append(mzxml_n + '_' + spectrum['params']['name'])
					except KeyError:
						names.append(mzxml_n + '_' + 'unknown spectrum (%s spectrum #%s)' % (mgfFile, j))
					parent_masses.append(spectrum['params']['pepmass'][0])
			scale(mzs, intensities)
	else:
		with mzxml.read(mzxmlFile) as reader: 
			for j,spectrum in enumerate(reader):
				mzs.append(spectrum['m/z array'].tolist())
				intensities.append(spectrum['intensity array'].tolist())
				identifiers.append(mzxmlFile + '_scan' + spectrum['params']['scans'])
				from_mgf.append(mzxmlFile)
				try:
					names.append(mzxmlFile + '_' + spectrum['params']['name'])
				except KeyError:
					names.append(mzxmlFile + '_' + 'unknown spectrum (%s spectrum #%s)' % (mgfFile[count], j))
				parent_masses.append(spectrum['params']['pepmass'][0])
		scale(mzs, intensities)

		return mzs, intensities, identifiers, names, from_mgf, parent_masses


def rmv_zero_intensities(mzs, intensities):
	""" removes all peaks that have intensity = 0

		Args:
			mzs: a list of lists of the m/z ratios
			intensities: a list of lists (same size as mzs) of the intensities corresponding to each peak
	"""
	poppers = []
	for n,intens in enumerate(intensities):
		for m,i in enumerate(intens):
			if i == 0:
				poppers.append([n, m])
	poppers.reverse()
	for p in poppers:
		mzs[p[0]].pop(p[1])
		intensities[p[0]].pop(p[1])


def scale(mzs, intensities):
	""" divides all the intensities by the largest intensity such that they range on a scale from 0-1

		Args:
			mzs: a list of lists of the m/z ratios
			intensities: a list of lists (same size as mzs) of the intensities corresponding to each peak
	"""
	rmv_zero_intensities(mzs, intensities)
	intens_1d = []
	for intens in intensities:
		if intens:
			for i in intens:
				intens_1d.append(i)
	intens_1d = np.array(intens_1d)
	max_int = max(intens_1d)
	for j,intens in enumerate(intensities):
		if intens:
			for k,i in enumerate(intens):
				intensities[j][k] = intensities[j][k]/max_int


def get_min_binsize(mzs):
	""" finds the maximum bin size such that each m/z ratio within a spectra will fall into its own bin

		Args:
			mzs: a list of lists of the m/z ratios

		Returns:
			max: maximum bin size such that each m/z ratio within a spectra will fall into its own bin
	"""
	max = 0
	for spec in mzs:
		if isinstance(spec, list):
			temp = copy.copy(spec)
			temp.sort()
			if max == 0:
				max = temp[1]-temp[0]
			for n,mz in enumerate(temp):
				if n < len(temp) - 1:
					diff = abs(mz - temp[n+1])
					if diff < max:
						max = diff
		else:
			if isinstance(mzs, list):
				for j,m in enumerate(mzs):
					if j == 0:
						max = mzs[j+1] - m
					else:
						if j < len(mzs) - 1:
 							diff = mzs[j+1] - m
 							if diff < max:
 								max = diff
				break
			else:
				raise TypeError('mzs should either be a 1D or 2D list')
	return max


def create_bins(mzs, binsize):
	""" creates a list of lists of bins of width binsize

		Args:
			mzs: a list of lists of the m/z ratios
			binsize: bin width

		Returns:
			bins: list of tuples (in R2) of width of size binsize
	"""
	if binsize <= 0:
		raise ValueError('binsize should be >= 0')
	elif isinstance(mzs, list):
		minmz = 0
		maxmz = 0
		if isinstance(mzs[0], list):
			minMax = findMinMax(mzs)
			minmz = minMax[0] - 0.1
			maxmz = minMax[1] + 0.1
		else:
			minmz = min(mzs)
			maxmz = max(mzs)
		bins = []
		quantity = math.ceil((maxmz-minmz)/binsize)
		i = 0;
		while (i < quantity):
			bins.append([i*binsize + minmz, (i+1)*binsize + minmz])
			i = i+1
		return bins
	else:	
		raise TypeError('mzs should either be a 1D or 2D list')


def findMinMax(mzs):
	""" finds the minimum and maximum m/z values in the lists

		Args:
			mzs: a list of lists of the m/z ratios

		Returns:
			minmz: smallest m/z value in mzs
			maxmz: largest m/z value in mzs
	"""
	minmz = 0
	maxmz = 0
	for i,mz in enumerate(mzs):
		if i == 0:
			for check in mzs:
				if check:
					minmz = check[0]
					maxmz = check[0]
					break
		if mz:
			for m in mz:
				if m < minmz:
					minmz = m
				if m > maxmz:
					maxmz = m
	return minmz, maxmz


def find_bin(value, bins):
	""" finds the corresponding bin index of bins that value falls into

		Args:
			value: the m/z value
			bins: the bins (as specified by create_bins)

		Returns:
			i: the index of the bin in bins that value falls into
	"""
	if isinstance(bins, list):
		if isinstance(bins[0], list):
		    for i in range(0, len(bins)):
		        if bins[i][0] <= value < bins[i][1]:
		            return i
		    raise ValueError('Value does not fall into bins')
		else:
			raise TypeError('Bins should be a 2D list')
	else:
		raise TypeError('Bins should be a 2D list')


def create_peak_matrix(mzs, intensities, bins, listIfMultMZ=False, minIntens=0, maxIntens=0):
	""" creates a matrix that finds the bins that the m/z ratios in a spectra fall into
		if a m/z ratio of a spectra falls into a bin, then the intensity of that ratio will be placed into the bin (if not, the bin will have a 0)
		columns (second dimension) correspond to the bins that hold either 0 or an intensity value, rows (first dimension) are the individual spectra
		if multiple m/z ratios of a spectra fall into a single bin, the intensities will be either summed together in that bin, or listed (if listIfMultMZ==True)

		Args:
			mzs: a list of lists of the m/z ratios
			intensities: a list of lists (same size as mzs) of the intensities corresponding to each peak
			listIfMultMZ: optional - if true, then if there is >1 m/z in a bin, it will create a list in that bin; if false, it will add the intensities together
			minIntens: optional - minimum intensity threshold level to add to peak matrix (default is 0)
			maxIntens: optional - maximum intensity threshold level to add to peak matrix (default is 0, which means that there is no max)

		Returns:
			peaks: peaks matrix (as specified above)
			blockedIntens: number of intensities blocked by minIntens and maxIntens
	"""
	if isinstance(mzs, list) and isinstance(intensities, list):
		if isinstance(mzs[0], list) and isinstance(intensities[0], list):
			peaks = []
			blockedIntens = 0
			for i,mz in enumerate(mzs):
				temp = [0] * len(bins)
				for j,m in enumerate(mz):
					index = find_bin(m,bins)
					if listIfMultMZ:
						if intensities[i][j] > minIntens:
							if maxIntens == 0:
								if temp[index] == 0:
									temp[index] = intensities[i][j]
								else:
									if isinstance(temp[index], list):
										temp[index].append(intensities[i][j])
									else:
										temp[index] = [temp[index], intensities[i][j]]
							else:
								if minIntens >= maxIntens:
									raise ValueError('minIntens must be < maxIntens')
								if intensities[i][j] < maxIntens:
									if temp[index] == 0:
										temp[index] = intensities[i][j]
									else:
										if isinstance(temp[index], list):
											temp[index].append(intensities[i][j])
										else:
											temp[index] = [temp[index], intensities[i][j]]
								else:
									blockedIntens += 1
						else:
							blockedIntens += 1
					else:
						if intensities[i][j] > minIntens:
							if maxIntens == 0:
								temp[index] = temp[index] + intensities[i][j]
							else:
								if minIntens >= maxIntens:
									raise ValueError('minIntens must be < maxIntens')
								if intensities[i][j] < maxIntens:
									temp[index] = temp[index] + intensities[i][j]
								else:
									blockedIntens += 1
						else:
							blockedIntens += 1
				peaks.append(temp)

			return peaks, blockedIntens
		else: 
			raise TypeError('mzs and intensities should be 2D lists')
	else:
		raise TypeError('mzs and intensities should be 2D lists')


def compress_bins(filled_bins, n_components=0):
	""" uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca to perform pca on peaks matrix

		Args:
			filled_bins: peaks matrix as specified and return by create_peak_matrix()
			n_components: number of components to keep (default is all, if 0 < n_components < 1 then components that explain n_components*100% of variance will be kept)

		Returns:
			compressed: compressed dataset (pca.fit_transform() on peaks matrix)
			components: components/loadings (pca.components_)
			var_ratio: explained variance ratio by each component (pca.explained_variance_ratio_)
	"""
	if isinstance(filled_bins, list):
		if isinstance(filled_bins[1], list):
			if isinstance(filled_bins[0][0], str):
				filled_bins.pop(0)
			np_bins = np.array(filled_bins)
			if n_components == 0:
				pca = PCA()
			elif n_components < 1:
				pca = PCA(n_components=n_components, svd_solver='full')
			else:
				pca = PCA(n_components=n_components)
			compressed = pca.fit_transform(np_bins)
			components = pca.components_
			var_ratio = pca.explained_variance_ratio_
			return compressed, components, var_ratio
		else:
			raise TypeError('filled_bins should be a 2D list')
	else:
		raise TypeError('filled_bins should be a 2D list')


def graph_mzs(mzs, numBins):
	""" plots m/z data as a histogram (corresponding to the number of peaks that fall into numBins # of bins)
		additionally plots the PDF of the distribution (in red)

		Args:
			mzs: a list of lists of the m/z ratios
			numBins: number of bins to place m/z values into
	"""
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


def graph_pca_compression(compressed):
	""" graphs the first two dimensions of the compressed dataset

		Args:
			compressed: peaks matrix that pca has been performed on (as specified by compress_bins())
	"""
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


def graph_components(components):
	""" graphs the directions of maximum variance from pca on peaks matrix

		Args:
			components: components/loadings (pca.components_)
	"""
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


def graph_scree_plot_variance(variance_ratio):
	""" creates a scree plot of the variance for all of the components from pca on peaks matrix

		Args:
			variance_ratio: explained variance ratio by each component (pca.explained_variance_ratio_)
	"""
	x = list(range(1, len(variance_ratio) + 1))
	plt.plot(x, variance_ratio, '-o')
	plt.ylabel('Variance')
	plt.xlabel('Bins')
	plt.title('Variance Explained by Each Bin')
	plt.show()


def graph_loadings_by_variance(components, variance_ratio):
	""" makes a bar chart and a histogram of abs(loadings)*variance for all components of pca on peaks matrix

		Args:
			components: components/loadings (pca.components_)
			variance_ratio: explained variance ratio by each component (pca.explained_variance_ratio_)
	"""
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


def graph_loadsxvar_mostvar(components, variance_ratio):
	""" scatter plot of abs(loadings)*variance for the axis that explain 95% of the variance for pca on the peaks matrix

		Args:
			components: components/loadings (pca.components_)
			variance_ratio: explained variance ratio by each component (pca.explained_variance_ratio_)
	"""
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


def graph_bins_vs_intens(binned_peaks):
	""" makes a bar chart and a histogram of the sum of the intensities in each bin for all spectra for the peaks matrix

		Args:
			binned_peaks: peaks matrix as specified and returned by create_peaks_matrix()
	"""
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



# # for testing
# def main():
	# # reads mgf file and initializes lists of m/z ratios and respective intensities
	# mgf_contents = read_mgf_binning(['./tests/test1.mgf', './tests/test2.mgf', './tests/test3.mgf'])
	# bins = create_bins(mgf_contents[0], 40)
	# test = create_peak_matrix(mgf_contents[0], mgf_contents[1], mgf_contents[2], bins, listIfMultMZ=False, minIntens=5, maxIntens=0)
	# print(compress_bins_sml(test[0]))

	# # reads the mzxml file and initializes lists of m/z ratios and respective intensities
	# mzxml_contents = read_mzxml('./data/000020661_RG2_01_5517.mzXML')
	# mzs = mzxml_contents[0]
	# intensities = mzxml_contents[1]
	# identifiers = mzxml_contents[2]

	# # adds gaussian noise to the m/z dataset (comment this line if you don't want noise)
	# mzs = create_gaussian_noise(mzs)

	# # creates bins
	# mgf_stuff = read_mgf_binning('./data/agp500.mgf')
	# mzs = mgf_stuff[0]
	# intensities = mgf_stuff[1]
	# bins = create_bins(mzs, 0.3)
	# peak_matrix = create_peak_matrix(mzs, intensities, bins)[0]

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

	# # opens the pickled .mzxml uncompressed data
	# pkl_data = open('binned_ms_mzxml.pkl', 'rb')
	# binned_peaks_mzxml = pickle.load(pkl_data)
	# pkl_data.close()

	# # uncompressed data plots
	# graph_bins_vs_intens(binned_peaks_mzxml)

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
	# main()
	parser = argparse.ArgumentParser(description='Various binning functions for mgfs')
	parser.add_argument('-mgf', '--mgf', nargs='*', type=str, metavar='', help='.mgf filepath')
	parser.add_argument('-mzxml', '--mzxml', nargs='*', type=str, metavar='', help='.mzxml filepath (do not do .mgf and .mzxml concurrently)')
	parser.add_argument('-b', '--binsize', type=float, metavar='', help='size of bins for which spectra fall into')
	parser.add_argument('-rh', '--row_header', type=int, metavar='', help='if 0, then filename_scan# will be used to label each spectra. if 1, then the spectra name will be used')
	parser.add_argument('-f', '--filename', type=str, metavar='', help='filepath to output data to')
	args = parser.parse_args()

	data = []
	if args.mgf:
		data = read_mgf_binning(args.mgf)
	else:
		data = read_mzxml_binning(args.mzxml)
	bins = create_bins(data[0], args.binsize)
	peaks = create_peak_matrix(data[0], data[1], bins)[0]
	
	low_b = [b[0] for b in bins]
	for i,l in enumerate(low_b):
		low_b[i] = 'Bin Lower Bound: %s' % str(l)

	if not args.row_header == 1:
		df = pd.DataFrame(data=peaks, columns=low_b, index=data[2])
	else:
		df = pd.DataFrame(data=peaks, columns=low_b, index=data[3])
	df.to_csv(args.filename)