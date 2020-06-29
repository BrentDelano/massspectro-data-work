# Written by Brent Delano
# 5/24/2020
# Noise filtering on .mgf/.mzxml files

import pyteomics
from pyteomics import mgf, mzxml
import math
import numpy as np
from sklearn.decomposition import PCA
import pickle
import logging
import argparse
import binning_ms
import time

def noise_filteration(mgf='', mzxml='', method=1, min_intens=0, binsize=0, peaks_per_bin=0, filename='data/noise_filtered.mgf'):
	"""takes in either .mgf files or an .mzxml file (reads only .mgf if both), reads it, applies noise filtering techniques, then creates a .mgf file of the updated mzs and intensities
		
		Args:
			mgf: filepath to a .mgf file or list of filepaths
			mzxml: filepath to a .mzxml file or list of filepaths
			method:
				if method==0: set_min_intens()
					min_intens: minimum intensity cutoff - must initialize if method==0
				if method==1: (default): choose_top_intensities()
					binsize: size of bins for which spectra fall into - must initialize if method==1
					peaks_per_bin: number of spectra to keep in each bin - must initialize if method==1
				if method==2: create_gaussian_noise()
				if method==3: pca_compression()
					binsize: size of bins for which spectra fall into - must initialize if method==3
			filename: .mgf filename to which new spectra are placed into

		Returns:
			removed: number of removed spectra due to filtration
			affected: number of affected affected due to filtration
			noise_filteration_count.Log: a .Log file reporting the number of affected spectra and removed peaks (only if method==0 or method==1)
	"""
	start_time = time.time()

	data = []
	if not mgf:
		if not mzxml:
			raise ValueError('Pass in a file path for mgf or mzxml')
		else:
			data = binning_ms.read_mzxml(mzxml)
	else:
		data = binning_ms.read_mgf_binning(mgf)
	mzs, intensities = data[0], data[1]

	remaff = False
	removed = 0
	affected = 0
	compressed = []
	if method == 0:
		remaff = True
		removed, affected = set_min_intens(mzs, intensities, min_intens)
	elif method == 1:
		remaff = True
		if binsize == 0 or peaks_per_bin == 0:
			raise ValueError('Pass in values for binsize and peaks_per_bin')
		ctp = choose_top_intensities(mzs, intensities, binsize, peaks_per_bin)
		removed, affected = ctp[0], ctp[1]
	elif method == 2:
		return create_gaussian_noise(mzs)
	elif method == 3:
		remaff = True
		identifiers = ['irr']
		removed, affected = pca_compression(mzs, intensities, identifiers, 95, binsize, False)
	else:
		return -1

	write_to_mgf(mgf, mzxml, mzs, intensities, filename)
	if remaff:
		logging.basicConfig(filename='filtration_tests/noise_filtration_tests.Log', level=logging.DEBUG, filemode='a+')
		log = logging.getLogger()
		message = 'Test 9: pca_compression() (mgf = agp500.mgf, binsize = 10)\nNumber of Removed Spectra: %s \nNumber of Affected Spectra: %s \nTime it took to complete: %s s' % (removed, affected, time.time()-start_time)
		log.info(message)

	return removed, affected


def write_to_mgf(mgfFile='', mzxmlFile='', mzs=[], intensities=[], filename=''):
	"""writes data to a specified .mgf file

		Args:
			mgfFile: original .mgf file which unspecified data is copied from (can be a list of files)
			mzxmlFile: original .mzxml file which unspecified data is copied from (can be a list of files) (DO NOT SET BOTH mgfFile AND mzxmlFile)
			mzs: 2D list of m/z ratios to be copied to new file
			intensities: 2D list of intensities to be copied to new file
			filename: filepath to create new .mgf file
			removed: if 0, then no spectra were removed; if not 0, then a list of removed indices
	"""
	spectrum = []
	if not mzxmlFile:
		count = 0
		for mgf_n in mgfFile:
			with mgf.MGF(mgf_n) as reader:
				for spectra in reader:
					if mzs[count]:
						s = spectra.copy()
						s['m/z array'] = mzs[count]
						s['intensity array'] = intensities[count]
						spectrum.append(s)
					count += 1
	else:
		count = 0
		for mzxml_n in mzxmlFile:
			with mzxml.read(mzxml_n) as reader:
				for spectra in reader:
					if mzs[count]:
						s = spectra.copy()
						s['m/z array'] = mzs[count]
						s['intensity array'] = intensities[count]
						spectrum.append(s)
					count += 1

	mgf.write(spectra=spectrum, output=filename)


def set_min_intens(mzs, intensities, min_intens):
	"""removes any peaks with an intensity less than min_intens

		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			min_intens: minimum intensity to be kept

		Returns:
			removed_peaks: number of removed peaks
	"""
	poppers = []
	removed_peaks = 0
	affected_spectra = 0
	for i,spec in enumerate(intensities):
		affected = False
		for j,intens in enumerate(spec):
			if intens < min_intens:
				poppers.append([i,j])
				removed_peaks += 1
				affected = True
		if affected:
			affected_spectra += 1
	for p in reversed(poppers):
		mzs[p[0]].pop(p[1])
		intensities[p[0]].pop(p[1])
	return removed_peaks, affected_spectra


# returns: [0]: # of removed spectra, [1]: # of affected spectra
def choose_top_intensities(mzs, intensities, binsize, peaks_per_bin):
	"""creates bins of width binsize of the m/z data for each spectra; sorts through peaks of each spectra and only keeps the k(peaks_per_bin) largest intensities of each bin
		
		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			binsize: width of bins for which m/z data will fall into
			peaks_per_bin: number of peaks to be kept within each bin

		Returns:
			[0]: number of removed peaks
			[1]: number of affected spectra
	"""
	poppers = []
	bins = binning_ms.create_bins(mzs, binsize)
	for spec_num,spec in enumerate(mzs):
		bs = []
		for m in spec:
			bs.append(binning_ms.find_bin(m, bins))
		done = []
		for j,b in enumerate(bs):
			if b not in done or j == 0:
				done.append(b)
				temp = [j]
				for i,s in enumerate(bs):
					if i > j and b == s:
						temp.append(i)
				intens_temp = []
				for t in temp:
					intens_temp.append(intensities[spec_num][t])
				int_inds = []
				if peaks_per_bin < len(intens_temp):
					rem = len(intens_temp) - peaks_per_bin
					int_inds = np.argsort(intens_temp)[:rem] # indices of the peaks_per_bin smallest elements
				indcs = []
				for ind in int_inds:
					indcs.append(temp[ind])
				indcs.sort()
				for ind in indcs:
					poppers.append([spec_num, ind])

	removed = len(poppers)
	affected = []
	for c,p in enumerate(reversed(poppers)):
		if c == 0:
			affected.append(p[0])
		else:
			if p[0] not in affected:
				affected.append(p[0])
		mzs[p[0]].pop(p[1])
		intensities[p[0]].pop(p[1])
	return removed, len(affected)


def pca_compression(mzs, intensities, identifiers, removalperc, binsize=0.3, only95var=True):
	"""uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca
	    reduces the number of bins by pca (only keeps the componenets that explain 95% of the variance)

	    Args: 
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			binsize: width of bins for which m/z data will fall into
			only95var: only keep componenets that explain 95% of the variance if true

		Returns:
			compressed[0]: compressed matrix
			compressed[1]: loadings
			compressed[2]: explained variances
	"""
	bins = binning_ms.create_bins(mzs, binsize)
	peaks = binning_ms.create_peak_matrix(mzs=mzs, intensities=intensities, bins=bins, identifiers=identifiers, minIntens=0)[0]

	compressed = []
	if only95var:
		compressed = binning_ms.compress_bins_sml(peaks)
	else:
		compressed = binning_ms.compress_bins(peaks)

	removal_indcs = remove_smallest_loadsxvar(compressed[1], compressed[2], removalperc)
	affected_spectra = 0;
	removed_peaks = 0;
	for i,mz in enumerate(mzs):
		affected = False
		for j in reversed(range(len(mz))):
			bin = binning_ms.find_bin(mz[j], bins)
			if bin in removal_indcs:
				mzs[i].pop(j)
				intensities[i].pop(j)
				removed_peaks += 1
				affected = True
		if affected:
			affected_spectra += 1

	return removed_peaks, affected_spectra


def create_gaussian_noise(mzs):
	"""adds Gaussian noise to the dataset

		Args:
			mzs: a list of lists of the m/z ratios of the spectra

		Returns:
			noisy_shaped: data with Guassian noise implemented
	"""
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


def remove_smallest_loadsxvar(loadings, expvars, removalperc):
	"""further removes noise from pca by taking the sum of the loadings * variances from pca and removes the lowest removalperc, rounded down

	    Args: 
			loadings: 2x2 list of loadings from pca
			expvars: list of explained variances from pca
			removalperc: percentage of sum of the loadings * variances to remove, rounded down (between 0 and 100)
	"""
	loadsxvar = []
	for i,l in enumerate(loadings):
		absload = []
		for j in l:
			absload.append(abs(j))
		loadsxvar.append(expvars[i] * sum(absload))

	removal_count = int(removalperc * 0.01 * len(loadsxvar))
	lv = np.array(loadsxvar)
	idcs = np.argsort(lv)[:removal_count]
	idcs.sort()

	return idcs


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Filter out noise from .mgf or .mzxml data')
	parser.add_argument('-mgf', '--mgf', nargs='*', type=str, metavar='', help='.mgf filepath')
	parser.add_argument('-mzxml', '--mzxml', nargs='*', type=str, metavar='', help='.mzxml filepath (do not do .mgf and .mzxml concurrently)')
	parser.add_argument('-m', '--method', type=int, metavar='', help='0: set_min_intens(), 1: choose_top_intensities() 2: create_gaussian_noise() 3: pca_compression()')
	parser.add_argument('-mi', '--min_intens', type=float, metavar='', help='minimum intensity cutoff - must initialize if method==0')
	parser.add_argument('-b', '--binsize', type=float, metavar='', help='size of bins for which spectra fall into - must initialize if method==1')
	parser.add_argument('-ppb', '--peaks_per_bin', type=int, metavar='', help='number of spectra to keep in each bin - must initialize if method==1')
	parser.add_argument('-f', '--filename', type=str, metavar='', help='.mgf filename to which new spectra are placed into')
	args = parser.parse_args()

	noise_filteration(args.mgf, args.mzxml, args.method, args.min_intens, args.binsize, args.peaks_per_bin, args.filename)
