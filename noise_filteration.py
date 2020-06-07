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
import binning_ms

def noise_filteration(mgf='', mzxml='', method=1, min_intens=0, binsize=0, peaks_per_bin=0, filename='data/noise_filtered.mgf'):
	"""takes in either .mgf files or an .mzxml file (reads only .mgf if both), reads it, applies noise filtering techniques, then creates a .mgf file of the updated mzs and intensities
		
		Args:
			mgf: filepath to a .mgf file
			mzxml: filepath to a .mzxml file
			method:
				if method==0: set_min_intens()
					min_intens: minimum intensity cutoff - must initialize if method==0
				if method==1 (default): choose_top_intensities()
					binsize: size of bins for which spectra fall into - must initialize if method==1
					peaks_per_bin: number of spectra to keep in each bin - must initialize if method==1
				if method==2: create_gaussian_noise()
			filename: .mgf filename to which new spectra are placed into

		Returns:
			mzs: 2D list of kept/altered m/z ratios of spectra
			intensities: 2D list of kept/altered intensity values corresponding to each m/z ratio
			noise_filteration_count.Log: a .Log file reporting the number of affected spectra and removed peaks (only if method==0 or method==1)
	"""

	data = []
	if not mgf:
		if not mzxml:
			raise ValueError('Either pass in a file path for mgf or mzxml')
		else:
			data = read_mzxml(mzxml)
	else:
		data = read_mgfs(mgf)
	mzs, intensities = data[0], data[1]

	remaff = False
	removed = 0
	affected = 0
	if method == 0:
		remaff = True
		removed = set_min_intens(mzs, intensities, min_intens)
	elif method == 1:
		remaff = True
		if binsize == 0 or peaks_per_bin == 0:
			raise ValueError('Pass in values for binsize and peaks_per_bin')
		ctp = choose_top_intensities(mzs, intensities, binsize, peaks_per_bin)
		removed, affected = ctp[0], ctp[1]
	elif method == 2:
		return create_gaussian_noise(mzs)
	else:
		return -1

	write_to_mgf(mgf, mzs, intensities, filename)
	if remaff:
		logging.basicConfig(filename='noise_filteration_count.Log', level=logging.DEBUG, filemode='w')
		log = logging.getLogger()
		message = 'Number of Removed Spectra: %s \nNumber of Affected Spectra: %s' % (removed, affected)
		log.info(message)

	return mzs, intensities


def read_mgfs(mgfs):
	"""reads in mgf files that have been passed in as a list of strings of file paths, or a single file path as a string

		Args:
			mgfs: filepath to a .mgf file, or a list of filepaths

		Returns:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
	"""
	binning_mgf = binning_ms.read_mgf_binning(mgfs)
	mzs, intensities, identifiers = binning_mgf[0], binning_mgf[1], binning_mgf[2]
	return mzs, intensities


def read_mzxml(mzxml):
	"""reads an mzxml file

		Args:
			mzxml: filepath to a .mzxml file

		Returns:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
	"""

	binning_mgf = binning_ms.read_mgf_binning(mzxml)
	mzs, intensities, identifiers = binning_mgf[0], binning_mgf[1], binning_mgf[2]
	return mzs, intensities


def write_to_mgf(mgfFile, mzs, intensities, filename):
	"""writes data to a specified .mgf file

		Args:
			mgfFile: original .mgf file which unspecified data is copied from
			mzs: 2D list of m/z ratios to be copied to new file
			intensities: 2D list of intensities to be copied to new file
			filename: filepath to create new .mgf file
	"""
	spectrum = []
	with mgf.MGF(mgfFile) as reader:
			for j, spectra in enumerate(reader):
				s = spectra.copy()
				s['m/z array'] = mzs[j]
				s['intensity array'] = intensities[j]
				spectrum.append(s)
	mgf.write(spectra=spectrum, output=filename)


def set_min_intens(mzs, intensities, min_intens):
	"""removes any peaks with an intensity less than min_intens

		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			min_intens: minimum intensity to be kept

		Returns:
			count: number of removed spectra
	"""

	poppers = []
	count = 0
	for i,spec in enumerate(intensities):
		for j,intens in enumerate(spec):
			if intens < min_intens:
				poppers.append([i,j])
				count += 1
	for p in reversed(poppers):
		mzs[p[0]].pop(p[1])
		intensities[p[0]].pop(p[1])
	return count


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


# for testing
def main():
	mgf_stuff = read_mgfs('./data/HMDB.mgf')
	mzs, intensities = mgf_stuff[0], mgf_stuff[1]
	read_mgfs('./data/HMDB.mgf')
	noise_filteration(mgf='./data/HMDB.mgf', binsize=100, peaks_per_bin=5)

if __name__ == "__main__":
	main()
