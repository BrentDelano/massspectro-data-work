# Written by Brent Delano
# 5/24/2020
# Noise filtering on .mgf/.mzxml files

import pyteomics
from pyteomics import mgf, mzxml
import math
import numpy as np
from sklearn.decomposition import PCA
import pickle
import binning_ms

# takes in either .mgf files or an .mzxml file, reads it, applies noise filtering techniques, then outputs 2D lists of m/z arrays and intensities
# if method = 0: set_min_intens() ** ALSO MUST INITIALIZE min_intens **
# if method = 1 (default): choose_top_intensities() ** ALSO MUST INITIALIZE binsize and peaks_per_bin **
# if method = 2: create_gaussian_noise()
def noise_filteration(method=1, mgfs='', mzxml='', min_intens=0, binsize=0, peaks_per_bin=0):
	mzxml_data = []
	if not mgfs:
		if not mzxml:
			return -1
		else:
			mzxml_data = read_mzxml(mzxml)
	else:
		mzxml_data = read_mgfs(mgfs)
	mzs, intensities = mzxml_data[0], mzxml_data[1]

	if method = 0:
		set_min_intens(mzs, intensities, min_intens)
	elif method = 1:
		choose_top_intensities(mzs, intensities, binsize, peaks_per_bin)
	elif method = 2:
		return create_gaussian_noise(mzs)
	else
		return -1

	return mzs, intensities


# reads in mgf files that have been passed in as a list of strings of file paths, or a single file path as a string
# returns three lists: 
#	mzs: a list of lists of the m/z ratios of the spectra
#	intensities: a list of lists of the intensities of the spectra
# 	identifiers: identifiers for the file and scan #
def read_mgfs(mgfs):
	binning_mgf = binning_ms.read_mgf_binning(mgfs)
	mzs, intensities, identifiers = binning_mgf[0], binning_mgf[1], binning_mgf[2]
	return mzs, intensities


# reads in an mzxml file
# returns three lists: 
#	mzs: a list of lists of the m/z ratios of the spectra
#	intensities: a list of lists of the intensities of the spectra
# 	identifiers: identifiers for the file and scan #
def read_mzxml(mzxml):
	binning_mgf = binning_ms.read_mgf_binning(mzxml)
	mzs, intensities, identifiers = binning_mgf[0], binning_mgf[1], binning_mgf[2]
	return mzs, intensities


# removes any peaks with an intensity less than min_intens
# returns a count of the number of peaks removed
def set_min_intens(mzs, intensities, min_intens):
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


# creates bins of width binsize of the m/z data for each spectra; sorts through peaks of each spectra and only keeps the k(peaks_per_bin) largest intensities of each bin
# returns: [0]: # of removed spectra, [1]: # of affected spectra
def choose_top_intensities(mzs, intensities, binsize, peaks_per_bin):
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


# for testing
def main():
	mgf_stuff = read_mgfs('./data/HMDB.mgf')
	mzs, intensities = mgf_stuff[0], mgf_stuff[1]

if __name__ == "__main__":
	main()