# Written by Brent Delano
# 5/24/2020
# Noise filtering on .mgf/.mzxml files

import pyteomics
from pyteomics import mgf, mzxml
import numpy as np
from sklearn.decomposition import PCA
import logging
import argparse
import binning_ms
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import sys
sys.path.insert(1, './jackstraw/jackstraw')
import jackstraw
from statsmodels.stats.multitest import multipletests

def noise_filteration(mgf='', mzxml='', method=1, min_intens=0, binsize=0, binsizes=[], removalperc=50, n_components=0.95, peaks_per_bin=0, rank=10, repetitions=10, within=5, row_header=0, mgf_out_filename='', log_out_filename='', csv_out_filename=''):
	""" takes in either .mgf files or an .mzxml file (reads only .mgf if both), reads it, applies noise filtering techniques, then creates a .mgf file of the updated mzs and intensities
		
		Args:
			mgf: filepath to a .mgf file or list of filepaths
			mzxml: filepath to a .mzxml file or list of filepaths
			method:
				if method==0: set_min_intens()
					min_intens: minimum intensity cutoff - must initialize if method==0
				if method==1: (default): greatest_peaks_in_windows()
					binsize: size of bins for which spectra fall into - must initialize if method==1
					peaks_per_bin: number of spectra to keep in each bin - must initialize if method==1
				if method==2: create_gaussian_noise()
				if method==3: pca_compression()
					binsize: size of bins for which spectra fall into - must initialize if method==3
					removalperc: percentage of loadsxvar to remove
					n_components: number of components to keep (see pca_compression for further explanation on possible inputs) - must initialize if method==3
				if method==4: jackstraw_method()
					binsize: size of bins for which spectra fall into - must initialize if method==4
					rank: must initialize if method==4
					repetitions: number of permutations - must initialize if method==4
				if method==5: mz_near_parentmass()
					within: counted masses will be within "within" of parent_masses
				if method==6: create_peak_matrix() (see binning_ms.py for full specifications)
					binsize: size of bins for which spectra fal into - must initialize if method==6
					csv_out_filename: csv filepath to output peaks matrix to - must initialize if method==6
					row_header: if 0, then filename_scan# will be used to label each spectra. if 1, then the spectra name will be used
				if method==7: pca_variance_ratios_for_binsizes()
					binsizes: list of sizes of bins for which spectra fal into - must initialize if method==7
					csv_out_filename: csv filepath to output variances to - must initialize if method==7
					n_components: number of components to keep (see pca_compression for further explanation on possible inputs) - must initialize if method==7
				if method is not 0-7: no filtration
			mgf_out_filename: .mgf filename to which new spectra are placed into
			log_out_filename: .log filename to which output data is placed into (note: do not append to existing log files)

		Returns:
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
	mzs, intensities, identifiers, names, from_mgf, parent_masses = data

	num_peaks = 0
	for i in intensities:
		num_peaks += len(i)
	num_spectra = len(intensities)

	logit = False
	removed = 0
	affected = 0
	victims = []
	compressed = []
	near = []
	if method == 0:
		logit = True
		removed, affected, victims = set_min_intens(mzs, intensities, min_intens, names, from_mgf)
	elif method == 1:
		logit = True
		if binsize == 0 or peaks_per_bin == 0:
			raise ValueError('Pass in values for binsize and peaks_per_bin')
		ctp = greatest_peaks_in_windows(mzs, intensities, binsize, peaks_per_bin, names, identifiers)
		removed, affected, victims = ctp[0], ctp[1], ctp[2]
	elif method == 2:
		return create_gaussian_noise(mzs)
	elif method == 3:
		logit = True
		removed, affected, victims = pca_compression(mzs, intensities, removalperc, names, identifiers, binsize, n_components)
	elif method == 4:
		jackstraw_method(mzs, intensities, binsize, identifiers, rank, repetitions)
	elif method == 5:
		logit = True
		near, perc, too_large = mz_near_parentmass(mzs, names, parent_masses, within)
	elif method == 6:
		bins = binning_ms.create_bins(mzs, binsize)
		peaks = binning_ms.create_peak_matrix(mzs, intensities, bins)[0]
		
		low_b = [b[0] for b in bins]
		for i,l in enumerate(low_b):
			low_b[i] = 'Bin Lower Bound: %s' % str(l)

		if not row_header == 1:
			df = pd.DataFrame(data=peaks, columns=low_b, index=data[2])
		else:
			df = pd.DataFrame(data=peaks, columns=low_b, index=data[3])
		df.to_csv(csv_out_filename[0])
	elif method == 7:
		pca_variance_ratios_for_binsizes(mzs, intensities, binsizes, n_components, csv_out_filename[0])
	else:
		write_to_mgf(mgf, mzxml, mzs, intensities, mgf_out_filename)

	if logit:
		logging.basicConfig(filename=log_out_filename, level=logging.DEBUG, filemode='a+')
		log = logging.getLogger()
		
		filt_method = ''
		if method == 0:
			filt_method = 'set_min_intens(min_intens=%s)' % min_intens
		elif method == 1:
			filt_method = 'greatest_peaks_in_windows(binsize=%s, peaks_per_bin=%s)' % (binsize, peaks_per_bin)
		elif method == 3:
			filt_method = 'pca_compression(binsize=%s)' % binsize
		else:
			filt_method = 'mz_near_parentmass(within=%s)' % within

		if 0 <= method <=3:
			message = '\nfiltration method;name of affected spectrum;m/z of removed peak;intensity of removed peak;.mgf file\n'
			for v in victims:
				message += '\n%s;%s;%s;%s;%s' % (filt_method, v[0], v[1], v[2], v[3])
		else:
			message = '\n;method;name of spectrum;number of m/z values within "within" of parent mass;number of m/z > parent_mass\n'
			for n in near:
				message += '\n%s;%s;%s;%s' % (filt_method, n[0], n[1], n[2])

		log.info(message)

	if mgf_out_filename:
		write_to_mgf(mgf, mzxml, mzs, intensities, mgf_out_filename)

	# if 0 <= method <= 5:
	# 	avg_removed = removed / num_spectra
	# 	x = [u'Number of\nRemoved Peaks', u'Average Removed\nPeaks per\nSpectra', u'Original Number\nof Peaks']
	# 	y = [removed, avg_removed, num_peaks]
		
	# 	fig, ax = plt.subplots()    
	# 	width = 0.75
	# 	ind = np.arange(len(y))
	# 	ax.barh(ind, y, width, color="blue")
	# 	ax.set_yticks(ind+width/2)
	# 	ax.set_yticklabels(x, minor=False)
	# 	for i, v in enumerate(y):
	# 		ax.text(v, i, str(v), color='blue', fontweight='bold')
	# 	plt.show()


def write_to_mgf(mgfFile='', mzxmlFile='', mzs=[], intensities=[], filename=''):
	""" writes data to a specified .mgf file

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


def set_min_intens(mzs, intensities, min_intens, names, identifiers):
	""" removes any peaks with an intensity less than min_intens

		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			min_intens: minimum intensity to be kept
			names: list of names of spectra
			identifiers: list of spectra identifiers

		Returns:
			[0]: number of removed peaks
			[1]: number of affected spectra
			[2]: names and values of affected spectra and removed peaks
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
	victims = []
	for p in reversed(poppers):
		victims.append([names[p[0]], mzs[p[0]][p[1]], intensities[p[0]][p[1]], identifiers[p[0]]])
		mzs[p[0]].pop(p[1])
		intensities[p[0]].pop(p[1])
	return removed_peaks, affected_spectra, reversed(victims)


# returns: [0]: # of removed spectra, [1]: # of affected spectra
def greatest_peaks_in_windows(mzs, intensities, binsize, peaks_per_bin, names, identifiers):
	""" creates bins of width binsize of the m/z data for each spectra; sorts through peaks of each spectra and only keeps the k(peaks_per_bin) largest intensities of each bin
		
		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			binsize: width of bins for which m/z data will fall into
			peaks_per_bin: number of peaks to be kept within each bin
			names: list of names of spectra
			identifiers: list of spectra identifiers

		Returns:
			[0]: number of removed peaks
			[1]: number of affected spectra
			[2]: names and values of affected spectra and removed peaks
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
	victims = []
	for c,p in enumerate(reversed(poppers)):
		if c == 0:
			affected.append(p[0])
		else:
			if p[0] not in affected:
				affected.append(p[0])
		victims.append([names[p[0]], mzs[p[0]][p[1]], intensities[p[0]][p[1]], identifiers[p[0]]])
		mzs[p[0]].pop(p[1])
		intensities[p[0]].pop(p[1])
	return removed, len(affected), reversed(victims)


def pca_compression(mzs, intensities, removalperc, names, identifiers, binsize=0.3, n_components=0.95):
	""" uses https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#examples-using-sklearn-decomposition-pca
	    reduces the number of bins by pca

	    Args: 
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			removalperc: percentage of components to remove
			names: names of the spectra (as specified by the .mgf and .mzxml reading in binning_ms.py)
			identifiers: list of spectra identifiers
			binsize: width of bins for which m/z data will fall into
			n_components: # of components to keep with pca (default is 0.95, if 0 < n_components < 1, then components that explain n_components * 100% of variance will be kept)

		Returns:
			[0]: number of removed peaks
			[1]: number of affected spectra
			[2]: names and values of affected spectra and removed peaks
	"""
	bins = binning_ms.create_bins(mzs, binsize)
	peaks = binning_ms.create_peak_matrix(mzs=mzs, intensities=intensities, bins=bins)[0]

	compressed = binning_ms.compress_bins(peaks, n_components)

	removal_indcs = remove_smallest_loadsxvar(compressed[1], compressed[2], removalperc)
	victims = []
	affected_spectra = 0
	removed_peaks = 0
	for i,mz in enumerate(mzs):
		affected = False
		vic = []
		for j in reversed(range(len(mz))):
			bin = binning_ms.find_bin(mz[j], bins)
			if bin in removal_indcs:
				vic.append([mzs[i][j], intensities[i][j]])
				mzs[i].pop(j)
				intensities[i].pop(j)
				removed_peaks += 1
				affected = True
		for v in reversed(vic):
			victims.append([names[i], v[0], v[1], identifiers[i]])
		if affected:
			affected_spectra += 1

	return removed_peaks, affected_spectra, victims


def remove_smallest_loadsxvar(loadings, expvars, removalperc):
	""" further removes noise from pca by taking the sum of the loadings * variances from pca and removes the lowest removalperc, rounded down

	    Args: 
			loadings: 2x2 list of loadings from pca
			expvars: list of explained variances from pca
			removalperc: percentage of sum of the loadings * variances to remove, rounded down (between 0 and 100)

		Returns:
			idcs: indices of bins to remove
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


def pca_variance_ratios_for_binsizes(mzs, intensities, binsizes, n_components=0.95, csv_out='pca_variance_ratio_sml.csv'):
	""" creates a csv file that shows the explained variance ratio by each of the components for various binsizes

		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			binsizes: widths of bins to tests (can be a list)
			n_components: # of components to keep with pca (default is 0.95, if 0 < n_components < 1, then components that explain n_components * 100% of variance will be kept)
			csv_out: fpath to output variance ratios to
	"""
	if not isinstance(binsizes, list):
		binsizes = [binsizes]

	var_ratios_list, headers = [], []
	for b in binsizes:
		bins = binning_ms.create_bins(mzs, b)
		peaks = binning_ms.create_peak_matrix(mzs=mzs, intensities=intensities, bins=bins)[0]

		var_ratios_list.append(binning_ms.compress_bins(peaks, n_components)[2])
		headers.append('binsize: %s' % str(b))

	df = pd.DataFrame(var_ratios_list, headers).T
	df.to_csv(csv_out)


def jackstraw_method(mzs, intensities, binsize, identifiers, rank, repetitions):
	""" jackstraw method as described here: https://arxiv.org/abs/1308.6013
		uses methods from https://github.com/idc9/jackstraw/blob/master/jackstraw/jackstraw.py
		uses https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html for statistical analysis

		Args: 
			mzs: a list of lists of the m/z ratios of the spectra
			intensities: a list of lists of the intensities of the spectra
			binsize: size of bins for which spectra fall into - must initialize if method==1
			identifiers: list of spectra identifiers
			rank: rank of matrix for jackstraw
			repetitions: number of permutations to run through

		Returns:
			pvals_raw: raw p-values
			corrected: see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
			F_obs: observed F-statistics
			F_null: null F-statistics
	"""
	bins = binning_ms.create_bins(mzs, binsize)
	peaks = binning_ms.create_peak_matrix(mzs=mzs, intensities=intensities, bins=bins)[0]
	peaks.pop(0)
	peaks = np.array(peaks)

	j = jackstraw.Jackstraw(B=repetitions)
	j.fit(peaks, 'svd', rank)
	pvals_raw = j.pvals_raw

	corrected = multipletests(pvals=pvals_raw)
	return pvals_raw, corrected, j.F_obs, j.F_null


def mz_near_parentmass(mzs, names, parent_masses, w=5):
	""" reports the number of peaks in spectra that are within w of the parent mass

		Args:
			mzs: a list of lists of the m/z ratios of the spectra
			names: list of names of spectra (len(names)==len(mzs))
			parent_masses: list of parent masses of spectra (len(parent_masses)==len(mzs))
			w: counted masses will be within w of parent_masses

		Returns:
			near: list of lists of [spectra ID, # of peaks within w of parent mass of spectra]
			perc: percentage of peaks that are within w of parent mass
			too_large: the spectra that have peaks larger than their parent mass
	"""
	near = []
	near_total = 0
	total = 0
	too_large = []
	for n,mz in enumerate(mzs):
		count = 0
		big_count = 0
		for m in mz:
			total += 1
			if m > parent_masses[n]:
				too_large.append(names[n])
				big_count += 1
			if 0 < parent_masses[n] - m < w:
				count += 1
		near.append([names[n], count, big_count])
		near_total += count
	perc = near_total/total * 100
	return near, perc, too_large


def create_gaussian_noise(mzs):
	""" adds Gaussian noise to the dataset

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


def read_log(log_file):
	""" takes in a list of log file paths and outputs data to a pandas dataframe

	    Args: 
			log_file: list of file paths to a log file
				- Format of log file: 'filtration method;name of affected spectrum;m/z of removed peak;intensity of removed peak;.mgf file'
				- First line ignored (INFO:ROOT etc.)

		Returns:
			df: pandas dataframe of data from log file
	"""
	df = pd.DataFrame()
	if isinstance(log_file, list):
		for i,f in enumerate(log_file):
			if i == 0:
				df = pd.read_csv(f, sep=';', skiprows=1)
			else:
				df = df.append(pd.read_csv(f, sep=';', skiprows=1))
	return df


def graph_removed_data(df):
	""" takes in a pandas dataframe (explained above) of data to graph (graphs a scatter of the data (x: m/z, y: intensity))

	    Args: 
			df: pandas dataframe (see read_log() for a detailed explanation of format of df)
	"""
	g = sns.FacetGrid(df, col='filtration method', hue='.mgf file')
	g.map(plt.scatter, 'm/z of removed peak', 'intensity of removed peak', alpha=.7)
	g.add_legend()
	plt.show()


def graph_pca_variance_ratios_for_binsizes(df, n_components=0):
	""" graphs the output of pca_variance_ratios_for_binsizes()

		Args:
			df: dataframe as created by pca_variance_ratios_for_binsizes()
			n_components: number of components to keep in graphs (Special cases: if n_components==0, then all components kept; if 0 < n_components < 1, then n_components * 100% of components are kept)
	"""
	x = int(np.round(np.sqrt(df.shape[1])))
	y = int(np.ceil(df.shape[1]/x))
	fig, axs = plt.subplots(y, x)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.suptitle('graph_pca_variance_ratios_for_binsizes()')

	yp, xp = 0, 0
	for name,cols in df.iteritems():
		vals = cols.to_numpy()
		vals = vals[np.logical_not(np.isnan(vals))]
		if 0 < n_components < 1:
			vals = vals[0:int(np.round(len(vals) * n_components))]
		elif n_components >= 1:
			vals = vals[0:n_components]
		if xp >= x:
			xp = 0
			yp += 1
		axs[yp][xp].scatter(x=range(1, len(vals)+1), y=vals)
		axs[yp][xp].set_title(name)
		xp += 1
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Filter out noise from .mgf or .mzxml data')
	parser.add_argument('-mgf', '--mgf', nargs='*', type=str, metavar='', help='.mgf filepath')
	parser.add_argument('-mzxml', '--mzxml', nargs='*', type=str, metavar='', help='.mzxml filepath (do not do .mgf and .mzxml concurrently)')
	parser.add_argument('-m', '--method', type=int, metavar='', help='0: set_min_intens(), 1: greatest_peaks_in_windows() 2: create_gaussian_noise() 3: pca_compression() 4: jackstraw_method(), 5: mz_near_parentmass(), 6: create_peak_matrix(), 7: pca_variance_ratios_for_binsizes(), 8: no filtration')
	parser.add_argument('-mi', '--min_intens', type=float, metavar='', help='minimum intensity cutoff - must initialize if method==0')
	parser.add_argument('-b', '--binsize', type=float, metavar='', help='size of bins for which spectra fall into - must initialize if method==1, 3, or 6')
	parser.add_argument('-bs', '--binsizes', nargs='*', type=float, metavar='', help='list of binsizes - must initialize if method==7')
	parser.add_argument('-rmp', '--removalperc', type=float, metavar='', help='percentage of loadings x variances to remove - must initialize if method==3')
	parser.add_argument('-nc', '--n_components', type=float, metavar='', help='number of components to keep (if 0, then all are kept; elif < 1, then the percentage of components that explain this percentage will be kept; else, then this number of components will be kept) - must initialize if method==3 or 7 or -csvi is initialized')
	parser.add_argument('-ppb', '--peaks_per_bin', type=int, metavar='', help='number of spectra to keep in each bin - must initialize if method==1')
	parser.add_argument('-ra', '--rank', type=int, metavar='', help='rank for jackstraw - only if method==4')
	parser.add_argument('-rep', '--repetitions', type=int, metavar='', help='number of permutations - must initialize if method==4')
	parser.add_argument('-w', '--within', type=int, metavar='', help='counted masses will be within "within" of parent_masses - must initialize if method==5')
	parser.add_argument('-rh', '--row_header', type=int, metavar='', help='if 0, then filename_scan# will be used to label each spectra. if 1, then the spectra name will be used')
	parser.add_argument('-mf', '--mgf_filename', type=str, metavar='', help='.mgf filename to which new spectra are placed into')
	parser.add_argument('-lf', '--log_filename', type=str, metavar='', help='.log filename to which output data is placed into')
	parser.add_argument('-csvf', '--csv_filename', nargs='*', type=str, metavar='', help='.csv filepath to output peaks matrix to - must initialize if method==6 or 7')
	parser.add_argument('-lfi', '--log_filename_input', nargs='*', type=str, metavar='', help='.log filepaths for which to graph data from (DO NOT INITIALIZE PREVIOUS VARIABLES)')
	parser.add_argument('-csvi', '--csv_filename_input', type=str, metavar='', help='.csv filepath (created by pca_variance_ratios_for_binsizes()) for which to graph data from (DO NOT INITIALIZE PREVIOUS VARIABLES, EXCEPT FOR n_components, will do graph_pca_variance_ratios_for_binsizes())')
	args = parser.parse_args()

	if args.log_filename_input:
		df = read_log(args.log_filename_input)
		graph_removed_data(df)
	elif args.csv_filename_input:
		df = pd.read_csv(args.csv_filename_input, index_col=0)
		graph_pca_variance_ratios_for_binsizes(df, args.n_components)
	else:
		noise_filteration(args.mgf, args.mzxml, args.method, args.min_intens, args.binsize, args.binsizes, args.removalperc, args.n_components, args.peaks_per_bin, args.rank, args.repetitions, args.within, args.row_header, args.mgf_filename, args.log_filename, args.csv_filename)
