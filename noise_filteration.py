# Written by Brent Delano
# 5/24/2020
# Noise filtering on .mgf/.mzxml files

import pyteomics
from pyteomics import mgf, mzxml
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA
import pickle
import binning_ms

# reads in mgf files that have been passed in as a list of strings of file paths, or a single file path as a string
# returns three lists: 
#	mzs: a list of lists of the m/z ratios of the spectra
#	intensities: a list of lists of the intensities of the spectra
# 	identifiers: identifiers for the file and scan #
def read_mgfs(mgfs):
	binning_mgf = binning_ms.read_mgf_binning(mgfs)
	mzs, intensities, identifiers = binning_mgf[0], binning_mgf[1], binning_mgf[2]
	return mzs, intensities, identifiers


# reads in an mzxml file
# returns three lists: 
#	mzs: a list of lists of the m/z ratios of the spectra
#	intensities: a list of lists of the intensities of the spectra
# 	identifiers: identifiers for the file and scan #
def read_mzxml(mzxml):
	binning_mgf = binning_ms.read_mgf_binning(mzxml)
	mzs, intensities, identifiers = binning_mgf[0], binning_mgf[1], binning_mgf[2]
	return mzs, intensities, identifiers


# uses the binning_ms.py create_peak_matrix function (more documentation found there)
# uses binWidth instead of bins (uses binning_ms.create_bins() to create bins of binWidth size)
# THIS FUNCTION ALLOWS FOR THRESHOLD NOISE FILTERING THROUGH minIntens and maxIntens (default is minIntens=10, maxIntens=0)
# If maxIntens = 0, then there is no maximum threshold
def create_peak_matrix(mzs, intensities, identifiers, binWidth, listIfMultMZ=False, minIntens=10, maxIntens=0):
	bins = binning_ms.create_bins(binWidth)
	peaks = binning_ms.create_peak_matrix(mzs, intensities, identifiers, bins, minIntens)
	return peaks