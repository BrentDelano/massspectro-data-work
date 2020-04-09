# Written by Brent Delano
# 4/7/2020
# Uses the cos-score.py and binning-ms.py files to find similarities between spectra

import pyteomics
from pyteomics import mgf
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './GNPS_Workflows/shared_code')
import spectrum_alignment
import cos_score
import binning_ms

# reads in mgf files that have been passed in as a list of strings of file paths
def read_mgfs(mgfs):
	cosine_mgf = cos_score.read_mgf_cosine(mgfs)
	binning_mgf = binning_ms.read_mgf_binning(mgfs)
	spectra, masses, mzs, intensities = cosine_mgf[0], cosine_mgf[1], binning_mgf[0], binning_mgf[1]
	return spectra, masses, mzs, intensities


# for testing
def main():
	mgf_data = read_mgfs(['./data/HMDB.mgf', './data/agp500.mgf', './data/agp3k.mgf'])
	print(mgf_data[3])

if __name__ == "__main__":
	main()
