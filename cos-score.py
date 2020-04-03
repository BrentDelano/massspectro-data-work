import pyteomics
import numpy as np
from pyteomics import mgf
import sys
sys.path.insert(1, '/Users/brent/Desktop/cosScore/GNPS_Workflows/shared_code')
import spectrum_alignment

# Lists to hold the spectra (m/z ratios and intensities) and masses
spectra = []
masses = []

def main():
	# sort through HMDB file and place data into spectra list
	with mgf.MGF('HMDB.mgf') as reader:
		for spectrum in reader:
			masses.append(spectrum['params']['pepmass'][0])
			spectra.append([spectrum['intensity array'].tolist(), spectrum['m/z array'].tolist()])

	# compare all spectra to each other; place cosine scores into an nxn array
	cosScores = []
	for i in spectra:
		temp = []
		for j in spectra:
			if (i == j):
				temp.append(1)
			else:
				temp.append(0)
		cosScores.append(temp)

if __name__ == "__main__":
    main()
