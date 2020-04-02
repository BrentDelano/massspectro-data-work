import pyteomics
import numpy as np
from pyteomics import mgf

# 3D list of the spectra, holding the m/z ratios and intensities
spectra = []

def main():
	# sort through HMDB file and place data into spectra list
	with mgf.MGF('HMDB.mgf') as reader:
	    for spectrum in reader:
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
