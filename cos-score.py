import pyteomics
import numpy as np
from pyteomics import mgf

# 2D lists to hold m/z ratios and intensities
mzs = []
intensities = []
spectra = []

def main():
	with mgf.MGF('HMDB.mgf') as reader:
	    for i, spectrum in enumerate(reader,start=0):
	   		spectra.append([spectrum['intensity array'].tolist(), spectrum['m/z array'].tolist()])
	print(spectra[0])

if __name__ == "__main__":
    main()
