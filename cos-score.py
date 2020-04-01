import pyteomics
import numpy as np
from pyteomics import mgf

mzs = []
intensities = []

def main():
	with mgf.MGF('HMDB.mgf') as reader:
	    for spectrum in reader:
	    	mzs.append(spectrum['m/z array'].tolist())
	    	intensities.append(spectrum['intensity array'].tolist())	
	    print(intensities)

if __name__ == "__main__":
    main()
