import pyteomics
import numpy as np
from pyteomics import mgf

mzs = []
intensities = []

def main():
	with mgf.MGF('HMDB.mgf') as reader:
	    for i, spectrum in enumerate(reader, start=1):
	    	mzs.append(spectrum['m/z array'])
	    	intensities.append(spectrum['intensity array'])	
	    print(intensities)

if __name__ == "__main__":
    main()
