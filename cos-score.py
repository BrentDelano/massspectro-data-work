import pyteomics
import numpy as np
from pyteomics import mgf, auxiliary

with mgf.MGF('HMDB.mgf') as reader:
    for spectrum in reader:
    	print(spectrum['params'])
