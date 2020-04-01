import pyteomics
import numpy as np
from pyteomics import mgf

# a = pyteomics.mgf.IndexedMGF(source='HMDB.mgf')
d = np.dtype([('m/z array', np.int8), ('intensity array', np.int16), ('charge array', np.int16)])
print(pyteomics.mgf.read(source='HMDB.mgf', use_header=False, convert_arrays=0, read_charges=False, dtype=d))
# print(pyteomics.mgf.get_spectrum(source='HMDB.mgf', title='MS/MS scan at 1.535 min with Intensity: 604.0'))

# with pyteomics.mgf.read("HMDB.mgf") as reader:
# 	pyteomics.auxiliary.print_tree(next(reader))