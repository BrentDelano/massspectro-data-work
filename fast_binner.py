from pyteomics import mgf, mzxml
import numpy as np
from scipy.sparse import dok_matrix
import math
import time

start = time.time()
mgf_file = "data/nematode_symbionts.mgf"
min_bin = 50
max_bin = 2000
bin_size = 0.01
bins = np.arange(min_bin, max_bin, bin_size)

reader = mgf.IndexedMGF(mgf_file)
X = dok_matrix((len(bins), len(reader.index)), dtype=np.float32)


for spectrum_index, spectrum in enumerate(reader):
    if len(spectrum['m/z array']) == 0:
        continue
#    if spectrum_index > 5:
#        break
    for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
        target_bin = math.floor((mz - min_bin)/bin_size)
        X[target_bin, spectrum_index] += intensity
print("Binned in %s seconds" % (time.time()-start))
    
