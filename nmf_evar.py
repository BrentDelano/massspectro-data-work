import numpy as np
import sys
import nimfa
import time
import fast_binner

mgf_data = sys.path[0] + "/data/concatenated_data.mgf"

input_data, bins, scan_names = fast_binner.bin_sparse_dok(mgf_data, verbose = True, bin_size = bin_size, output_file = "agp3k.mgf_matrix.pkl")

start = time.time()
nmf_model = nimfa.Nmf(input_data)
evar = nmf_model().fit.evar()
end = time.time()

print('Evar: {0}'.format(evar))
print('It took {0:0.1f} seconds to fit NMF model and calculate EVar'.format(end - start))
