import numpy as np
import sys
import nimfa
import time
import fast_binner
import pandas as pd
import pickle as pkl

mgf_data = [sys.path[0] + "/data/agp3k.mgf", sys.path[0] + "/data/MSV000082354.mgf", sys.path[0] + "/data/MSV000082374.mgf", sys.path[0] + "/data/MSV000083462.mgf", sys.path[0] + "/data/MSV000083759.mgf"]
bin_size = 0.1

specified_spectra = pd.read_csv("agp3k_for_umap_plot.csv").values

input_data, bins, scan_names = fast_binner.bin_sparse_dok(mgf_files = mgf_data, spectra_watchlist=specified_spectra, verbose = True, bin_size = bin_size)

start = time.time()
nmf_model = nimfa.Nmf(input_data)
model = nmf_model()
evar = model.fit.evar()
end = time.time()

print('Evar: {0}'.format(evar))
print('It took {0:0.1f} seconds to fit NMF model and calculate EVar'.format(end - start))

start = time.time()
output_file = "model.pkl"
pkl.dump((model.fit),open(output_file, "wb"))
end = time.time()
print('It took {0:0.1f} seconds to serialize model fit object'.format(end - start))
