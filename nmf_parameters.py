import numpy as np
import pandas as pd
import sys
import nimfa
import time
import scipy.sparse
import fast_binner


mgf_data = sys.path[0] + "/data/agp3k.mgf"

output = "output_measurements.csv"
f = open(output, "w")
f.write("Bin Size,Min Intensity,Num Rows,Rank,Sparseness Basis Vector, Sparseness Coefficient,RSS,Evar,Cophenetic\n")
f.close()

rank_range = np.concatenate(([2,5,10], np.arange(30,70,10)))
output_params = ["sparseness", "rss", "evar", "cophenetic"]
intensity_range = np.arange(0.05, 0.25, 0.05)
bin_size = 0.05

for x in intensity_range:
    input_data, bins, scan_names = fast_binner.bin_sparse_dok(mgf_data, verbose = True, bin_size = bin_size, output_file = "agp3k.mgf_matrix.pkl")
    input_data = input_data.T
    input_data,bins = fast_binner.row_filter_intensity(input_data,bins, threshold=x)
    nmf_model = nimfa.Nmf(input_data)

    start = time.time()
    ranks = nmf_model.estimate_rank(rank_range, n_run=2, what=output_params)

    f = open(output, "a")

    for t in ranks:
        measures = ranks[t]
        sparseness = measures["sparseness"]
        rss = measures["rss"]
        evar = measures["evar"]
        cophenetic = measures["cophenetic"]
        f.write(str(bin_size) + "," + str(x) + "," + str(input_data.shape[0])+","+ str(t) + "," + str(sparseness[0]) + "," + str(sparseness[1]) + "," + str(rss) + "," + str(evar) + "," + str(cophenetic) +\n")

    f.close()
    end = time.time()
    print(end-start)