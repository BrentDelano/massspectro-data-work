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
f.write("Rank,Sparseness Basis Vector, Sparseness Coefficient,RSS,Evar\n")
f.close()

rank_range = np.concatenate(([2,5,10], np.arange(30,70,10)))
output_params = ["sparseness", "rss", "evar", "cophenetic"]

input_data, bins, scan_names = fast_binner.bin_sparse_dok(mgf_data, verbose = True, output_file = "agp3k.mgf_matrix.pkl")
nmf_model = nimfa.Nmf(input_data)
ranks = nmf_model.estimate_rank(rank_range, what=output_params)

f = open(output, "a")

for t in ranks:
    measures = ranks[t]
    sparseness = measures["sparseness"]
    rss = measures["rss"]
    evar = measures["evar"]
    f.write(str(t) + "," + str(sparseness[0]) + "," + str(sparseness[1]) + "," + str(rss) + "," + str(evar) + "\n")

f.close()