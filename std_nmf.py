import numpy as np
import pandas as pd
import sys
import time
import scipy.sparse
import math
import fast_binner
import matplotlib
matplotlib.use('Agg') #for plotting w/out GUI - for use on server
import matplotlib.pyplot as plt
import nimfa

mgf_data = sys.path[0] + "/data/agp3k.mgf"
bin_size = 0.01
rank = 30

output = "output_measurements.csv"
f = open(output, "w")
f.write("Bin Size,Num Rows,Standard Dev,Rank,EVar\n")
f.close()

sd_range = [10,50,100]

for threshold in sd_range:

    input_data, bins, scan_names = fast_binner.bin_sparse_dok(mgf_data, verbose = True, bin_size = bin_size, output_file = "agp3k.mgf_matrix.pkl")

    for row in input_data:
        mean = row.mean()
        squared = row.copy()
        squared.data **= 2
        variance = squared.mean() - (mean**2)
        std_dev = math.sqrt(variance)
        if std_dev > threshold:
            input_data.remove(row)
    
    nmf_model = nimfa.Nmf(input_data, rank=rank)
    evar = nmf_model().evar()

    f = open(output, "a")
    f.write(str(bin_size) + "," str(input_data.shape[0])+"," + str(threshold) + "," + str(rank) + "," + str(evar) +\n")
    f.close()