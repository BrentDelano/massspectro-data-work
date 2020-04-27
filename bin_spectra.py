#!/usr/bin/env python3

from pyteomics import mgf
import numpy as np
import pandas as pd
import os
bin_number = 3000
mz_range = (0, 1200)
bins = np.linspace(mz_range[0], mz_range[1], num = bin_number )


mgf_data = ['./data/agp500.mgf']

reader = [(os.path.basename(x), mgf.MGF(x)) for x in mgf_data]

spectra_bins = {}
for (name, mgfs) in reader:
    for index, m in enumerate([x for x in mgfs]):
        spectra_bins[name+"_" + str(m['params']['scans'])] =  np.digitize(m["m/z array"], bins)

matrix = pd.DataFrame(0, index = bins, columns = spectra_bins.keys())
colnames = []
print(matrix.shape)
for k, v in spectra_bins.items():
    colnames.append(k)
    for i in v:
#        print(i,k)
        matrix.iloc[i-1].loc[k] = 1

print(matrix.shape)
matrix = matrix[(matrix.T != 0).any()]
print(matrix.shape)


matrix.to_csv("agp500_matrix.csv")
print("done")
