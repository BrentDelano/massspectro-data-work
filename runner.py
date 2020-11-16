import noise_filteration
import binning_ms
import numpy as np
import pandas as pd
import os
import sys
import re
import nimfa

range = np.concatenate(([0], np.arange(0.091,.2,.001)))
mgf_data = sys.path[0] + "/data/nematode_symbionts.mgf"
temp_mgf = "temp.mgf"
if os.path.exists(temp_mgf):
    os.remove(temp_mgf)
method = 0
bin = 5
output = "output_measurements_part2.csv"
f = open(output, "w")
f.write("Method,Min Intensity,Num Rows,Rank,Sparseness Basis Vector, Sparseness Coefficient,RSS,Evar\n")
f.close()

rank_range = np.concatenate(([2,5,10], np.arange(30,70,10)))
output_params = ["sparseness", "rss", "evar", "cophenetic"]
for x in range:
    noise_filteration.noise_filteration(mgf=[mgf_data], method=method, min_intens=x, mgf_out_filename=temp_mgf)
    data = binning_ms.read_mgf_binning(temp_mgf) 
    new_bins = binning_ms.create_bins(data[0], 5)
    new_peaks = binning_ms.create_peak_matrix(data[0], data[1], new_bins)[0]
    low_b = [q[0] for q in new_bins]
    for v,c in enumerate(low_b):
        low_b[v] = str(c)
        low_b[v] = low_b[v].replace(',', '_')
    output_df = pd.DataFrame(data=new_peaks, columns=low_b, index=data[2])
    size = output_df.shape
    
    #Post filtration process
    input_data = output_df.drop(output_df.columns[0], axis=1) #Trims the data so the first column isn't included
    bin_lower_bounds = []

    # Loops through each column header in the .csv file to get the lower bound for plotting
    for column in input_data.columns:
        bound = re.findall(r"[-+]?\d*\.\d+|\d+", column) # Parses the float bound from the column header
        bin_lower_bounds.append(float(bound[0]))
    
    # Convert to np array and transpose it so that the bin numbers are the rows and it's vectors of spectra intensity
    data = np.transpose(input_data.values)
    nmf_model = nimfa.Nmf(data)
    ranks = nmf_model.estimate_rank(rank_range, what=output_params)

    f = open(output, "a")

    for t in ranks:
        measures = ranks[t]
        sparseness = measures["sparseness"]
        rss = measures["rss"]
        evar = measures["evar"]
        f.write(str(method) + "," + str(round(x, 4)) + "," + str(size[0])+ "," + str(t) + "," + str(sparseness[0]) + "," + str(sparseness[1]) + "," + str(rss) + "," + str(evar) + "\n")

    f.close()
    os.remove(temp_mgf)

"""
method = 1
range = np.arange(1,11,1)
print()
for x in range:
    noise_filteration.noise_filteration(mgf=[mgf_data], method=method, peaks_per_bin=x,binsize=5, mgf_out_filename=temp_mgf)
    data = binning_ms.read_mgf_binning(temp_mgf)
    new_bins = binning_ms.create_bins(data[0], 5)
    new_peaks = binning_ms.create_peak_matrix(data[0], data[1], new_bins)[0]
    low_b = [q[0] for q in new_bins]
    for v,c in enumerate(low_b):
        low_b[v] = str(c)
        low_b[v] = low_b[v].replace(',', '_')
    output_df = pd.DataFrame(data=new_peaks, columns=low_b, index=data[2])
    size = output_df.shape
    f = open(output, "a")
    f.write(str(method) + "," + str(round(x, 4)) + "," + str(size) + "\n")
    f.close()
    os.remove(temp_mgf)
"""
