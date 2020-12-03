import noise_filteration
import binning_ms
import numpy as np
import pandas as pd
import os
import sys
import re
import nimfa
import glob
import matplotlib
matplotlib.use('Agg') #for plotting w/out GUI - for use on server
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import time
start = time.time()

filetype = "pdf" # "png" or "pdf"
figdpi = 72 #int: DPI of the PDF output file
fig = None
output_pdf = None
output_filename = "plot"
output = "final_motifs.csv"

if output_filename != None:
       output_pdf = PdfPages(output_filename + "." + filetype) # Creates object used to write plots to a pdf file

# Key listener function used to close all plt windows on escape
def close_windows(event):
    if event.key == 'escape':
        plt.close('all')
        sys.exit(0)

#Saves/Appends the plot to the predetermined PDF file
def savePlot():
    if(output_pdf != None):
        output_pdf.savefig(plt.gcf())
        # plt.savefig(output_filename + "." + filetype, dpi=figdpi, format=filetype)
        print("Plot saved to " + output_filename + "." + filetype)

'''Configures a graph according to the given settingsAssumes that x_lim and y_lim are
2D vectors with an upper and lower limit such as [1,2]'''
def graphSetup(title, x_label, y_label, x_lim, y_lim):
    fig = plt.figure(title)
    ax = fig.add_subplot()

    # remove top and right axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # set x-axis
    plt.xticks(rotation='75')
    start, end = x_lim
    
    # Percentage is 10% percent of the difference between the min and max, rounded. 
    # This adds onto the end for extra space to make the GUI nice
    # It's also used to set the tick mark distance so its evenly spaced and scales based on the axis size
    percentage = round((end-start)*.05)
    percentage = percentage if percentage > 0 else 0.5
    end = end + percentage
    ax.set_xlim(start, end)
    ax.xaxis.set_ticks(np.arange(start, end, percentage))

    # set y-axis
    start, end = y_lim

    # Percentage is 10% percent of the difference between the min and max, rounded. 
    # This adds onto the end for extra space to make the GUI nicer
    # It's also used to set the tick mark distance so its evenly spaced and scales based on the axis size
    percentage = round((end-start)*.05)
    percentage = percentage if percentage > 0 else 0.5
    end = end + percentage
    ax.set_ylim(start, end)
    ax.yaxis.set_ticks(np.arange(start, end, percentage))

    # set grid
    plt.grid(True, axis="y", color='black', linestyle=':', linewidth=0.1)
    plt.tight_layout()

    #Object used for plotting
    return ax
range = np.concatenate(([0], np.arange(0.091,.2,.001)))
mgf_data = sys.path[0] + "/data/nematode_symbionts.mgf"
temp_mgf = "temp.mgf"
#if os.path.exists(temp_mgf):
#    os.remove(temp_mgf)

motif_path = sys.path[0] + "/MS2LDA_motifs"
all_files = glob.glob(os.path.join(motif_path, "*.csv")) #Makes an array with all the motif filenames
motif_dfs = []

start = time.time()
for file in all_files:
    all_bins = np.arange(94.0225, 1372.04, 0.005)
    temp_df = pd.read_csv(file)
    temp_df = temp_df[~temp_df['Feature'].str.contains("loss")] #Removes all rows containing the string "loss"
    temp_df['Feature'] = pd.to_numeric(temp_df["Feature"].astype(str).str[9:], errors="coerce") #removes the word "fragment_" from the df columns
    for value in temp_df['Feature'].values:
        all_bins = np.delete(all_bins, np.where(abs(all_bins-value) < 0.0000001)) #Removes all the bins already in the motif from the all_bins list
    for bin in all_bins:
        temp_df = temp_df.append({"Feature": bin, "Probability": 0}, ignore_index=True) # adds zeros for all the bins that weren't in the motif
    temp_df = temp_df.sort_values("Feature") #sorts for post processing purposes
    temp_df.index.name = "Motif " + re.findall(r'.*(?:\D|^)(\d+)', file)[0] #gets last number (motif #) from string
    motif_dfs.append(temp_df)
print('It took {0:0.1f} seconds for processing motifs'.format(time.time() - start))

bin_size = 0.005
'''
#noise_filteration.noise_filteration(mgf=[mgf_data], method=0, min_intens=0.015, mgf_out_filename=temp_mgf)
data = binning_ms.read_mgf_binning(temp_mgf) 
new_bins = binning_ms.create_bins(data[0], bin_size)
new_peaks = binning_ms.create_peak_matrix(data[0], data[1], new_bins)[0]
low_b = [q[0] for q in new_bins]
for v,c in enumerate(low_b):
    low_b[v] = str(c)
    low_b[v] = low_b[v].replace(',', '_')
output_df = pd.DataFrame(data=new_peaks, columns=low_b, index=data[2])
'''
start = time.time()
output_df = pd.read_csv("binned_data.csv")
print('It took {0:0.1f} seconds to read csv'.format(time.time() - start))
#Post filtration process
input_data = output_df.drop(output_df.columns[0], axis=1) #Trims the data so the first column isn't included

bin_lower_bounds = []
# Loops through each column header in the .csv file to get the lower bound for plotting
for column in input_data.columns:
    bound = re.findall(r"[-+]?\d*\.\d+|\d+", column) # Parses the float bound from the column header
    bin_lower_bounds.append(float(bound[0]))

start = time.time()
# Convert to np array and transpose it so that the bin numbers are the rows and it's vectors of spectra intensity
data = np.transpose(input_data.values)
nmf_model = nimfa.Nmf(data, rank=30)
basis = nmf_model().basis()
basis = np.transpose(basis) #makes the basis vectors rows not columns
print('It took {0:0.1f} seconds to do NMF processes'.format(time.time() - start))
#np.savetxt("basis.csv", np.asarray(basis), delimiter=",")
euc_distances = []

start = time.time()
# print(motif_dfs)
for vector in basis:
    # print(vector)
    basis_distances = []
    for motif in motif_dfs:
        motif_vector = motif.get("Probability").to_numpy()
        motif_padded = np.pad(motif_vector, (0, np.shape(basis)[1]-motif_vector.size))
        distance = np.linalg.norm(vector-motif_padded)
        basis_distances.append(distance)
    euc_distances.append(basis_distances)
print('It took {0:0.1f} seconds to calculate distance matrix'.format(time.time() - start))

# print(euc_distances)
# print(np.shape(euc_distances))
start = time.time()
k = 5 #number of min values to collect
idx = np.argpartition(euc_distances, k, axis=None)[:k]
row_size = np.shape(euc_distances)[1]
final_basis = []
final_motifs = []

f = open(output, "w")
for x in idx:
    #Because the partition sorting disregards multidimensionality, it treats it as one vector
    # Therefore, below you need to divide+truncate and mod to get the i,j index for the original matrix
    basis_index = int(x/row_size)
    motif_index = x%row_size
    final_basis.append(basis[basis_index])
    final_motifs.append(np.transpose(motif_dfs[motif_index].get("Probability").to_numpy()))

    f.write(motif_dfs[motif_index].index.name + "\n")

f.close()
print('It took {0:0.1f} seconds to organize final basis/motif arrays'.format(time.time() - start))

# print(np.shape(final_basis))
# print(np.shape(final_motifs))

ax = graphSetup("MassSpectra NMF Basis Vector vs Motif Plot", "Bin Lower Bounds [m/z]", r"$Intensity\,[\%]$", [np.min(bin_lower_bounds), np.max(bin_lower_bounds)], [0,100])

start = time.time()
for v in final_basis:
    v = np.asarray(v)
    v = v[0]
    v = v/np.max(v) * 100 #normalizes based on the largest number in the vector
    ax.plot(bin_lower_bounds, v, color="blue")
    # ax.bar(bin_lower_bounds, v, color="blue") #Bar graph not displaying values properly

# print(np.shape(bin_lower_bounds))
for m in final_motifs:
    # print(np.shape(m))
    m = m.get("Probability").to_numpy()
    m = m/np.max(m) * 100 #normalizes based on the largest number in the vector
    ax.plot(bin_lower_bounds, m, color="green")
    # ax.bar(bin_lower_bounds, m, color="green") #Bar graph not displaying values properly
print('It took {0:0.1f} seconds for graphs'.format(time.time() - start))

basis_patch = mpatches.Patch(color='blue', label='Basis Vectors')
motif_patch = mpatches.Patch(color='green', label='MS2LDA Motifs')

plt.legend(handles=[basis_patch, motif_patch])

savePlot()

if output_filename != None:
    output_pdf.close()
    
plt.gcf().canvas.mpl_connect('key_press_event', close_windows) #attaches keylistener to plt figure

plt.show()

"""
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
