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

filetype = "png" # "png" or "pdf"
figdpi = 200 #int: DPI of the PDF output file
fig = None
output_pdf = None
output_filename = "plot"

#Saves/Appends the plot to the predetermined PDF file
def savePlot():
    # output_pdf.savefig(plt.gcf())
    plt.savefig(output_filename + "." + filetype, dpi=figdpi, format=filetype)
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
    
    # Percentage is 5% percent of the difference between the min and max, rounded. 
    # This adds onto the end for extra space to make the GUI nice
    # It's also used to set the tick mark distance so its evenly spaced and scales based on the axis size
    percentage = round((end-start)*.05)
    percentage = percentage if percentage > 0 else 0.5
    end = end + percentage
    ax.set_xlim(start, end)
    ax.xaxis.set_ticks(np.arange(start, end, percentage))

    # set y-axis
    start, end = y_lim

    # Percentage is 5% percent of the difference between the min and max, rounded. 
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


mgf_data = sys.path[0] + "/data/agp3k.mgf"
bin_size = 0.01

input_data, bins, scan_names = fast_binner.bin_sparse_dok(mgf_data, verbose = True, bin_size = bin_size, output_file = "agp3k.mgf_matrix.pkl")
input_data = input_data.T
mean = input_data.mean()
squared = input_data.copy()
squared.data **= 2

variance = squared.mean() - (mean**2)
std_dev = math.sqrt(variance)
output = ((input_data).toarray() - mean)/std_dev

ax = graphSetup("Standard Deviation Histogram of Agp3k", "Standard Deviation (Z-Scores)", "Frequency", [np.min(output), np.max(output)], [0, 1000])

ax.hist(std_dev)

savePlot()