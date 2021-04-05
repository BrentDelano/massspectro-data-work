import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg') #for plotting w/out GUI on server
import matplotlib.pyplot as plt
import time
import fast_binner
import umap
import umap.plot
import nimfa


filetype = "png" # "png" or "pdf"
figdpi = 200 #int: DPI of the PDF output file
output_filename = "umap_agp3k"

# Key listener function used to close all plt windows on escape
def close_windows(event):
    if event.key == 'escape':
        plt.close('all')
        sys.exit(0)

def savePlot():
    plt.savefig(output_filename + "." + filetype, dpi=figdpi, format=filetype)
    print("Plot saved to " + output_filename + "." + filetype)

mgf_data = sys.path[0] + "/data/agp3k.mgf"
bin_size = 0.1

CSR_data, bin_lower_bounds, scan_names = fast_binner.bin_sparse_dok(mgf_data, bin_size=bin_size, verbose = True)

nmf_model = nimfa.Cnmf(CSR_data, rank=30)
model = nmf_model()
K = nmf_model.self.K

mapper = umap.UMAP().fit(CSR_data.T)

# spectra = pd.read_csv("agp3k_for_umap_plot.csv").values
labels = []
for i in range(0, CSR.data.shape[1]):
    if int(i) in K:
        labels.append("Picked out by CNMF")
    else:
        labels.append("Other")

labels = np.asarray(labels)

umap.plot.points(mapper, labels=labels)

savePlot();

plt.gcf().canvas.mpl_connect('key_press_event', close_windows) #attaches keylistener to plt figure

plt.show()
