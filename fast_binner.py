from pyteomics import mgf, mzxml
import numpy as np
from scipy.sparse import dok_matrix
import math
import time
import pickle as pkl


mgf_file = "data/nematode_symbionts.mgf"

def filter_zero_cols(csr):
    M = csr[:,csr.getnnz(0)>0]
    return(csr)

def filter_zero_rows(csr):
    csr = csr[csr.getnnz(1)>0]
    return(csr)

def bin_sparse_dok(mgf_file, min_bin = 50, max_bin = 2000, bin_size = 0.01, verbose = False, remove_zero_sum_rows = True, remove_zero_sum_cols = True):
    start = time.time()
    bins = np.arange(min_bin, max_bin, bin_size)


    reader = mgf.IndexedMGF(mgf_file)
    X = dok_matrix((len(reader.index), len(bins)), dtype=np.float32)

    scan_names = []
    for spectrum_index, spectrum in enumerate(reader):
        if len(spectrum['m/z array']) == 0:
            continue
        for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
            target_bin = math.floor((mz - min_bin)/bin_size)
            X[ spectrum_index,target_bin] += intensity
            scan_names.append(spectrum['params']['scans'])

    X = X.tocsr()
    X_orig_shape = X.shape
    if remove_zero_sum_rows:
        X = filter_zero_rows(X)

    if remove_zero_sum_cols:
        X = filter_zero_cols(X)
        
    if verbose:
            print("Binned in %s seconds with dimensions %sx%s, %s nonzero entries (%s), removed %s rows" % (time.time()-start, len(reader.index), len(bins), X.count_nonzero(), X.count_nonzero()/(len(reader.index)*len(bins)), X_orig_shape[0] - X.shape[0]))

    return(X, bins, scan_names)


X, bins, scan_names = bin_sparse_dok(mgf_file, verbose = True)
