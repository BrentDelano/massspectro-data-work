from pyteomics import mgf
import sys
import time
sys.path.insert(1, './lib')
import spectrum_alignment
import pickle
import multiprocessing
from joblib import Parallel, delayed

def calc_sub_mp(spec1, mgfs):
    out = open("QUERT_distances.txt", "a")
    spec1list = [x for x in zip(spec1["m/z array"], spec1["intensity array"])]
    pm1 = spec1["params"]["pepmass"][0]
    s1 = spec1["params"]["scans"]
    scores = []
    for j, spec2 in enumerate(mgfs):
        spec2list = [x for x in zip(spec2["m/z array"], spec2["intensity array"])]
        pm2 = spec2["params"]["pepmass"][0]
        s2 = spec2["params"]["scans"]
        score = spectrum_alignment.score_alignment(spec1list, spec2list, pm1, pm2, tolerance = 2)   
        scores.append((s1, s2, score))
        outstr = s1 + "\t" + s2 + "\t" + str(score[0]) + "\n"
#        print(outstr)
        out.write(outstr)
    out.close()
    return(score)
                                                   
def compute_pairwise_cosines(mgfs):
    cosines = []
    results = Parallel(n_jobs=num_cores)(delayed(calc_sub_mp)(spec1, mgfs[i+1:len(mgfs)]) for i, spec1 in enumerate(mgfs[0:len(mgfs)-1]) )

mgf_data = ['./data/QUERT.mgf']

reader = mgf.MGF(mgf_data[0])
mgfs = [x for x in reader]
num_cores = multiprocessing.cpu_count()
print("Number of cores: " + str(num_cores))
score = compute_pairwise_cosines(mgfs)


print("done")
