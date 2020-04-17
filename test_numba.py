from pyteomics import mgf
import sys
import time
from numba import jit
sys.path.insert(1, './lib')
import spectrum_alignment
import spectrum_alignment_numba
import pprofile
def compute_pairwise_cosines(mgfs):
    cosines = []
    for i, spec1 in enumerate(mgfs):
        spec1list = [x for x in zip(spec1["m/z array"], spec1["intensity array"])]
        pm1 = spec1["params"]["pepmass"][0]
        for j, spec2 in enumerate(mgfs):
            spec2list = [x for x in zip(spec2["m/z array"], spec2["intensity array"])]
            pm2 = spec2["params"]["pepmass"][0]
            if i == j: continue
            score = spectrum_alignment.score_alignment(spec1list, spec2list, pm1, pm2, tolerance = 2)
            cosines.append(score[0])
    return(cosines)



def compute_pairwise_cosines_numba(mgfs):
    cosines = []
    for i, spec1 in enumerate(mgfs):
        spec1list = [x for x in zip(spec1["m/z array"], spec1["intensity array"])]
        pm1 = spec1["params"]["pepmass"][0]
        for j, spec2 in enumerate(mgfs):
            spec2list = [x for x in zip(spec2["m/z array"], spec2["intensity array"])]
            pm2 = spec2["params"]["pepmass"][0]
            if i == j: continue
            score = spectrum_alignment_numba.score_alignment(spec1list, spec2list, pm1, pm2, tolerance = 2)
            cosines.append(score[0])
    return(cosines)


mgf_data = ['./data/HMDB.mgf']

reader = mgf.MGF(mgf_data[0])
mgfs = [x for x in reader][1:100]

t = time.process_time()
score = compute_pairwise_cosines(mgfs)
elapsed_time = time.process_time() - t
profiler = pprofile.Profile()

#with profiler:
t2 = time.process_time()
score = compute_pairwise_cosines_numba(mgfs)
elapsed_time2 = time.process_time() - t2
#profiler.dump_stats("/tmp/profiler_stats.txt")
print(elapsed_time, elapsed_time2)

print("done")

