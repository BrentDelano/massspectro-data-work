import pyteomics
import numpy as np
from pyteomics import mgf
import sys
sys.path.insert(1, '/Users/brent/Desktop/cosScore/GNPS_Workflows/shared_code')
import spectrum_alignment

# Lists to hold the spectra (m/z ratios and intensities) and masses
spectra = []
masses = []

# compare all spectra to each other; place cosine scores into an nxn array
cosScores = []

# Text file that cosine score data will be printed to
cosScoreTxt = open('cos-score-data', 'w')

def main():
	# sort through HMDB file and place data into spectra list
	with mgf.MGF('HMDB.mgf') as reader:
		for spectrum in reader:
			masses.append(spectrum['params']['pepmass'][0])
			temp = []
			for i in range(len(spectrum['m/z array'])):
				temp.append([spectrum['m/z array'][i], spectrum['intensity array'][i]])
			spectra.append(temp)

	# calculate first nxn cosine scores
	# cosScoreTxt.write('[')
	# for i in range(10):
	# 	cosScoreTxt.write('[')
	# 	for j in range(10):
	# 		if (j > 0 and j < 10):
	# 			cosScoreTxt.write(', ')
	# 		if (i == j):
	# 			cosScoreTxt.write('1.0')
	# 		else:
	# 			x = spectrum_alignment.score_alignment(spectra[i], spectra[j], masses[i], masses[j], 0.02)[0]
	# 			cosScoreTxt.write(str(x))
	# 	cosScoreTxt.write(']')
	# cosScoreTxt.write(']')

	# calculate all cosine scores
	cosScoreTxt.write('[')
	for i,rSpec in enumerate(spectra):
		cosScoreTxt.write('[')
		for j,cSpec in enumerate(spectra):
			if (j > 0 and j < len(spectra)):
				cosScoreTxt.write(', ')
			if (i == j):
				cosScoreTxt.write('1.0')
			else:
				x = spectrum_alignment.score_alignment(rSpec, cSpec, masses[i], masses[j], 0.02)[0]
				cosScoreTxt.write(str(x))
		cosScoreTxt.write(']')
	cosScoreTxt.write(']')


if __name__ == "__main__":
	main()
