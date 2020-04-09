# Written by Brent Delano
# 4/3/2020
# Takes in an .mgf file and creates a text file holding an nxn matrix of all possible cosine scores within the ms/ms data
# Uses pyteomics api (https://pyteomics.readthedocs.io/en/latest/) for reading .mgf files
# Uses UCSD - Center for Computational Mass Spectrometry GNPS_Workflows on Github https://github.com/CCMS-UCSD to calculate cosine scores

import pyteomics
from pyteomics import mgf
import sys
sys.path.insert(1, './GNPS_Workflows/shared_code')
import spectrum_alignment

# takes in either a string or a list of strings (for multiple file paths), each string representing a file path to the .mgf file
# sort through HMDB file and place data into spectra list
# returns a spectra array with m/z ratios and intensities as a list of lists (different from binning-ms.py)
def read_mgf_cosine(mgfFile):
	spectra = []
	masses = []
	if (isinstance(mgfFile, list)):
		for mgfs_n in mgfFile:
			with mgf.MGF(mgfs_n) as reader:
				for spectrum in reader:
					masses.append(spectrum['params']['pepmass'][0])
					temp = []
					for i in range(len(spectrum['m/z array'])):
						temp.append([spectrum['m/z array'][i], spectrum['intensity array'][i]])
					spectra.append(temp)
	else:
		with mgf.MGF(mgfFile) as reader:
			for spectrum in reader:
				masses.append(spectrum['params']['pepmass'][0])
				temp = []
				for i in range(len(spectrum['m/z array'])):
					temp.append([spectrum['m/z array'][i], spectrum['intensity array'][i]])
				spectra.append(temp)
	return spectra, masses


# calculates all cosine scores and creates a .txt file titled 'cos_score_data' with the scores
# the .txt file has a nxn array with each row representing a spectra and each column also representing a spectra
#	therefore, [i, j] is the cosine score between spectra i and spectra j
def calc_cos_scores(mgfFile):
	mgf_contents = read_mgf(mgfFile)
	spectra, masses = mgf_contents[0], mgf_contents[1]

	cosScoreTxt = open('cos_score_data', 'w')
	cosScores = []
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
	cosScoreTxt.close()

	# # calculate first nxn cosine scores
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
	# cosScoreTxt.close()

# # for testing
# def main():
# 	spectra_data = read_mgf_cosine(['./data/HMDB.mgf','./data/agp500.mgf'])
# 	spectra = spectra_data[0]
# 	masses = spectra_data[1]
# 	print(spectra)

# if __name__ == "__main__":
# 	main()
