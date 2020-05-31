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
# optional: specific_spectra - a list of lists ([[file # within given lists, spectra # in file], ...]) (file # & spectra # start from one)
#	Note: if only one .mgf file is given, then specific_spectra should be a 1D list ([spectra # in file, ...])
#	used to make process more efficient and not calculate cosine scores with all spectra
# sort through HMDB file and place data into spectra list
# returns a spectra array with m/z ratios and intensities as a list of lists (different from binning-ms.py)
def read_mgf_cosine(mgfFile, specific_spectra=0):
	spectra = []
	masses = []
	if (specific_spectra == 0):
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
	else:
		if (isinstance(mgfFile, list)):
			for n, mgfs_n in enumerate(mgfFile):
				with mgf.MGF(mgfs_n) as reader:
					s_in_mgf = []
					for s in specific_spectra:
						if (s[0] - 1 == n):
							s_in_mgf.append(s[1])
					for k, spectrum in enumerate(reader):
						for j in s_in_mgf:
							if (k == j - 1):
								masses.append(spectrum['params']['pepmass'][0])
								temp = []
								for i in range(len(spectrum['m/z array'])):
									temp.append([spectrum['m/z array'][i], spectrum['intensity array'][i]])
								spectra.append(temp)
		else:
			with mgf.MGF(mgfFile) as reader:
				for k, spectrum in enumerate(reader):
					for j in specific_spectra:
						if (k == j - 1):
							masses.append(spectrum['params']['pepmass'][0])
							temp = []
							for i in range(len(spectrum['m/z array'])):
								temp.append([spectrum['m/z array'][i], spectrum['intensity array'][i]])
							spectra.append(temp)

	return spectra, masses


# calculates all cosine scores and creates a .txt file titled 'cos_score_data' with the scores
# the .txt file has a nxn array with each row representing a spectra and each column also representing a spectra
#	therefore, [i, j] is the cosine score between spectra i and spectra j
def calc_cos_scores(spectra, masses):
	if (len(spectra) != len(masses)):
		raise IndexError('spectra and masses must be lists of the same length')
	cosScoreTxt = open('cos_score_data', 'w')
	cosScores = []
	cosScoreTxt.write('[')
	for i,rSpec in enumerate(spectra):
		cosScoreTxt.write('[')
		temp = []
		for j,cSpec in enumerate(spectra):
			if j > 0 and j < len(spectra):
				cosScoreTxt.write(', ')
			if i == j:
				cosScoreTxt.write('1.0')
				temp.append(1.0)
			else:
				x = spectrum_alignment.score_alignment(rSpec, cSpec, masses[i], masses[j], 0.02)[0]
				cosScoreTxt.write(str(x))
				temp.append(x)
		if i < len(spectra)-1:
			cosScoreTxt.write('], ')
		else:
			cosScoreTxt.write(']')
		cosScores.append(temp)
	cosScoreTxt.write(']')
	cosScoreTxt.close()
	return cosScores

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

# for testing
def main():
	r = read_mgf_cosine('./tests/test3.mgf')
	print(r)
	print(calc_cos_scores(r[0], r[1]))

if __name__ == "__main__":
	main()
