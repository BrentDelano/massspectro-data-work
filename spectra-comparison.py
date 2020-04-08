# Written by Brent Delano
# 4/7/2020
# Uses the cos-score.py and binning-ms.py files to find similarities between spectra

import pyteomics
from pyteomics import mgf
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './GNPS_Workflows/shared_code')
import spectrum_alignment
import cos-score.py
import binning-ms.py

def read_mult_mgfs(mgfs):
	
	for mgf in mgfs:

