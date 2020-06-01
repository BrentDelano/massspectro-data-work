# Written by Brent Delano
# 5/31/2020
# Test cases for functions of noise_filteration.py

import unittest
import noise_filteration as nf

class TestNoiseFilteration(unittest.TestCase):
	def test_noise_filteration(self):
		ans1 = ([[173.0, 188.0], [86.0, 104.0]], [[100.0, 87.625], [54.053, 100.0]])
		self.assertTupleEqual(nf.noise_filteration(mgf='./tests/test1.mgf', method=0, min_intens=50), ans1)
		ans2 = ([[143.0, 144.0, 173.0, 175.0, 188.0, 190.0], [71.0, 73.0, 75.0, 85.0, 86.0, 87.0, 102.0, 103.0, 104.0]], 
				[[7.605, 2.369, 100.0, 3.178, 87.625, 2.304], [5.81, 26.348, 9.937, 7.418, 54.053, 6.039, 9.516, 30.481, 100.0]])
		self.assertTupleEqual(nf.noise_filteration(mgf='./tests/test1.mgf', method=0), ans2)
		ans3 = ([[143.0, 144.0, 173.0, 188.0], [73.0, 86.0, 103.0, 104.0]], [[7.605, 2.369, 100.0, 87.625], [26.348, 54.053, 30.481, 100.0]])
		self.assertTupleEqual(nf.noise_filteration(mgf='./tests/test1.mgf', method=1, binsize=20, peaks_per_bin=2), ans3)
		with self.assertRaises(ValueError):
			nf.noise_filteration()
			nf.noise_filteration(mgf='./tests/test1.mgf', method=1, peaks_per_bin=2)
			nf.noise_filteration(mgf='./tests/test1.mgf', method=1, binsize=20)
