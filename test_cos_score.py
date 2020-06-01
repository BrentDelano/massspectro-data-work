# Written by Brent Delano
# 5/31/2020
# Test cases for functions of cos_score.py

import unittest
import cos_score
import pyteomics
from pyteomics import mgf

class TestCosineScore(unittest.TestCase):
	def test_read_mgf_cosine(self):
		ans = ([[[143.0, 7.605], [144.0, 2.369], [173.0, 100.0], [175.0, 3.178], [188.0, 87.625], 
					 [190.0, 2.304]], [[71.0, 5.81], [73.0, 26.348], [75.0, 9.937], [85.0, 7.418], 
					 [86.0, 54.053], [87.0, 6.039], [102.0, 9.516], [103.0, 30.481], [104.0, 100.0]]], [189.103, 104.071])
		self.assertTupleEqual(cos_score.read_mgf_cosine('./tests/test1.mgf'), ans)
		self.assertTupleEqual(cos_score.read_mgf_cosine('./tests/test1.mgf',[2]), ([ans[0][1]], [ans[1][1]]))
		ans[0].append([[143.0, 7.605]])
		ans[1].append(189.103)
		self.assertTupleEqual(cos_score.read_mgf_cosine(['./tests/test1.mgf', './tests/test2.mgf']), ans)

	def test_calc_cos_scores(self):
		test = cos_score.read_mgf_cosine('./tests/test3.mgf')
		ans = [[1.0, 0.14918123833989144, 0.0], [0.14918123833989144, 1.0, 0.0], [0.0, 0.0, 1.0]]
		with self.assertRaises(IndexError):
			cos_score.calc_cos_scores(test[0], [100, 200])
		self.assertListEqual(cos_score.calc_cos_scores(test[0], test[1]), ans)
		