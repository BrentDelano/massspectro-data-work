# Written by Brent Delano
# 5/31/2020
# Test cases for functions of binning_ms.py

import unittest
import binning_ms

class TestBinningMS(unittest.TestCase):
	def test_read_mgf_binning(self):
		ans = ([[143.0, 144.0, 173.0, 175.0, 188.0, 190.0], [71.0, 73.0, 75.0, 85.0, 86.0, 87.0, 102.0, 103.0, 104.0]], 
			   [[7.605, 2.369, 100.0, 3.178, 87.625, 2.304], [5.81, 26.348, 9.937, 7.418, 54.053, 6.039, 9.516, 30.481, 100.0]], 
			   ['./tests/test1.mgf_1', './tests/test1.mgf_2'])
		self.assertTupleEqual(binning_ms.read_mgf_binning('./tests/test1.mgf'), ans)
		ans[0].append([143.0])
		ans[1].append([7.605])
		ans[2].append('./tests/test2.mgf_1')
		self.assertTupleEqual(binning_ms.read_mgf_binning(['./tests/test1.mgf', './tests/test2.mgf']), ans)

	def test_bins(self):
		data = binning_ms.read_mgf_binning('./tests/test1.mgf')[0]
		self.assertEqual(binning_ms.get_min_binsize(data), 1.0)
		self.assertEqual(binning_ms.get_min_binsize([1, 1.5, 5, 10]), 0.5)
		self.assertRaises(TypeError, binning_ms.get_min_binsize, 5)
		
		bins = [[70.9, 85.9], [85.9, 100.9], [100.9, 115.9], [115.9, 130.9], [130.9, 145.9], [145.9, 160.9], [160.9, 175.9], [175.9, 190.9]]
		self.assertListEqual(binning_ms.create_bins(data, 15), bins)
		self.assertListEqual(binning_ms.create_bins([1, 1.5, 5, 10], 3), [[1, 4], [4, 7], [7, 10]])
		with self.assertRaises(ValueError):
			binning_ms.create_bins([1, 1.5, 5, 10], 0)
		with self.assertRaises(TypeError):
			binning_ms.create_bins(3, 5)

		self.assertEqual(binning_ms.find_bin(180, bins), 7)
		with self.assertRaises(ValueError):
			binning_ms.find_bin(60, bins)
		with self.assertRaises(TypeError):
			binning_ms.find_bin(60, [1,2,3])
			binning_ms.find_bin(60, 1)

	def test_create_peak_matrix(self):
		mgf_contents = binning_ms.read_mgf_binning('./tests/test1.mgf')
		ans1 = ([['./tests/test1.mgf_1', './tests/test1.mgf_2'], [0, 0, 0, 7.605, 100.0, 87.625], [49.513, 60.092, 139.997, 0, 0, 0]], 3)
		ans2 = ([['./tests/test1.mgf_1', './tests/test1.mgf_2'], [0, 0, 0, 7.605, 100.0, 87.625], [[5.81, 26.348, 9.937, 7.418], [54.053, 6.039], [9.516, 30.481, 100.0], 0, 0, 0]], 3)
		bins = [[70.9, 85.9], [85.9, 100.9], [100.9, 115.9], [115.9, 130.9], [130.9, 145.9], [145.9, 160.9], [160.9, 175.9], [175.9, 190.9]]
		self.assertTupleEqual(binning_ms.create_peak_matrix(mgf_contents[0], mgf_contents[1], mgf_contents[2], bins, listIfMultMZ=False, minIntens=5, maxIntens=0), ans1)
		self.assertTupleEqual(binning_ms.create_peak_matrix(mgf_contents[0], mgf_contents[1], mgf_contents[2], bins, listIfMultMZ=True, minIntens=5, maxIntens=0), ans2)
		with self.assertRaises(ValueError):
			self.assertTupleEqual(binning_ms.create_peak_matrix(mgf_contents[0], mgf_contents[1], mgf_contents[2], bins, listIfMultMZ=False, minIntens=5, maxIntens=2), ans1)
		with self.assertRaises(TypeError):
			binning_ms.create_peak_matrix([0, 1, 2, 3], mgf_contents[1], mgf_contents[2], bins, listIfMultMZ=False, minIntens=5, maxIntens=0)
			binning_ms.create_peak_matrix(mgf_contents[0], 5, mgf_contents[2], bins, listIfMultMZ=False, minIntens=5, maxIntens=0)

	def test_compression(self):
		test1 = [['./tests/test1.mgf_1', './tests/test1.mgf_2', './tests/test2.mgf_1', './tests/test3.mgf_1', './tests/test3.mgf_2', './tests/test3.mgf_3'], 
				 [0, 7.605, 187.625], [249.602, 0, 0], [0, 7.605, 0], [0, 7.605, 187.625], [0, 7.605, 188.625], [249.602, 0, 0]]
		test2 = [[0, 7.605, 187.625], [249.602, 0, 0], [0, 7.605, 0], [0, 7.605, 187.625], [0, 7.605, 188.625], [249.602, 0, 0]]
		ans12 = ([[-122.43063135, -26.62817616], [189.88653865, -22.14260239], [-11.89203684, 124.97775957], [-122.43063135, -26.62817616], [-123.01977776, -27.43620247],
				[189.88653865, -22.14260239]], [[0.80765151, -0.02460793, -0.58914641], [-0.58887314, 0.01794208, -0.80802631]], [0.86211913, 0.13788087])
		ans3 = ([[-122.43063135, -26.62817616], [189.88653865, -22.14260239], [-11.89203684, 124.97775957], [-122.43063135, -26.62817616], [-123.01977776, -27.43620247],
				[189.88653865, -22.14260239]], [[ 0.80765151, -0.02460793, -0.58914641], [-0.58887314,  0.01794208, -0.80802631]], [0.86211913, 0.13788087])
		result1 = binning_ms.compress_bins(test1)
		result2 = binning_ms.compress_bins(test2)
		result3 = binning_ms.compress_bins_sml(test2)
		for i,a in enumerate(ans12[0]):
			self.assertAlmostEqual(a[0], result1[0].tolist()[i][0])
			self.assertAlmostEqual(a[1], result1[0].tolist()[i][1])
			self.assertAlmostEqual(a[0], result2[0].tolist()[i][0])
			self.assertAlmostEqual(a[1], result2[0].tolist()[i][1])
		for i,a in enumerate(ans12[1]):
			self.assertAlmostEqual(a[0], result1[1].tolist()[i][0])
			self.assertAlmostEqual(a[1], result1[1].tolist()[i][1])
			self.assertAlmostEqual(a[2], result1[1].tolist()[i][2])
			self.assertAlmostEqual(a[0], result2[1].tolist()[i][0])
			self.assertAlmostEqual(a[1], result2[1].tolist()[i][1])
			self.assertAlmostEqual(a[2], result2[1].tolist()[i][2])
		for i,a in enumerate(ans12[2]):
			self.assertAlmostEqual(a, result1[2].tolist()[i])
			self.assertAlmostEqual(a, result2[2].tolist()[i])

		for i,a in enumerate(ans3[0]):
			self.assertAlmostEqual(a[0], result3[0].tolist()[i][0])
			self.assertAlmostEqual(a[1], result3[0].tolist()[i][1])
		for i,a in enumerate(ans3[1]):
			self.assertAlmostEqual(a[0], result3[1].tolist()[i][0])
			self.assertAlmostEqual(a[1], result3[1].tolist()[i][1])
			self.assertAlmostEqual(a[2], result3[1].tolist()[i][2])
		for i,a in enumerate(ans3[2]):
			self.assertAlmostEqual(a, result3[2].tolist()[i])
		
		self.assertRaises(TypeError, binning_ms.compress_bins, [1,2,3,4,5])
		self.assertRaises(TypeError, binning_ms.compress_bins, 'not a 2D array')
