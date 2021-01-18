import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from importlib import reload  
import fast_binner
reload(fast_binner)

class TestFilterFunctions(unittest.TestCase):


    def test_filter_cols(self):        
        row = np.array([0, 0, 1, 2, 2, 2, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2,3,3,3])
        data = np.array([1, 2, 0, 4, 0, 6, 0, 0,0])
        csr = csr_matrix((data, (row, col)), shape=(3, 4))

        self.assertTupleEqual(fast_binner.filter_zero_cols(csr)[0].shape, (3,2))

    def test_filter_rows(self):        
        row = np.array([0, 0, 1, 2, 2, 2, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2,3,3,3])
        data = np.array([1, 2, 0, 4, 0, 6, 0, 0,0])
        csr = csr_matrix((data, (row, col)), shape=(3, 4))
        print(csr.toarray())
        print(fast_binner.filter_zero_rows(csr)[0].toarray())
        print(fast_binner.filter_zero_rows(csr)[1])
        self.assertTupleEqual(fast_binner.filter_zero_rows(csr)[0].shape, (2,4))

if __name__ == '__main__':
    unittest.main()
