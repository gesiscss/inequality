import unittest
from calculate import h_index
import numpy as np
from scholarmetrics import hindex

class TestHindex(unittest.TestCase):
    def test_hindex(self):
        data = [1,2,3,4]
        result = h_index(data)
        expected = 2
        self.assertEqual(result, expected)

    def test_hindex2(self):
        data = [0,0,3,2,3,4]
        result = h_index(data)
        expected = 3
        self.assertEqual(result, expected)

    def test_hindex3(self):
        data = np.random.randint(10, 100, 100)
        result1 = h_index(data)
        result2 = hindex(data)
        self.assertEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()
