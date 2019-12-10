import unittest
from calculate import gini, gini_nonzero

class TestGini(unittest.TestCase):
    def test_gini(self):
        data = [1.,2.,3.,4.]
        result = gini(data)
        expected = 0.250
        self.assertEqual(round(result,3), expected)

    def test_gini2(self):
        data = [10., 2., 5., 1., 6.]
        result = gini(data)
        expected = 0.367
        self.assertEqual(round(result, 3), expected)

    def test_gini_nonzero(self):
        data = [1., 2., 3., 4.]
        result = gini_nonzero(data)
        expected = 0.250
        self.assertEqual(round(result, 3), expected)

    def test_gini_nonzero2(self):
        data = [10., 2., 5., 1., 6., 0., 0.]
        result = gini_nonzero(data)
        expected = 0.367
        self.assertEqual(round(result, 3), expected)

    def test_gini_nonzero3(self):
        data = [1., 2., 3., 4., 0., 0., 0.]
        result = gini_nonzero(data)
        expected = 0.250
        self.assertEqual(round(result, 3), expected)


if __name__ == '__main__':
    unittest.main()