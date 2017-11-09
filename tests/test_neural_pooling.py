'''
Unit test suite for the :py:mod:`tdparse.neural_pooling` module.
'''
from unittest import TestCase

import numpy as np

from tdparse.neural_pooling import matrix_min
from tdparse.neural_pooling import matrix_max
from tdparse.neural_pooling import matrix_avg
from tdparse.neural_pooling import matrix_median
from tdparse.neural_pooling import matrix_std
from tdparse.neural_pooling import matrix_prod
from tdparse.neural_pooling import matrix_checking

@matrix_checking
def matrix_row_error(matrix, transpose=False):
    '''
    Converts the matrix into a column vector from a matrix e.g.
    Input = m, n
    output = (m * n), 1
    '''
    m_rows = matrix.shape[0]
    m_columns = matrix.shape[1]
    return np.reshape(matrix, (m_rows * m_columns, ))
@matrix_checking
def matrix_dim_error(matrix, transpose=False):
    '''
    Outputs a matrix instead of a vector.
    '''
    m_rows = matrix.shape[0]
    matrix = matrix.min(axis=1)
    matrix = np.reshape(matrix, (m_rows, 1))
    if len(matrix.shape) == 1:
        raise Exception('This test function is not working properly')
    return matrix

@matrix_checking
def matrix_error(matrix, transpose=False):
    '''
    Tests that this function never gets applied.
    '''
    raise Exception('Should never get here')

class TestNeuralPooling(TestCase):
    '''
    Contains the following functions:
    '''

    num_array = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    minus_num_array = np.asarray([[-1, 2, 3, 4], [5, -6, 7, 8], [9, 10, 11, 12]])
    float_array = np.asarray([[-1, 0.112], [0.000005, -0.6], [0.009, 0.1]])
    float_med_array = np.asarray([[-1, 0.112, 1], [0.000005, -0.6, 0.5]])

    def test_check_decorator(self):
        '''
        Tests the decorator :py:func:`tdparse.neural_pooling.matrix_checking`
        used by the following functions:

        1. :py:func:`tdparse.neural_pooling.matrix_min`
        2. :py:func:`tdparse.neural_pooling.matrix_max`
        3. :py:func:`tdparse.neural_pooling.matrix_avg`
        '''

        with self.assertRaises(TypeError, msg='Should only not accept lists only '\
                               'np.ndarray'):
            matrix_min([1, 2, 3, 4])
        test_array = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError, msg='Should not accept returned matrix' \
                               ' from neural functions that have different row '\
                               'row dimensions'):
            matrix_row_error(test_array)
        with self.assertRaises(ValueError, msg='Should not accept returned matrix' \
                               ' from neural functions that have more than one '\
                               'dimension as it should be a vector not a matrix'):
            matrix_dim_error(test_array)
        vector_max = [[5]]
        vector_2d = np.asarray([1, 2, 3, 4, 5]).reshape(5, 1)
        self.assertEqual(True, np.array_equal(vector_max, matrix_max(vector_2d)),
                         msg='Should be able to handle normal cases')
        vector_max = [[1], [2], [3], [4], [5]]
        vector_2d = np.asarray([1, 2, 3, 4, 5]).reshape(1, 5)
        self.assertEqual(True, np.array_equal(vector_max, matrix_max(vector_2d, True)),
                         msg='Should be able to handle normal cases with transpose')


    def test_matrix_min(self):
        '''
        Tests :py:func:`tdparse.neural_pooling.matrix_min`
        '''

        num_cor = np.asarray([1, 2, 3, 4]).reshape(1, 4)
        minus_cor = np.asarray([-1, -6, 3, 4]).reshape(1, 4)
        float_cor = np.asarray([-1, -0.6]).reshape(1, 2)

        num_out = matrix_min(self.num_array)
        minus_out = matrix_min(self.minus_num_array)
        float_out = matrix_min(self.float_array)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(minus_cor, minus_out), msg='Cannot '\
                         'handle negatives')
        self.assertEqual(True, np.array_equal(float_cor, float_out), msg='Cannot '\
                         'handle float values')

    def test_matrix_max(self):
        '''
        Tests :py:func:`tdparse.neural_pooling.matrix_max`
        '''

        num_cor = np.asarray([9, 10, 11, 12]).reshape(1, 4)
        minus_cor = np.asarray([9, 10, 11, 12]).reshape(1, 4)
        float_cor = np.asarray([0.009, 0.112]).reshape(1, 2)

        num_out = matrix_max(self.num_array)
        minus_out = matrix_max(self.minus_num_array)
        float_out = matrix_max(self.float_array)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(minus_cor, minus_out), msg='Cannot '\
                         'handle negatives: real out {} correct values {}'\
                         .format(minus_out, minus_cor))
        self.assertEqual(True, np.array_equal(float_cor, float_out), msg='Cannot '\
                         'handle float values: real out {} correct values {}'\
                         .format(float_out, float_cor))

    def test_matrix_avg(self):
        '''
        Tests :py:func:`tdparse.neural_pooling.matrix_mean`
        '''

        num_cor = np.asarray([5, 6, 7, 8]).reshape(1, 4)
        minus_cor = np.asarray([4.333333333333333, 2, 7, 8]).reshape(1, 4)
        float_cor = np.asarray([-0.33033166666666663, -0.12933333333333333])\
                    .reshape(1, 2)

        num_out = matrix_avg(self.num_array)
        minus_out = matrix_avg(self.minus_num_array)
        float_out = matrix_avg(self.float_array)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(minus_cor, minus_out), msg='Cannot '\
                         'handle negatives: real out {} correct values {}'\
                         .format(minus_out, minus_cor))
        self.assertEqual(True, np.array_equal(float_cor, float_out), msg='Cannot '\
                         'handle float values: real out {} correct values {}'\
                         .format(float_out, float_cor))

    def test_matrix_median(self):
        '''
        Tests :py:func:`tdparse.neural_pooling.matrix_median`
        '''

        num_cor = np.asarray([5, 6, 7, 8]).reshape(1, 4)
        minus_cor = np.asarray([5, 2, 7, 8]).reshape(1, 4)
        float_med_cor = np.asarray([-0.4999975, -0.244, 0.75]).reshape(1, 3)

        num_out = matrix_median(self.num_array)
        minus_out = matrix_median(self.minus_num_array)
        float_out = matrix_median(self.float_med_array)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(minus_cor, minus_out), msg='Cannot '\
                         'handle negatives: real out {} correct values {}'\
                         .format(minus_out, minus_cor))
        self.assertEqual(True, np.array_equal(float_med_cor, float_out), msg='Cannot '\
                         'handle float or odd number values: real out {} correct '\
                         'values {}'.format(float_out, float_med_cor))

    def test_matrix_std(self):
        '''
        Tests :py:func:`tdparse.neural_pooling.matrix_std`
        '''
        num_array = np.asarray([[1, 2], [5, 6], [9, 10]])
        std_array = np.asarray([[1, 2, 3, 4]])

        num_out = matrix_std(num_array)
        std_out = matrix_std(std_array)

        num_cor = np.asarray([3.2659863237109041, 3.2659863237109041]).reshape(1, 2)
        std_corr = np.asarray([0, 0, 0, 0]).reshape(1, 4)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(std_corr, std_out), msg='Cannot '\
                         'handle vectors: real out {} correct values {}'\
                         .format(std_out, std_corr))
    def test_matrix_prod(self):
        '''
        Tests :py:func:`tdparse.neural_pooling.matrix_std`
        '''
        num_array = np.asarray([[1, 2], [5, 6], [9, 10]])
        std_array = np.asarray([[1, 2, 3, 4]])
        zero_array = np.asarray([[-1, 0], [0.5, -0.6], [0.009, 0.1]])

        num_out = matrix_prod(num_array)
        std_out = matrix_prod(std_array)
        zero_out = matrix_prod(zero_array)

        num_cor = np.asarray([45, 120]).reshape(1, 2)
        std_corr = np.asarray([1, 2, 3, 4]).reshape(1, 4)
        zero_corr = np.asarray([-0.0045, 0]).reshape(1, 2)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(std_corr, std_out), msg='Cannot '\
                         'handle vectors: real out {} correct values {}'\
                         .format(std_out, std_corr))
        self.assertEqual(True, np.array_equal(zero_corr, zero_out), msg='Cannot '\
                         'handle vectors with zeros: real out {} correct values {}'\
                         .format(zero_out, zero_corr))
