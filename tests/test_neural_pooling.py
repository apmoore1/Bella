'''
Unit test suite for the :py:mod:`bella.neural_pooling` module.
'''
from unittest import TestCase

import numpy as np

from bella.neural_pooling import matrix_min
from bella.neural_pooling import matrix_max
from bella.neural_pooling import matrix_avg
from bella.neural_pooling import matrix_median
from bella.neural_pooling import matrix_std
from bella.neural_pooling import matrix_prod
from bella.neural_pooling import matrix_checking
from bella.neural_pooling import inf_nan_check

@matrix_checking
def matrix_row_error(matrix, transpose=True):
    '''
    Converts the matrix into a column vector from a matrix e.g.
    Input = m, n
    output = (m * n), 1
    '''
    m_rows = matrix.shape[0]
    m_columns = matrix.shape[1]
    return np.reshape(matrix, (m_rows * m_columns, ))
@matrix_checking
def matrix_dim_error(matrix, transpose=True):
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
def matrix_error(matrix, transpose=True):
    '''
    Tests that this function never gets applied.
    '''
    raise Exception('Should never get here')

@inf_nan_check
def matrix_inf_nan_check(matrix):
    '''
    Returns the given matrix after it has been checked for INF and NAN values.
    '''
    return matrix

class TestNeuralPooling(TestCase):
    '''
    Contains the following functions:
    '''

    num_array = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                           dtype=np.float32)
    minus_num_array = np.asarray([[-1, 2, 3, 4], [5, -6, 7, 8], [9, 10, 11, 12]],
                                 dtype=np.float32)
    float_array = np.asarray([[-1, 0.112], [0.000005, -0.6], [0.009, 0.1]],
                             dtype=np.float32)
    float_med_array = np.asarray([[-1, 0.112, 1], [0.000005, -0.6, 0.5]],
                                 dtype=np.float32)

    def test_inf_nan_checker(self):
        '''
        Tests the decorator `inf_nan_check`.
        '''
        # Ensure that it does nothing when it doesn't require to do anything
        # e.g. on a zero matrix
        valid_zero_matrix = np.zeros((3, 3), dtype=np.float32)
        test_zero_matrix = matrix_inf_nan_check(valid_zero_matrix)
        the_same = np.array_equal(valid_zero_matrix, test_zero_matrix)
        self.assertEqual(True, the_same, msg='These two matrices should be the '\
                         'same {} {}'.format(valid_zero_matrix, test_zero_matrix))
        # 64 bit version
        valid_zero_matrix = np.zeros((3, 3), dtype=np.float64)
        test_zero_matrix = matrix_inf_nan_check(valid_zero_matrix)
        the_same = np.array_equal(valid_zero_matrix, test_zero_matrix)
        self.assertEqual(True, the_same, msg='These two matrices should be the '\
                         'same {} {}'.format(valid_zero_matrix, test_zero_matrix))

        # Check that it does not accept matrixs that are not of numpy float type
        with self.assertRaises(TypeError, msg='Should not accept any numpy type '\
                               'that is not float type'):
            error_zero_matrix = np.zeros((3, 3), dtype=np.int32)
            matrix_inf_nan_check(error_zero_matrix)

        # Below ensures that it can convert values that are lower or greater
        # than inf for float 32 is converted to numbers that are within the
        # correct range. The correct range is 1/2 the max and min value. 1/2
        # was selected due to calculating the range requires max - min.
        float_32_info = np.finfo(np.float32)

        # Check that it works with -inf on float 32
        lower_than_values = np.asarray([-np.inf, 0], dtype=np.float32)
        valid_values = np.asarray([float_32_info.min / 2, 0], dtype=np.float32)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(lower_than_values))
        self.assertEqual(True, test_result, msg='Should changed the lower than '\
                         'inf value to the lowest possible value for 32 bit float')

        # Check that it works with inf on float 32
        higher_than_values = np.asarray([np.inf, 0], dtype=np.float32)
        valid_values = np.asarray([float_32_info.max / 2, 0], dtype=np.float32)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(higher_than_values))
        self.assertEqual(True, test_result, msg='Should changed the higher than '\
                         'inf value to the highest possible value for 32 bit float')

        # Check that it works for inf and -inf on float 32
        higher_and_lower = np.asarray([np.inf, -np.inf],
                                      dtype=np.float32)
        valid_values = np.asarray([float_32_info.max / 2, float_32_info.min / 2],
                                  dtype=np.float32)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(higher_and_lower))
        self.assertEqual(True, test_result, msg='Should be able to handle '\
                         'both lower and higher than inf for float 32')

        # Below does the same as above but for float 64 instead of float 32
        float_64_info = np.finfo(np.float64)

        # Check that it works with -inf on float 64
        lower_than_values = np.asarray([-np.inf, 0], dtype=np.float64)
        valid_values = np.asarray([float_64_info.min / 2, 0], dtype=np.float64)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(lower_than_values))
        self.assertEqual(True, test_result, msg='Should changed the lower than '\
                         'inf value to the lowest possible value for 64 bit float')

        # Check that it works with inf on float 64
        higher_than_values = np.asarray([np.inf, 0], dtype=np.float64)
        valid_values = np.asarray([float_64_info.max / 2, 0], dtype=np.float64)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(higher_than_values))
        self.assertEqual(True, test_result, msg='Should changed the higher than '\
                         'inf value to the highest possible value for 64 bit float')

        # Check that it works with inf and -inf on float 64
        higher_and_lower = np.asarray([np.inf, -np.inf],
                                      dtype=np.float64)
        valid_values = np.asarray([float_64_info.max / 2, float_64_info.min / 2],
                                  dtype=np.float64)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(higher_and_lower))
        self.assertEqual(True, test_result, msg='Should be able to handle '\
                         'both lower and higher than inf for float 64')

        # Ensures that it converts Nan values to 0 for float 32
        nan_values = np.asarray([np.inf, -np.inf, 3.0, np.nan], dtype=np.float32)
        valid_values = np.asarray([float_32_info.max / 2, float_32_info.min / 2,
                                   3.0, 0], dtype=np.float32)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(valid_values))
        self.assertEqual(True, test_result, msg='Cannot convert NAN values to 0 '\
                         'for float 32')
        # Ensures that it converts Nan values to 0 for float 64
        nan_values = np.asarray([np.inf, -np.inf, 3.0, np.nan], dtype=np.float64)
        valid_values = np.asarray([float_64_info.max / 2, float_64_info.min / 2,
                                   3.0, 0], dtype=np.float64)
        test_result = np.array_equal(valid_values,
                                     matrix_inf_nan_check(valid_values))
        self.assertEqual(True, test_result, msg='Cannot convert NAN values to 0 '\
                         'for float 64')



    def test_check_decorator(self):
        '''
        Tests the decorator :py:func:`bella.neural_pooling.matrix_checking`
        used by the following functions:

        1. :py:func:`bella.neural_pooling.matrix_min`
        2. :py:func:`bella.neural_pooling.matrix_max`
        3. :py:func:`bella.neural_pooling.matrix_avg`
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
        vector_max = np.asarray([[5]], dtype=np.float32)
        vector_2d = np.asarray([1, 2, 3, 4, 5], dtype=np.float32).reshape(5, 1)
        self.assertEqual(True, np.array_equal(vector_max, matrix_max(vector_2d)),
                         msg='Should be able to handle normal cases')
        vector_max = np.asarray([[1], [2], [3], [4], [5]], dtype=np.float32)
        vector_2d = np.asarray([1, 2, 3, 4, 5], dtype=np.float32).reshape(1, 5)
        test_vec_max = matrix_max(vector_2d, transpose=True)
        self.assertEqual(True, np.array_equal(vector_max, test_vec_max),
                         msg='Should be able to handle normal cases with transpose '\
                         'returned {} valid {}'.format(test_vec_max, vector_max))


    def test_matrix_min(self):
        '''
        Tests :py:func:`bella.neural_pooling.matrix_min`
        '''

        num_cor = np.asarray([1, 2, 3, 4], dtype=np.float32).reshape(1, 4)
        minus_cor = np.asarray([-1, -6, 3, 4], dtype=np.float32).reshape(1, 4)
        float_cor = np.asarray([-1, -0.6], dtype=np.float32).reshape(1, 2)

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
        Tests :py:func:`bella.neural_pooling.matrix_max`
        '''

        num_cor = np.asarray([9, 10, 11, 12], dtype=np.float32).reshape(1, 4)
        minus_cor = np.asarray([9, 10, 11, 12], dtype=np.float32).reshape(1, 4)
        float_cor = np.asarray([0.009, 0.112], dtype=np.float32).reshape(1, 2)

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
        Tests :py:func:`bella.neural_pooling.matrix_mean`
        '''

        num_cor = np.asarray([5, 6, 7, 8], dtype=np.float32).reshape(1, 4)
        minus_cor = np.asarray([4.333333333333333, 2, 7, 8], dtype=np.float32)\
                    .reshape(1, 4)
        float_cor = np.asarray([-0.33033166666666663, -0.12933335],
                               dtype=np.float32).reshape(1, 2)

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
        Tests :py:func:`bella.neural_pooling.matrix_median`
        '''

        num_cor = np.asarray([5, 6, 7, 8], dtype=np.float32).reshape(1, 4)
        minus_cor = np.asarray([5, 2, 7, 8], dtype=np.float32).reshape(1, 4)
        float_med_cor = np.asarray([-0.4999975, -0.24400002, 0.75],
                                   dtype=np.float32).reshape(1, 3)

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
        Tests :py:func:`bella.neural_pooling.matrix_std`
        '''
        num_array = np.asarray([[1, 2], [5, 6], [9, 10]], dtype=np.float32)
        std_array = np.asarray([[1, 2, 3, 4]], dtype=np.float32)

        num_out = matrix_std(num_array)
        std_out = matrix_std(std_array)

        num_cor = np.asarray([3.2659863237109041, 3.2659863237109041],
                             dtype=np.float32).reshape(1, 2)
        std_corr = np.asarray([0, 0, 0, 0], dtype=np.float32).reshape(1, 4)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(std_corr, std_out), msg='Cannot '\
                         'handle vectors: real out {} correct values {}'\
                         .format(std_out, std_corr))
    def test_matrix_prod(self):
        '''
        Tests :py:func:`bella.neural_pooling.matrix_std`
        '''
        num_array = np.asarray([[1, 2], [5, 6], [9, 10]], dtype=np.float32)
        std_array = np.asarray([[1, 2, 3, 4]], dtype=np.float32)
        zero_array = np.asarray([[-1, 0], [0.5, -0.6], [0.009, 0.1]],
                                dtype=np.float32)

        num_out = matrix_prod(num_array)
        std_out = matrix_prod(std_array)
        zero_out = matrix_prod(zero_array)

        num_cor = np.asarray([45, 120], dtype=np.float32).reshape(1, 2)
        std_corr = np.asarray([1, 2, 3, 4], dtype=np.float32).reshape(1, 4)
        zero_corr = np.asarray([-0.0045, 0], dtype=np.float32).reshape(1, 2)

        self.assertEqual(True, np.array_equal(num_cor, num_out), msg='Cannot handle '\
                         'basic numbers: real out {} correct values {}'\
                         .format(num_out, num_cor))
        self.assertEqual(True, np.array_equal(std_corr, std_out), msg='Cannot '\
                         'handle vectors: real out {} correct values {}'\
                         .format(std_out, std_corr))
        self.assertEqual(True, np.array_equal(zero_corr, zero_out), msg='Cannot '\
                         'handle vectors with zeros: real out {} correct values {}'\
                         .format(zero_out, zero_corr))
