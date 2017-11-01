'''
Contains the following neural pooling functions:
1. min
2. max
3. avg
'''
from functools import wraps

import numpy as np

def matrix_checking(neural_func):
    '''
    Contains decorator function to check argument compbatbility and the
    decorated functions return.
    '''
    @wraps(neural_func)
    def pre_post_check(matrix):
        '''
        Checks the matrix is of the correct type and that the return matrix
        is of the correct size after the neural_func function has been applied to
        the matrix. The neural_func can be one of the following functions:
        1. :py:func:`tdparse.neural_pooling.matrix_min`
        2. :py:func:`tdparse.neural_pooling.matrix_max`
        3. :py:func:`tdparse.neural_pooling.matrix_avg`

        It also return the matrix without applying the neural_func function is the
        matrix is a vector as none of the pooling functions will affect the values
        in a vector.

        :param matrix: matrix or vector
        :type matrix: np.ndarray
        :returns: The output of the neural_func function.
        :rtype: np.ndarray
        '''
        # Pre check
        matrix_type = type(matrix)
        if not isinstance(matrix_type, np.ndarray):
            raise TypeError('The matrix has to be of type numpy.ndarray and not '\
                            '{}'.format(matrix_type))
        if len(matrix.shape) == 1:
            return matrix
        m_rows = matrix.shape[0]
        m_columns = matrix.shape[1]
        if m_columns == 1:
            return np.reshape(matrix, (m_rows, ))
        # Applying the relevant neural pooling function
        reduced_matrix = neural_func(matrix)
        # Post check
        rm_rows = reduced_matrix.shape[0]
        rm_dim = len(reduced_matrix.shape)
        if rm_rows != m_rows and rm_dim != 1:
            raise ValueError('The returned matrix should be a vector and have '\
                             'a dimension of 1 it is: {} and should have the same'\
                             'number of rows as the original {} but has {}'\
                             .format(rm_dim, m_rows, rm_rows))
    return pre_post_check

@matrix_checking
def matrix_min(matrix):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The minimum row values in the matrix. If vector returns the vector
    as is.
    :rtype: np.ndarray
    '''

    return matrix.min(axis=1)
@matrix_checking
def matrix_max(matrix):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The maximum row values in the matrix. If vector returns the vector
    as is.
    :rtype: np.ndarray
    '''

    return matrix.max(axis=1)
@matrix_checking
def matrix_avg(matrix):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The mean row values in the matrix. If vector returns the vector
    as is.
    :rtype: np.ndarray
    '''

    return matrix.mean(axis=1)
