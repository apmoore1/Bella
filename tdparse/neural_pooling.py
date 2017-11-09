'''
Contains the following neural pooling functions:

1. min
2. max
3. avg

Which are from
`Tang et al <https://aclanthology.coli.uni-saarland.de/papers/P14-1146/p14-1146>`_.

and the following pooling functions:

4. prod
5. std

Which are from
`Vo and Zhang <https://www.ijcai.org/Proceedings/15/Papers/194.pdf>`_.

and finally the following pooling function:

6. median

From `Bo Wang et al.
<https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046>`_

All the functions are applied over the columns and not the rows e.g. matrix
of (m, n) size and apply mean it will return a vector of (1, n). Therefore by
default all of the vectors returned are row vectors but if transpose is True
then column vectors are returned.
'''
from functools import wraps

import numpy as np

def matrix_checking(neural_func):
    '''
    Contains decorator function to check argument compbatbility and the
    decorated functions return. The functions decorated are the neural functions
    which are:

    1. :py:func:`tdparse.neural_pooling.matrix_min`
    2. :py:func:`tdparse.neural_pooling.matrix_max`
    3. :py:func:`tdparse.neural_pooling.matrix_avg`
    '''
    @wraps(neural_func)
    def pre_post_check(matrix, transpose=False):
        '''
        Checks the matrix is of the correct type and that the return matrix
        is of the correct size after the neural_func function has been applied to
        the matrix.

        Applies transpose to convert row vectors into column vectors if
        transpose == False

        :param matrix: matrix or vector
        :param transpose: If to convert the column vector into row vector
        :type matrix: np.ndarray
        :type transpose: bool default False
        :returns: The output of the neural_func function.
        :rtype: np.ndarray
        '''
        # Pre check
        if not isinstance(matrix, np.ndarray):
            raise TypeError('The matrix has to be of type numpy.ndarray and not '\
                            '{}'.format(type(matrix)))
        # Applying the relevant neural pooling function
        reduced_matrix = neural_func(matrix)
        # Post check
        rm_cols = reduced_matrix.shape[0]
        rm_dim = len(reduced_matrix.shape)
        if rm_dim != 1:
            raise ValueError('The returned matrix should be a vector and have '\
                             'a dimension of 1 it is: {}'.format(rm_dim))
        m_columns = matrix.shape[1]
        if rm_cols != m_columns:
            raise ValueError('The number of columns has changed during the pooling'\
                             'func from {} to {}'.format(m_columns, rm_cols))
        if transpose:
            return reduced_matrix.reshape(rm_cols, 1)
        return reduced_matrix.reshape(1, rm_cols)
    return pre_post_check

@matrix_checking
def matrix_min(matrix, transpose=False):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The minimum column values in the matrix.
    :rtype: np.ndarray
    '''

    return matrix.min(axis=0)
@matrix_checking
def matrix_max(matrix, transpose=False):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The maximum column values in the matrix.
    :rtype: np.ndarray
    '''

    return matrix.max(axis=0)
@matrix_checking
def matrix_avg(matrix, transpose=False):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The mean column values in the matrix.
    :rtype: np.ndarray
    '''

    return matrix.mean(axis=0)

@matrix_checking
def matrix_median(matrix, transpose=False):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The median column values in the matrix.
    :rtype: np.ndarray
    '''

    return np.median(matrix, axis=0)

@matrix_checking
def matrix_std(matrix, transpose=False):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The standard deviation of the column values in the matrix.
    :rtype: np.ndarray
    '''
    return np.std(matrix, axis=0)

@matrix_checking
def matrix_prod(matrix, transpose=False):
    '''
    :param matrix: matrix or vector
    :type matrix: np.ndarray
    :returns: The product of the column values in the matrix.
    :rtype: np.ndarray
    '''
    return np.prod(matrix, axis=0)
