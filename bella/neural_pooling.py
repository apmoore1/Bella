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

def inf_nan_check(neural_func):
    '''
    Contains decorator function that converts any inf or NAN value to a real
    number to avoid any potential problems with inf and NAN's latter on in the
    processing chain.

    Inf conversion - Converts it to the max (min) value of the numpy array/matrix
    dtype based on it being positive (negative) value.

    NAN conversion - based on the following
    `post <https://stackoverflow.com/questions/25506281/what-are-all-the-possib\
    le-calculations-that-could-cause-a-nan-in-python>`_ about how NAN's occur.
    It converts NAN's to zeros as the majority of the operation should equal
    zero or are close to zero. This is a rough approximation but it should not
    affect that many numbers.
    '''

    @wraps(neural_func)
    def func_wrapper(matrix, **kwargs):
        '''
        :param matrix: Numpy array/matrix that could contain NAN or inf values.
        :param transpose: If to convert the column vector into row vector
        :type matrix: np.ndarray
        :type transpose: bool
        :returns: The numpy array/matrix with NAN and inf values converted to \
        real values.
        :rtype: np.ndarray
        '''

        matrix = neural_func(matrix, **kwargs)
        if not issubclass(matrix.dtype.type, np.floating):
            raise TypeError('Only accept floating value word embeddings not '\
                            '{}'.format(matrix.dtype.type))
        # Convert all NAN values to zero
        if np.any(np.isnan(matrix)):
            matrix[np.where(np.isnan(matrix))] = 0
        # Find any value that is greater than half the min and max values and
        # convert them to half the min or max value respectively. This is
        # done to ensure that range can be done without overflow exception
        dtype_info = np.finfo(matrix.dtype)
        min_value = dtype_info.min / 2
        max_value = dtype_info.max / 2
        if np.any(matrix[matrix < min_value]) or np.any(matrix[matrix > max_value]):
            matrix[matrix < min_value] = min_value
            matrix[matrix > max_value] = max_value

        return matrix

    return func_wrapper

def matrix_checking(neural_func):
    '''
    Contains decorator function to check argument compbatbility and the
    decorated functions return. The functions decorated are the neural functions
    which are:

    1. :py:func:`bella.neural_pooling.matrix_min`
    2. :py:func:`bella.neural_pooling.matrix_max`
    3. :py:func:`bella.neural_pooling.matrix_avg`
    '''
    @wraps(neural_func)
    def func_wrapper(matrix, transpose=False):
        '''
        Checks the matrix is of the correct type and that the return matrix
        is of the correct size after the neural_func function has been applied to
        the matrix.

        inf values are converted to max (min) value defined by the dtype if
        the value is positive (negative).

        Applies transpose to convert row vectors into column vectors if
        transpose == False

        :param matrix: matrix or vector
        :param transpose: If to convert the column vector into row vector
        :type matrix: np.ndarray
        :type transpose: bool
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
    return func_wrapper

@inf_nan_check
@matrix_checking
def matrix_min(matrix, **kwargs):
    '''
    :param matrix: matrix or vector
    :param kwargs: Can keywords that are accepted by `matrix_checking` function
    :type matrix: np.ndarray
    :type kwargs: dict
    :returns: The minimum column values in the matrix.
    :rtype: np.ndarray
    '''

    return matrix.min(axis=0)

@inf_nan_check
@matrix_checking
def matrix_max(matrix, **kwargs):
    '''
    :param matrix: matrix or vector
    :param kwargs: Can keywords that are accepted by `matrix_checking` function
    :type matrix: np.ndarray
    :type kwargs: dict
    :returns: The maximum column values in the matrix.
    :rtype: np.ndarray
    '''

    return matrix.max(axis=0)

@inf_nan_check
@matrix_checking
def matrix_avg(matrix, **kwargs):
    '''
    :param matrix: matrix or vector
    :param kwargs: Can keywords that are accepted by `matrix_checking` function
    :type matrix: np.ndarray
    :type kwargs: dict
    :returns: The mean column values in the matrix.
    :rtype: np.ndarray
    '''

    return matrix.mean(axis=0)

@inf_nan_check
@matrix_checking
def matrix_median(matrix, **kwargs):
    '''

    :param matrix: matrix or vector
    :param kwargs: Can keywords that are accepted by `matrix_checking` function
    :type matrix: np.ndarray
    :type kwargs: dict
    :returns: The median column values in the matrix.
    :rtype: np.ndarray
    '''

    return np.median(matrix, axis=0)

@inf_nan_check
@matrix_checking
def matrix_std(matrix, **kwargs):
    '''
    :param matrix: matrix or vector
    :param kwargs: Can keywords that are accepted by `matrix_checking` function
    :type matrix: np.ndarray
    :type kwargs: dict
    :returns: The standard deviation of the column values in the matrix.
    :rtype: np.ndarray
    '''
    return np.std(matrix, axis=0)

@inf_nan_check
@matrix_checking
def matrix_prod(matrix, **kwargs):
    '''
    :param matrix: matrix or vector
    :param kwargs: Can keywords that are accepted by `matrix_checking` function
    :type matrix: np.ndarray
    :type kwargs: dict
    :returns: The product of the column values in the matrix.
    :rtype: np.ndarray
    '''
    return np.prod(matrix, axis=0)
