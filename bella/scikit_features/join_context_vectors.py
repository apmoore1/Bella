import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella import neural_pooling

class JoinContextVectors(BaseEstimator, TransformerMixin):

    def __init__(self, pool_func=neural_pooling.matrix_median):
        self.pool_func = pool_func

    def fit(self, context_pool_vectors, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, context_pool_vectors, y=None):
        '''see self.transform'''

        return self.transform(context_pool_vectors)

    def transform(self, context_pool_vectors):
        '''
        Given a list of train data which contain a list of numpy.ndarray one for
        each context. Return a list of train data of numpy.ndarray which are the
        contexts joined together using one of the pool functions.
        '''
        vec_size = context_pool_vectors[0].shape[1]
        train_vectors = []
        len_train_data = len(context_pool_vectors)
        for context_pool_vector in context_pool_vectors:
            train_vectors.append(self.pool_func(context_pool_vector))
        train_vectors = np.asarray(train_vectors).reshape(len_train_data, vec_size)
        return train_vectors
