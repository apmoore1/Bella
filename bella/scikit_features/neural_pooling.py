import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella.neural_pooling import matrix_max

class NeuralPooling(BaseEstimator, TransformerMixin):

    def __init__(self, pool_func=matrix_max):
        self.pool_func = pool_func

    def fit(self, context_word_matrixs, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, context_word_matrixs, y=None):
        '''see self.transform'''

        return self.transform(context_word_matrixs)

    def transform(self, context_word_matrixs):
        context_pool_vectors = []
        for context_word_matrix in context_word_matrixs:
            all_contexts = []
            for word_matrix in context_word_matrix:
                all_contexts.append(self.pool_func(word_matrix))
            num_contexts = len(all_contexts)
            vec_size = all_contexts[0].shape[1]
            all_contexts = np.asarray(all_contexts).reshape(num_contexts, vec_size)
            context_pool_vectors.append(all_contexts)
        return context_pool_vectors
