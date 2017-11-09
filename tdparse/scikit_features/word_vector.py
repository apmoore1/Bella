import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from tdparse.word_vectors import WordVectors

class ContextWordVectors(BaseEstimator, TransformerMixin):

    def __init__(self, word_vector):
        self.word_vector = word_vector

    def fit(self, context_tokens, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, context_tokens, y=None):
        '''see self.transform'''

        return self.transform(context_tokens)

    def transform(self, contexts_tokens):
        '''
        Given a list of contexts (either right, left or target) which are made
        up of lists of tokens return the tokens as a word vector matrix.

        The word vector matrix is a word vector for each token but instead of
        storing in a list it is stored in a numpy.ndarray of shape:
        (length of word vector, number of tokens).

        Example of the input
        [[['context', 'one'], ['context', 'two']], [['another context']]]

        :param contexts_tokens: A list of data of which each data contains a list \
        of contexts which contains a list of tokens.
        :type context_tokens: list
        :returns: The same list but with word vectors as numpy.ndarray instead \
        of tokens which are Strings
        :rtype: list
        '''
        word_vector_size = self.word_vector.vector_size
        context_word_vectors = []
        for context_tokens in contexts_tokens:
            all_contexts = []
            for context in context_tokens:
                context_word_vector = []
                for token in context:
                    context_word_vector.append(self.word_vector.lookup_vector(token))
                # Padding
                if len(context_word_vector) == 0:
                    context_word_vector.append(np.zeros(word_vector_size))
                context_matrix = self.list_to_matrix(context_word_vector)
                all_contexts.append(context_matrix)
            context_word_vectors.append(all_contexts)
        return context_word_vectors

    @staticmethod
    def list_to_matrix(word_vector_list):
        '''
        Converts a list of numpy.ndarrays (vectors) into a numpy.ndarray (matrix).

        :param word_vector_list: list of numpy.ndarray
        :type word_vector_list: list
        :returns: a matrix of the numpy.ndarray
        :rtype: numpy.ndarray
        '''

        num_rows = len(word_vector_list)
        matrix = np.asarray(word_vector_list)
        if matrix.shape[0] != num_rows:
            raise ValueError('The matrix row should equal the number of tokens')
        return matrix
