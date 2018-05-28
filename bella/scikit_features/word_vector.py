import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella.word_vectors import WordVectors

class ContextWordVectors(BaseEstimator, TransformerMixin):

    def __init__(self, vectors=None, zero_token='$$$ZERO_TOKEN$$$'):
        self.vectors = vectors
        self.zero_token = zero_token

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
        context_word_vectors = []
        for context_tokens in contexts_tokens:
            all_contexts = []
            for context in context_tokens:
                context_word_vector = []
                for token in context:
                    token_vector = []
                    for word_vector in self.vectors:
                        token_vector.append(word_vector.lookup_vector(token))
                    context_word_vector.append(np.hstack(token_vector))
                # Padding
                if len(context_word_vector) == 0:
                    token_vector = []
                    for word_vector in self.vectors:
                        token_vector.append(word_vector.lookup_vector(self.zero_token))
                    context_word_vector.append(np.hstack(token_vector))

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
