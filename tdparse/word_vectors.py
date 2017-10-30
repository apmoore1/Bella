'''
Contains classes that train and/or load semantic vectors.


Classes:
1. GensimVectors - Creates `Word2Vec <https://arxiv.org/pdf/1301.3781.pdf>`_
and `FastText <https://arxiv.org/abs/1607.04606>`_ vectors.
'''

import os

import numpy as np

from gensim.models import word2vec
from gensim.models.wrappers import FastText

class GensimVectors:
    '''
    Class that can create one of the following word vector models:

    1. `Word2Vec <https://radimrehurek.com/gensim/models/word2vec.html>`_.
    2. `Fasttext <https://radimrehurek.com/gensim/models/wrappers/fasttext.html>`_

    instance attributes:
    1. self.model = Instance of the chosen model
    2. self.vector_size = Size of the word vectors computed e.g. 100
    '''

    def __init__(self, file_path, train_data, model=None, **kwargs):
        '''
        Trains or loads the model specified.

        :param file_path: Path to the saved model or Path to save the model
        once trained. Can be None if you only want to train.
        :param train_data: An object like a list that can be iterated that contains
        tokenised text, to train the model. Not required if `file_path` contains
        a trained model.
        :param model: The name of the model
        :type file_path: String
        :type train_data: iterable object e.g. list
        :type model: String
        '''

        allowed_models = {'word2vec' : word2vec.Word2Vec,
                          'fasttext' : FastText}
        if model not in allowed_models:
            raise ValueError('model parameter has to be one of the following {} '\
                             'not {}'.format(allowed_models.keys(), model))
        model = allowed_models[model]

        file_path = os.path.abspath(file_path)

        if os.path.isfile(file_path):
            self.model = model.load(file_path)
        elif hasattr(train_data, '__iter__'):
            self.model = model(train_data, **kwargs)
            if isinstance(file_path, str):
                self.model.save(file_path)
                print('{} model has been saved to {}'.format(model.__name__,
                                                             file_path))
        else:
            raise Exception('Cannot create model as there is no path to extract '\
                            'a model from {} or any data to train on which has '\
                            'to have the __iter__ function {}'\
                            .format(file_path, train_data))
        self.vector_size = self.model.wv[self.model.index2word[0]].shape[0]

    def lookup_vector(self, word):
        '''
        Given a word returns the vector representation of that word. If the model
        does not have a representation for that word returns a vector of zeros.

        :param word: A word
        :type word: String
        :returns: The word vector for that word. If no word vector can be found
        returns a vector of zeros.
        :rtype: numpy.ndarray
        '''

        if isinstance(word, str):
            try:
                return self.model.wv[word]
            except KeyError:
                return np.zeros(self.vector_size)
        raise ValueError('The word parameter must be of type str not {}'\
                         .format(type(word)))
