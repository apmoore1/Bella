'''
Contains classes that train and/or load semantic vectors. All classes are sub
classes of WordVectors

Classes:

1. WordVectors - Base class of all classes within this module. Ensures
consistent API for all word vectors classes.
2. GensimVectors - Creates `Word2Vec <https://arxiv.org/pdf/1301.3781.pdf>`_
and `FastText <https://arxiv.org/abs/1607.04606>`_ vectors.
3. PreTrained - Creates a Wordembedding for those that are stored in TSV files
where the first item in the line is the word and the rest of the tab sep values
are its vector representation. Currently loads the Tang et al. vectors
`from <https://github.com/bluemonk482/tdparse/tree/master/resources/wordemb/sswe>`_
'''

import os
import types

import numpy as np

from gensim.models import word2vec
from gensim.models.wrappers import FastText

class WordVectors(object):
    '''
    Base class for all WordVector classes. Contains the following instance
    attributes:

    1. vector_size - Size of the word vectors e.g. 100
    2. index2word - Mapping between index number and associated word
    3. index2vector - mapping between index and vector
    4. word2index - Mapping between word and associated index
    5. name - This is used to identify the model when reading cross validation \
    results from models, used mainly for debugging. Default is None. Used

    Following methods:

    1. :py:func:`tdparse.word_vectors.WordVectors.lookup_vector`
    '''
    def __init__(self, word2vector, word_list, name=None):
        if not isinstance(word_list, list):
            raise TypeError('word_list should be of type list not {}'\
                            .format(type(word_list)))
        try:
            word2vector[word_list[0]]
            if isinstance(word2vector, list):
                raise Exception('word2vector cannot be a list')
        except Exception as exception:
            raise TypeError('Exception {}: word2vector parameter should be a dict'\
                  ' type Structure that has word list values as keys'\
                  .format(exception))
        self._word2vector = word2vector
        self._word_list = word_list
        self.vector_size = word2vector[word_list[0]].shape[0]
        self.name = '{}'.format(name)

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
                return self._word2vector[word]
            except KeyError:
                return np.zeros(self.vector_size)
        raise ValueError('The word parameter must be of type str not {}'\
                         .format(type(word)))

    @property
    def index2word(self):
        '''
        :returns: A dictionary matching word indexs to there corresponding words.
        Inverse of :py:func:`tdparse.word_vectors.GensimVectors.word2index`
        :rtype: dict
        '''

        index_word = {}
        for index, word in enumerate(self._word_list):
            index_word[index] = word
        return index_word

    @property
    def word2index(self):
        '''
        :returns: A dictionary matching words to there corresponding index.
        Inverse of :py:func:`tdparse.word_vectors.GensimVectors.index2word`
        :rtype: dict
        '''

        return {word : index for index, word in self.index2word.items()}

    @property
    def index2vector(self):
        '''
        :returns: A dictionary of word index to corresponding word vector. Same
        as :py:func:`tdparse.word_vectors.GensimVectors.lookup_vector` but
        instead of words that are looked up it is the words index.
        :rtype: dict
        '''

        index_vector = {}
        for index, word in self.index2word.items():
            index_vector[index] = self.lookup_vector(word)
        return index_vector


class GensimVectors(WordVectors):
    '''
    Class that can create one of the following word vector models:

    1. `Word2Vec <https://radimrehurek.com/gensim/models/word2vec.html>`_.
    2. `Fasttext <https://radimrehurek.com/gensim/models/wrappers/fasttext.html>`_

    private attributes:

    1. self._model = Gensim instance of the chosen model e.g. if word2vec
    was chosen then it would be `gensim.models.word2vec.Word2Vec`
    '''

    def __init__(self, file_path, train_data, name=None, model=None, **kwargs):
        '''
        Trains or loads the model specified.

        :param file_path: Path to the saved model or Path to save the model
        once trained. Can be None if you only want to train.
        :param train_data: An object like a list that can be iterated that contains
        tokenised text, to train the model e.g. [['hello', 'how'], ['another']].
        Not required if `file_path` contains a trained model.
        :param model: The name of the model
        :param name: The name of given to the instance.
        :param kwargs: The keyword arguments to give to the Gensim Model that is \
        being used i.e. keyword argument to `Word2Vec <https://radimrehurek.com/\
        gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_
        :type file_path: String
        :type train_data: iterable object e.g. list
        :type model: String
        :type name: String Default None
        :type kwargs: dict
        '''

        allowed_models = {'word2vec' : word2vec.Word2Vec,
                          'fasttext' : FastText}
        if model not in allowed_models:
            raise ValueError('model parameter has to be one of the following {} '\
                             'not {}'.format(allowed_models.keys(), model))
        model = allowed_models[model]
        failed_to_load = True

        if isinstance(file_path, str):
            file_path = os.path.abspath(file_path)

        if isinstance(file_path, str):
            if os.path.isfile(file_path):
                try:
                    self._model = model.load(file_path)
                    failed_to_load = False
                except EOFError:
                    failed_to_load = True

        if hasattr(train_data, '__iter__') and failed_to_load:
            # Generators throws an error in Gensim
            if isinstance(train_data, types.GeneratorType):
                train_data = map(lambda x: x, train_data)
            self._model = model(train_data, **kwargs)
            if isinstance(file_path, str):
                self._model.save(file_path)
                print('{} model has been saved to {}'.format(model.__name__,
                                                             file_path))
        elif failed_to_load:
            raise Exception('Cannot create model as there is no path to extract '\
                            'a model from {} or any data to train on which has '\
                            'to have the __iter__ function {}'\
                            .format(file_path, train_data))
        super().__init__(self._model.wv, self._model.wv.index2word, name=name)

class PreTrained(WordVectors):
    '''
    Class that loads word vectors that have been pre-trained.

    All pre-trained word vectors have to follow the following conditions:

    1. New word vector on each line
    2. Each line is tab seperated
    3. The first tab sperated value on the line is the word
    4. The rest of the tab seperated values on that line represent the values
    for the associtaed word.
    '''

    def __init__(self, file_path, name=None):
        '''
        :param file_path: The file path to lad the word vectors from
        :param name: The name given to the instance.
        :type file_path: String
        :type name: String Default None
        '''
        if not isinstance(file_path, str):
            raise TypeError('The type of the file path should be str not {}'\
                            .format(type(file_path)))
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise ValueError('There is no file at file path {}'.format(file_path))

        word2vector = {}
        word_list = []
        with open(file_path, 'r') as data_file:
            for line in data_file:
                line = line.strip()
                word_values = line.split('\t')
                word = word_values[0]
                word_list.append(word)
                word_vector = np.asarray(word_values[1:], dtype='float32')
                if word in word2vector:
                    raise KeyError('{} already has a vector in the word vector '\
                                   'dict'.format(word))
                else:
                    word2vector[word] = word_vector
        super().__init__(word2vector, word_list, name=name)

    def lookup_vector(self, word):
        '''
        Given a word returns the vector representation of that word. If the model
        does not have a representation for that word returns the vector representation
        of `<unk>`

        Overridden without calling super as we want to change the return of the
        unknown word.

        :param word: A word
        :type word: String
        :returns: The word vector for that word. If no word vector can be found
        returns the vector of `<unk>`
        :rtype: numpy.ndarray
        '''

        if isinstance(word, str):
            try:
                return self._word2vector[word]
            except KeyError:
                return self._word2vector['<unk>']
        raise ValueError('The word parameter must be of type str not {}'\
                         .format(type(word)))
