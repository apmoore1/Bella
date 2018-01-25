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

from collections import defaultdict
import os
import types
import zipfile

import numpy as np
import requests
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
from gensim.models.wrappers import FastText

from tdparse import helper

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
    6. unknown_vector - The vector that is returned for any unknwon words and \
    for the index=0.
    7. unknown_word - The word that is returned for the 0 index. Default is \
    `<unk>`
    6. unit_length - If the vectors when returned are their unit norm value \
    instead of their raw values.

    The index 0 is used as a special index to map to unknown words. Therefore \
    the size of the vocabularly is len(index2word) - 1.

    Following methods:

    1. :py:func:`tdparse.word_vectors.WordVectors.lookup_vector`
    '''
    def __init__(self, word2vector, name=None, unit_length=False):
        size_vector_list = self._check_vector_size(word2vector)
        self.vector_size, self._word2vector, self._word_list = size_vector_list
        self.name = '{}'.format(name)
        self.unit_length = unit_length
        # Method attributes
        self.unknown_word = self._unknown_word()
        self.unknown_vector = self._unknown_vector()
        self.index2word = self._index2word()
        self.word2index = self._word2index()
        self.index2vector = self._index2vector()
        self.embedding_matrix = self._embedding_matrix()
    @staticmethod
    def _check_vector_size(word2vector):
        '''
        This finds the most common vector size in the word 2 vectors dictionary
        mapping. Normally they should all have the same mapping but it has been
        found that their could be some mistakes in pre-compiled word vectors
        therefore this function removes all words and vectors that do not conform
        to the majority vector size. An example of this would be if all the
        word to vector mappings are on dimension 50 but one word that one word
        would be removed from the dictionary mapping and not included in the
        word list returned.

        :param word2vector: A dictionary containing words as keys and their \
        associated vector representation as values.
        :type word2vector: dict or gensim.models.keyedvectors.KeyedVectors
        :returns: A tuple of length 3 containg 1. The dimension of the vectors, \
        2. The dictionary of word to vectors, 3. The list of words in the dictionary
        :rtype: tuple
        '''
        # Gensim does not used a dictionary but a special class
        if isinstance(word2vector, KeyedVectors):
            accepted_words = word2vector.index2word
            most_likely_size = word2vector[accepted_words[0]].shape[0]
            return most_likely_size, word2vector, accepted_words
        vector_sizes = {}
        for _, vector in word2vector.items():
            vector_size = vector.shape[0]
            vector_sizes[vector_size] = vector_sizes.get(vector_size, 0) + 1
        most_likely_size = sorted(vector_sizes.items(), reverse=True,
                                  key=lambda size_freq: size_freq[0])[0][0]
        words_to_remove = []
        accepted_words = []
        for word, vector in word2vector.items():
            if vector.shape[0] != most_likely_size:
                words_to_remove.append(word)
            else:
                accepted_words.append(word)
        for word in words_to_remove:
            del word2vector[word]
        return most_likely_size, word2vector, accepted_words

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

        def unit_norm(vector):
            '''
            :param vector: A 1-dimension vector
            :type vector: numpy.ndarray
            :returns: The vector normalised to it's unit vector.
            :rtype: numpy.ndarray
            '''

            # Check it is not a zero vector
            if np.array_equal(np.zeros(self.vector_size), vector):
                return vector
            l2_norm = np.linalg.norm(vector)
            return vector / l2_norm

        if isinstance(word, str):
            try:
                word_vector = self._word2vector[word]
                if self.unit_length:
                    return unit_norm(word_vector)
                return word_vector
            except KeyError:
                unknown_vector = self.unknown_vector
                if self.unit_length:
                    return unit_norm(unknown_vector)
                return unknown_vector
        raise ValueError('The word parameter must be of type str not {}'\
                         .format(type(word)))

    def _index2word(self):
        '''
        The index starts at one as zero is a special value assigned to words that
        are unknown. The word return for zero is defined by `unknown_word` property.

        :returns: A dictionary matching word indexs to there corresponding words.
        Inverse of :py:func:`tdparse.word_vectors.GensimVectors.word2index`
        :rtype: dict
        '''

        index_word = {}
        for index, word in enumerate(self._word_list):
            index_word[index + 1] = word
        index_word[0] = self.unknown_word
        return index_word

    @staticmethod
    def _return_zero():
        '''
        :returns: zero. Used as lambda is not pickleable
        :rtype: int
        '''
        return 0

    def _word2index(self):
        '''
        :returns: A dictionary matching words to there corresponding index.
        Inverse of :py:func:`tdparse.word_vectors.GensimVectors.index2word`
        :rtype: dict
        '''

        # Cannot use lambda function as it cannot be pickled
        word2index_dict = defaultdict(self._return_zero)
        for index, word in self.index2word.items():
            word2index_dict[word] = index
        return word2index_dict

    def _index2vector(self):
        '''
        NOTE: the zero index is mapped to the unknown index
        :returns: A dictionary of word index to corresponding word vector. Same
        as :py:func:`tdparse.word_vectors.GensimVectors.lookup_vector` but
        instead of words that are looked up it is the words index.
        :rtype: dict
        '''

        index_vector = {}
        for index, word in self.index2word.items():
            index_vector[index] = self.lookup_vector(word)
        index_vector[0] = self.unknown_vector
        return index_vector

    def __repr__(self):
        return self.name

    def _unknown_vector(self):
        '''
        This is to be Overridden by sub classes if they want to return a custom
        unknown vector.

        :returns: A vector for all unknown words. In this case it is a zero
        vector.
        :rtype: numpy.ndarray
        '''

        return np.zeros(self.vector_size)


    def _unknown_word(self):
        '''
        :returns: The word that is returned for the 0 index.
        :rtype: String
        '''

        return '<unk>'

    def _embedding_matrix(self):
        '''
        The embedding matrix that can be used in Keras embedding layer as the
        weights. It is very much a simple lookup where the key is the word index.

        :retunrs: The embedding matrix of dimension (vocab_size + 1, vector_size) \
        where the vocab size is + 1 due to the unknown vector.
        :rtype: numpy.ndarray
        '''
        matrix = np.zeros((len(self.index2vector), self.vector_size),
                          dtype=np.float32)
        for index, vector in self.index2vector.items():
            try:
                matrix[index] = vector
            except Exception as e:
                word = self.index2word[index]
                print('{} {} {}'.format(word, index, vector))
        return matrix


class GensimVectors(WordVectors):
    '''
    Class that can create one of the following word vector models:

    1. `Word2Vec <https://radimrehurek.com/gensim/models/word2vec.html>`_.
    2. `Fasttext <https://radimrehurek.com/gensim/models/wrappers/fasttext.html>`_

    private attributes:

    1. self._model = Gensim instance of the chosen model e.g. if word2vec
    was chosen then it would be `gensim.models.word2vec.Word2Vec`
    '''

    def __init__(self, file_path, train_data, name=None, model=None,
                 unit_length=False, **kwargs):
        '''
        Trains or loads the model specified.

        :param file_path: Path to the saved model or Path to save the model
        once trained. Can be None if you only want to train.
        :param train_data: An object like a list that can be iterated that contains
        tokenised text, to train the model e.g. [['hello', 'how'], ['another']].
        Not required if `file_path` contains a trained model.
        :param model: The name of the model
        :param name: The name of given to the instance.
        :param unit_length: If the word vectors should be normalised to unit \
        vectors
        :param kwargs: The keyword arguments to give to the Gensim Model that is \
        being used i.e. keyword argument to `Word2Vec <https://radimrehurek.com/\
        gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_
        :type file_path: String
        :type train_data: iterable object e.g. list
        :type model: String
        :type name: String Default None
        :type unit_length: bool. Default False.
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
        super().__init__(self._model.wv, name=name, unit_length=unit_length)

class PreTrained(WordVectors):
    '''
    Class that loads word vectors that have been pre-trained.

    All pre-trained word vectors have to follow the following conditions:

    1. New word vector on each line
    2. Each line is tab seperated (by default but a tab is just delimenter which
    can be changed by setting delimenter argument in the constructor)
    3. The first tab sperated value on the line is the word
    4. The rest of the tab seperated values on that line represent the values
    for the associtaed word.
    '''

    def __init__(self, file_path, name=None, unit_length=False,
                 delimenter='\t'):
        '''
        :param file_path: The file path to load the word vectors from
        :param name: The name given to the instance.
        :param unit_length: If the word vectors should be normalised to unit \
        vectors
        :param delimenter: The value to be used to split the values in each line \
        of the word vectors.
        :type file_path: String
        :type name: String Default None
        :type unit_length: bool. Default False
        :type delimenter: String. Default `\t`
        '''
        if not isinstance(file_path, str):
            raise TypeError('The type of the file path should be str not {}'\
                            .format(type(file_path)))
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise ValueError('There is no file at file path {}'.format(file_path))

        word2vector = {}
        with open(file_path, 'r') as data_file:
            for org_line in data_file:
                line = org_line.strip()
                word_values = line.split(delimenter)
                word = word_values[0]
                # This attempts to remove words that have whitespaces in them
                # a sample of this problem can be found within the Glove
                # Common Crawl 840B vectors where \xa0name@domain.com ==
                # name@domain.com after strip is applied and they have different
                # vectors
                if word in word2vector:
                    org_word = org_line.split(delimenter)[0]
                    if word != org_word:
                        continue
                    else:
                        del word2vector[word]
                word_vector = np.asarray(word_values[1:], dtype='float32')
                if word in word2vector:
                    dict_vector = word2vector[word]
                    raise KeyError('{} already has a vector in the word vector '\
                                   'dict. Vector in dict {} and alternative vector {}'\
                                   .format(word, dict_vector, word_vector))
                else:
                    word2vector[word] = word_vector
        super().__init__(word2vector, name=name, unit_length=unit_length)

    def _unknown_vector(self):
        '''
        Overrides. Instead of returnning zero vector it return the vector for
        the word `<unk>`.

        :returns: The vector for the word `<unk>`
        :rtype: numpy.ndarray
        '''

        return self._word2vector['<unk>']

class GloveTwitterVectors(PreTrained):

    @staticmethod
    def download(skip_conf):
        '''
        This method checks if the
        `Glove Twitter word vectors <https://nlp.stanford.edu/projects/glove/>`_
        are already in the repoistory if not it downloads and unzips the word
        vectors if permission is granted.

        :param skip_conf: Whether to skip the permission step as it requires \
        user input. True to skip permission.
        :type skip_conf: bool
        :returns: A dict containing word vector dimension as keys and the \
        absolute path to the vector file.
        :rtype: dict
        '''

        glove_folder = os.path.join(helper.package_dir(), 'data', 'word_vectors',
                                    'glove_twitter')
        os.makedirs(glove_folder, exist_ok=True)
        current_glove_files = set(os.listdir(glove_folder))
        all_glove_files = set(['glove.twitter.27B.25d.txt',
                               'glove.twitter.27B.50d.txt',
                               'glove.twitter.27B.100d.txt',
                               'glove.twitter.27B.200d.txt'])
        interset = all_glove_files.intersection(current_glove_files)
        # If the files in the folder aren't all the glove files that would be
        # downloaded re-download the zip and unzip the files.
        if interset != all_glove_files:
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the glove vectors this is '\
                               'a large download of 1.4GB and takes 5.4GB of disk '\
                               'space after being unzipped. Would you like to '\
                               'continue? If so type `yes`\n'
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                download_link = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
                glove_zip_path = os.path.join(glove_folder, 'glove_zip.zip')
                # Reference:
                # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                with open(glove_zip_path, 'wb') as glove_zip_file:
                    glove_requests = requests.get(download_link, stream=True)
                    for chunk in glove_requests.iter_content(chunk_size=128):
                        glove_zip_file.write(chunk)
                with zipfile.ZipFile(glove_zip_path, 'r') as glove_zip_file:
                    glove_zip_file.extractall(path=glove_folder)
            else:
                raise Exception('Glove Twitter vectors not downloaded therefore'\
                                ' cannot load them')

        add_full_path = lambda vec_file: os.path.join(glove_folder, vec_file)
        return {25 : add_full_path('glove.twitter.27B.25d.txt'),
                50 : add_full_path('glove.twitter.27B.50d.txt'),
                100 : add_full_path('glove.twitter.27B.100d.txt'),
                200 : add_full_path('glove.twitter.27B.200d.txt')}



    def __init__(self, dimension, name=None, unit_length=False, skip_conf=False):
        '''
        :param dimension: Dimension size of the word vectors you would like to \
        use. Choice: 25, 50, 100, 200
        :param skip_conf: Whether to skip the permission step for downloading \
        the word vectors as it requires user input. True to skip permission.
        :type dimension: int
        :type skip_conf: bool. Default False
        '''

        dimension_file = self.download(skip_conf)
        if not isinstance(dimension, int):
            raise TypeError('Type of dimension has to be int not {}'\
                            .format(type(dimension)))
        if dimension not in dimension_file:
            raise ValueError('Dimension avliable are the following {}'\
                             .format(list(dimension_file.keys())))
        if name is None:
            name = 'glove twitter {}d'.format(dimension)
        vector_file = dimension_file[dimension]
        super().__init__(vector_file, name=name, unit_length=unit_length,
                         delimenter=' ')

    def _unknown_vector(self):
        '''
        This is to be Overridden by sub classes if they want to return a custom
        unknown vector.

        :returns: A vector for all unknown words. In this case it is a zero
        vector.
        :rtype: numpy.ndarray
        '''

        return np.zeros(self.vector_size, dtype=np.float32)

class GloveCommonCrawl(PreTrained):

    @staticmethod
    def download(skip_conf):
        '''
        This method checks if the
        `Glove Common Crawl 840B <https://nlp.stanford.edu/projects/glove/>`_
        word vectors are already in the repoistory if not it downloads and
        unzips the 300 Dimension word vector if permission is granted.

        :param skip_conf: Whether to skip the permission step as it requires \
        user input. True to skip permission.
        :type skip_conf: bool
        :returns: The filepath to the 300 dimension word vector
        :rtype: String
        '''

        glove_folder = os.path.join(helper.package_dir(), 'data', 'word_vectors',
                                    'glove_common_crawl_840b')
        os.makedirs(glove_folder, exist_ok=True)
        glove_file_path = os.path.join(glove_folder, 'glove.840B.300d.txt')
        # If the files in the folder aren't all the glove files that would be
        # downloaded re-download the zip and unzip the files.
        if not os.path.isfile(glove_file_path):
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the glove vectors this is '\
                               'a large download of 2GB and takes 5.6GB of disk '\
                               'space after being unzipped. Would you like to '\
                               'continue? If so type `yes`\n'
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                download_link = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
                glove_zip_path = os.path.join(glove_folder, 'glove.840B.300d.zip')
                # Reference:
                # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                with open(glove_zip_path, 'wb') as glove_zip_file:
                    glove_requests = requests.get(download_link, stream=True)
                    for chunk in glove_requests.iter_content(chunk_size=128):
                        glove_zip_file.write(chunk)
                with zipfile.ZipFile(glove_zip_path, 'r') as glove_zip_file:
                    glove_zip_file.extractall(path=glove_folder)
            else:
                raise Exception('Glove Common Crawl 840b vectors not downloaded '\
                                'therefore cannot load them')
            if not os.path.isfile(glove_file_path):
                raise Exception('Error in either downloading the glove vectors '\
                                'or file path names. Files in the glove folder '\
                                '{} and where the golve file should be {}'\
                                .format(os.listdir(glove_folder), glove_file_path))
        return glove_file_path

    def __init__(self, name=None, unit_length=False, skip_conf=False):
        '''
        :param skip_conf: Whether to skip the permission step for downloading \
        the word vectors as it requires user input. True to skip permission.
        :type skip_conf: bool. Default False
        '''

        glove_file = self.download(skip_conf)
        if name is None:
            name = 'glove 300d common crawl'
        super().__init__(glove_file, name=name, unit_length=unit_length,
                         delimenter=' ')

    def _unknown_vector(self):
        '''
        This is to be Overridden by sub classes if they want to return a custom
        unknown vector.

        :returns: A vector for all unknown words. In this case it is a zero
        vector.
        :rtype: numpy.ndarray
        '''

        return np.zeros(self.vector_size, dtype=np.float32)
