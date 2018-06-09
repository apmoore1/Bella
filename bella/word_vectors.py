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
import math
import types
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm

BELLA_VEC_DIR = Path.home().joinpath('.Bella', 'Vectors')


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
    8. unit_length - If the vectors when returned are their unit norm value \
    instead of their raw values.
    9. unknown_index - The index of the unknown word normally 0
    10. padding_word - The word (<pad>) that defines padding indexs.
    11. padding_index - index of the padding word
    12. padding vector - padding vector for the padding word.

    padding index, word and vector are equal to the unknown equilavents if
    padding_value = None in the constructor. Else padding index = 0, word = <pad>
    and vector is what you have defined, this then moves the unknown index to
    vocab size + 1 and the everything else is the same. The idea behind the 2
    are that pad is used for padding and unknown is used for words that are
    truely unknown therefore allowing you to only skip the pad vectors when
    training a model by using a masking function in keras.

    The index 0 is used as a special index to map to unknown words. Therefore \
    the size of the vocabularly is len(index2word) - 1.

    Following methods:

    1. :py:func:`bella.word_vectors.WordVectors.lookup_vector`
    '''
    def __init__(self, word2vector, name=None, unit_length=False,
                 padding_value=None, filter_words=None):
        self.filter_words = [] if filter_words is None else filter_words
        size_vector_list = self._check_vector_size(word2vector)
        self.vector_size, self._word2vector, self._word_list = size_vector_list
        self.name = '{}'.format(name)
        self.unit_length = unit_length
        # Method attributes
        self.unknown_word = self._unknown_word()
        self.unknown_vector = self._unknown_vector()
        self.unknown_index = 0
        self.padding_word = self.unknown_word
        self.padding_vector = None
        if padding_value is not None:
            self.unknown_index = len(self._word_list) + 1
            self.padding_vector = np.asarray([padding_value] *
                                             self.vector_size)
            self.padding_word = '<pad>'
        else:
            self.padding_vector = self.unknown_vector
        if self.padding_vector is None:
            raise ValueError('Padding Vector is None')
        self.index2word = self._index2word()
        self.word2index = self._word2index()
        self.index2vector = self._index2vector()
        self.embedding_matrix = self._embedding_matrix()

    def _keyed_vec_2_dict(self, key_vector):
        word_2_vec_dict = {}
        for word in key_vector.vocab:
            word_2_vec_dict[word] = key_vector[word]
        return word_2_vec_dict

    def _check_vector_size(self, word2vector):
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
            word2vector = self._keyed_vec_2_dict(word2vector)
            #accepted_words = word2vector.index2word
            #most_likely_size = word2vector[accepted_words[0]].shape[0]
            #return most_likely_size, word2vector, accepted_words
        vector_sizes = {}
        for _, vector in word2vector.items():
            vector_size = vector.shape[0]
            vector_sizes[vector_size] = vector_sizes.get(vector_size, 0) + 1
        most_likely_size = sorted(vector_sizes.items(), reverse=True,
                                  key=lambda size_freq: size_freq[0])[0][0]
        words_to_remove = []
        unk_word = self._unknown_word()
        accepted_words = []
        for word, vector in word2vector.items():
            if vector.shape[0] != most_likely_size:
                words_to_remove.append(word)
            elif self.filter_words != []:
                if word not in self.filter_words and word != unk_word:
                    words_to_remove.append(word)
                else:
                    accepted_words.append(word)
            else:
                accepted_words.append(word)
        for word in words_to_remove:
            del word2vector[word]
        return most_likely_size, word2vector, accepted_words

    @staticmethod
    def glove_txt_binary(glove_file_path: Path):
        '''
        Converts the Glove word embedding file which is a text file to a
        binary file that can be loaded through gensims
        KeyedVectors.load_word2vec_format method and deletes the text file
        version and returns the file path to the new binary file.

        :param glove_file_path: File path to the downloaded glove vector text \
        file.
        :type glove_file_path: String
        :returns: The file path to the binary file version of the glove vector
        :rtype: String
        '''
        # File path to the binary file
        binary_file_path = os.path.splitext(glove_file_path.name)
        binary_file_path = binary_file_path[0]
        binary_file_path += '.binary'
        binary_file_path = glove_file_path.parent.joinpath(binary_file_path)
        if binary_file_path.is_file():
            return str(binary_file_path.resolve())
        with tempfile.NamedTemporaryFile('w', encoding='utf-8') as temp_file:
            # Converts to word2vec file format
            print('Converting word vectors file from text to binary for '
                  'quicker load time')
            glove2word2vec(str(glove_file_path.resolve()), temp_file.name)
            word_vectors = KeyedVectors.load_word2vec_format(temp_file.name,
                                                             binary=False)
            # Creates the binary file version of the word vectors
            binary_file_path = str(binary_file_path.resolve())
            word_vectors.save_word2vec_format(binary_file_path, binary=True)
        # Delete the text version of the glove vectors
        glove_file_path.unlink()
        return binary_file_path

    def lookup_vector(self, word):
        '''
        Given a word returns the vector representation of that word. If the model
        does not have a representation for that word it returns word vectors
        unknown word vector (most models this is zeors)

        :param word: A word
        :type word: String
        :returns: The word vector for that word. If no word vector can be found
        returns a vector of zeros.
        :rtype: numpy.ndarray
        '''

        if isinstance(word, str):
            word_index = self.word2index[word]
            return self.index2vector[word_index]
        raise ValueError('The word parameter must be of type str not '
                         f'{type(word)}')

    def _index2word(self):
        '''
        The index starts at one as zero is a special value assigned to words that
        are padded. The word return for zero is defined by `padded_word` attribute.
        If the padded value is different to the unknown word value then the index
        will contain an extra index for the unknown word index which is the
        vocab size + 1 index.

        :returns: A dictionary matching word indexs to there corresponding words.
        Inverse of :py:func:`bella.word_vectors.GensimVectors.word2index`
        :rtype: dict
        '''

        index_word = {}
        index_word[0] = self.padding_word
        index = 1
        for word in self._word_list:
            if word == self.unknown_word:
                continue
            index_word[index] = word
            index += 1
        # Required as the unknown word might have been learned and in
        # self._word_list
        if self.unknown_index != 0:
            self.unknown_index = len(index_word)
        index_word[self.unknown_index] = self.unknown_word
        return index_word

    def _return_unknown_index(self):
        '''
        :returns: zero. Used as lambda is not pickleable
        :rtype: int
        '''
        return self.unknown_index

    def _word2index(self):
        '''
        If you have specified a special padded index vector then the <pad> word
        would match to index 0 and the vocab + 1 index will be <unk> else if
        no special pad index vector then vocab + 1 won't exist and <unk> will be
        0

        :returns: A dictionary matching words to there corresponding index.
        Inverse of :py:func:`bella.word_vectors.GensimVectors.index2word`
        :rtype: dict
        '''

        # Cannot use lambda function as it cannot be pickled
        word2index_dict = defaultdict(self._return_unknown_index)
        for index, word in self.index2word.items():
            word2index_dict[word] = index
        return word2index_dict

    def _index2vector(self):
        '''
        NOTE: the zero index is mapped to the unknown index unless padded vector
        is specified then zero index is padded index and vocab + 1 index is
        unknown index

        :returns: A dictionary of word index to corresponding word vector. Same
        as :py:func:`bella.word_vectors.GensimVectors.lookup_vector` but
        instead of words that are looked up it is the words index.
        :rtype: dict
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

        index_vector = {}
        if self.unit_length:
            index_vector[0] = unit_norm(self.padding_vector)
            index_vector[self.unknown_index] = unit_norm(self.unknown_vector)
        else:
            index_vector[0] = self.padding_vector
            index_vector[self.unknown_index] = self.unknown_vector
        for index, word in self.index2word.items():
            if index == 0 or index == self.unknown_index:
                continue
            if self.unit_length:
                index_vector[index] = unit_norm(self._word2vector[word])
            else:
                index_vector[index] = self._word2vector[word]
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
                print(f'{word} {index} {vector} {e}')
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
                 unit_length=False, padding_value=None,
                 filter_words=None, **kwargs):
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

        allowed_models = {'word2vec': word2vec.Word2Vec,
                          'fasttext': FastText}
        if model not in allowed_models:
            raise ValueError('model parameter has to be one of the following '
                             f'{allowed_models.keys()} not {model}')
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
        super().__init__(self._model.wv, name=name, unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)


class VoVectors(GensimVectors):

    def download(self, skip_conf):
        vo_folder = BELLA_VEC_DIR.joinpath('Vo Word Vectors')
        vo_folder.mkdir(parents=True, exist_ok=True)
        current_vector_files = set([vo_file.name for vo_file
                                    in vo_folder.iterdir()])
        all_vector_files = set(['c10_w3_s100',
                                'c10_w3_s100.syn0.npy',
                                'c10_w3_s100.syn1.npy'])
        interset = all_vector_files.intersection(current_vector_files)
        # If the files in the folder aren't all the glove files that would be
        # downloaded re-download the zip and unzip the files.
        if interset != all_vector_files:
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the Vo Word vectors '\
                               'this is a download of 120MB '\
                               'Would you like to continue? If so type '\
                               '`yes`\n>> '
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                base_url = 'https://github.com/bluemonk482/tdparse/raw/'\
                           'master/resources/wordemb/w2v/'
                link_locations = [(f'{base_url}c10_w3_s100',
                                   vo_folder.joinpath('c10_w3_s100')),
                                  (f'{base_url}c10_w3_s100.syn0.npy',
                                   Path(vo_folder, 'c10_w3_s100.syn0.npy')),
                                  (f'{base_url}c10_w3_s100.syn1.npy',
                                   Path(vo_folder, 'c10_w3_s100.syn1.npy'))]
                print('Downloading Vo vectors')
                for download_link, file_location in link_locations:
                    # Reference:
                    # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                    with file_location.open('wb') as vo_file:
                        request = requests.get(download_link, stream=True)
                        total_size = int(request.headers.get('content-length',
                                                             0))
                        for chunk in tqdm(request.iter_content(chunk_size=128),
                                          total=math.ceil(total_size//128)):
                            vo_file.write(chunk)
            else:
                raise Exception('Vo vectors not downloaded therefore '
                                'cannot load them')
        return str(vo_folder.joinpath('c10_w3_s100').resolve())

    def __init__(self, name=None, unit_length=False,
                 padding_value=None, skip_conf=False,
                 filter_words=None):
        vector_file = self.download(skip_conf)

        if name is None:
            name = 'w2v'
        super().__init__(vector_file, train_data=None, name=name,
                         model='word2vec', unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)



class PreTrained(WordVectors):
    '''
    Class that loads word vectors that have been pre-trained.

    All pre-trained word vectors have to follow the following conditions:

    1. New word vector on each line
    2. Each line is tab seperated (by default but a tab is just delimenter \
    which can be changed by setting delimenter argument in the constructor)
    3. The first tab sperated value on the line is the word
    4. The rest of the tab seperated values on that line represent the values
    for the associtaed word.
    '''

    def __init__(self, file_path, name=None, unit_length=False,
                 delimenter='\t', padding_value=None, filter_words=None):
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
        super().__init__(word2vector, name=name, unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)

    def _unknown_vector(self):
        '''
        Overrides. Instead of returnning zero vector it return the vector for
        the word `<unk>`.

        :returns: The vector for the word `<unk>`
        :rtype: numpy.ndarray
        '''

        return self._word2vector['<unk>']


class SSWE(PreTrained):

    def download(self, skip_conf):
        '''
        '''
        sswe_folder = BELLA_VEC_DIR.joinpath('SSWE')
        sswe_folder.mkdir(parents=True, exist_ok=True)
        sswe_fp = sswe_folder.joinpath('sswe')
        # If the files in the folder aren't all the SSWE files that would be
        # downloaded re-download the zip and unzip the files.
        if not sswe_fp.is_file():
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the SSWE vectors '\
                               'this is a download of 74MB '\
                               'Would you like to continue? If so type '\
                               '`yes`\n>> '
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                download_link = 'https://github.com/bluemonk482/tdparse/raw/'\
                                'master/resources/wordemb/sswe/sswe-u.txt'
                # Reference:
                # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                with sswe_fp.open('wb') as sswe_file:
                    request = requests.get(download_link, stream=True)
                    total_size = int(request.headers.get('content-length', 0))
                    print('Downloading SSWE vectors')
                    for chunk in tqdm(request.iter_content(chunk_size=128),
                                      total=math.ceil(total_size//128)):
                        sswe_file.write(chunk)
            else:
                raise Exception('SSWE vectors not downloaded therefore '
                                'cannot load them')
            sswe_folder_files = list(sswe_folder.iterdir())
            if not sswe_fp.is_file():
                raise Exception('Error in either downloading the SSWE vectors'
                                ' or file path names. Files in the SSWE '
                                f'folder {sswe_folder_files} and where the '
                                f'SSWE file should be {str(sswe_fp)}')
        return str(sswe_fp.resolve())

    def __init__(self, name=None, unit_length=False, skip_conf=False,
                 padding_value=None, filter_words=None):

        vector_file = self.download(skip_conf)

        if name is None:
            name = 'sswe'
        super().__init__(vector_file, name=name, unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)



class GloveTwitterVectors(WordVectors):

    def download(self, skip_conf):
        '''
        This method checks if the
        `Glove Twitter word vectors \
        <https://nlp.stanford.edu/projects/glove/>`_
        are already in the repoistory if not it downloads and unzips the word
        vectors if permission is granted and converts them into a gensim
        KeyedVectors binary representation.

        :param skip_conf: Whether to skip the permission step as it requires \
        user input. True to skip permission.
        :type skip_conf: bool
        :returns: A dict containing word vector dimension as keys and the \
        absolute path to the vector file.
        :rtype: dict
        '''

        glove_folder = BELLA_VEC_DIR.joinpath('glove_twitter')
        glove_folder.mkdir(parents=True, exist_ok=True)
        current_glove_files = set([glove_file.name for glove_file
                                   in glove_folder.iterdir()])
        all_glove_files = set(['glove.twitter.27B.25d.binary',
                               'glove.twitter.27B.50d.binary',
                               'glove.twitter.27B.100d.binary',
                               'glove.twitter.27B.200d.binary'])
        interset = all_glove_files.intersection(current_glove_files)
        # If the files in the folder aren't all the glove files that would be
        # downloaded re-download the zip and unzip the files.
        if interset != all_glove_files:
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the glove vectors '\
                               'this is a large download of 1.4GB and takes '\
                               '5.4GB of disk space after being unzipped. '\
                               'Would you like to continue? If so type '\
                               '`yes`\n>> '
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                download_link = 'http://nlp.stanford.edu/data/glove.twitter.'\
                                '27B.zip'
                glove_zip_path = glove_folder.joinpath('glove_zip.zip')
                # Reference:
                # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                with glove_zip_path.open('wb') as glove_zip_file:
                    request = requests.get(download_link, stream=True)
                    total_size = int(request.headers.get('content-length', 0))
                    print('Downloading Glove Twitter vectors')
                    for chunk in tqdm(request.iter_content(chunk_size=128),
                                      total=math.ceil(total_size//128)):
                        glove_zip_file.write(chunk)
                print('Unzipping word vector download')
                glove_zip_path = str(glove_zip_path.resolve())
                with zipfile.ZipFile(glove_zip_path, 'r') as glove_zip_file:
                    glove_zip_file.extractall(path=glove_folder)
            else:
                raise Exception('Glove Twitter vectors not downloaded '
                                'therefore cannot load them')

        def add_full_path(file_name):
            file_path = glove_folder.joinpath(file_name)
            return self.glove_txt_binary(file_path)

        return {25: add_full_path('glove.twitter.27B.25d.txt'),
                50: add_full_path('glove.twitter.27B.50d.txt'),
                100: add_full_path('glove.twitter.27B.100d.txt'),
                200: add_full_path('glove.twitter.27B.200d.txt')}

    def __init__(self, dimension, name=None, unit_length=False,
                 skip_conf=False, padding_value=None,
                 filter_words=None):
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
            raise TypeError('Type of dimension has to be int not {}'
                            .format(type(dimension)))
        if dimension not in dimension_file:
            raise ValueError('Dimension avliable are the following {}'
                             .format(list(dimension_file.keys())))
        if name is None:
            name = f'glove twitter {dimension}d'
        vector_file = dimension_file[dimension]
        print(f'Loading {name} from file')
        glove_key_vectors = KeyedVectors.load_word2vec_format(vector_file,
                                                              binary=True)
        super().__init__(glove_key_vectors, name=name, unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)

    def _unknown_vector(self):
        '''
        This is to be Overridden by sub classes if they want to return a custom
        unknown vector.

        :returns: A vector for all unknown words. In this case it is a zero
        vector.
        :rtype: numpy.ndarray
        '''

        return np.zeros(self.vector_size, dtype=np.float32)


class GloveCommonCrawl(WordVectors):

    def download(self, skip_conf, version):
        '''
        This method checks if either the `Glove Common Crawl \
        <https://nlp.stanford.edu/projects/glove/>`_ 840 or 42 Billion token
        word vectors were downloaded already into the repoistory if not it
        downloads and unzips the 300 Dimension word vector if permission is
        granted.

        :param skip_conf: Whether to skip the permission step as it requires \
        user input. True to skip permission.
        :param version: Choice of either the 42 or 840 Billion token 300 \
        dimension common crawl glove vectors. The values can be only 42 or \
        840 and default is 42.
        :type skip_conf: bool
        :type version: int
        :returns: The filepath to the 300 dimension word vector
        :rtype: String
        '''

        glove_folder = BELLA_VEC_DIR.joinpath(f'glove_common_crawl_{version}b')
        glove_folder.mkdir(parents=True, exist_ok=True)
        glove_file_name = f'glove.{version}B.300d'
        glove_txt_fp = glove_folder.joinpath(glove_file_name + '.txt')
        glove_binary_fp = glove_folder.joinpath(glove_file_name + '.binary')
        # If the files in the folder aren't all the glove files that would be
        # downloaded re-download the zip and unzip the files.
        if not glove_binary_fp.is_file() and not glove_txt_fp.is_file():
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the glove vectors '\
                               'this is a large download of ~2GB and takes '\
                               '~5.6GB of diskspace after being unzipped. '\
                               'Would you like to continue? If so type '\
                               '`yes`\n>> '
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                zip_file_name = f'glove.{version}B.300d.zip'
                download_link = 'http://nlp.stanford.edu/data/' + zip_file_name

                glove_zip_path = glove_folder.joinpath(zip_file_name)

                # Reference:
                # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                with glove_zip_path.open('wb') as glove_zip_file:
                    request = requests.get(download_link, stream=True)
                    total_size = int(request.headers.get('content-length', 0))
                    print(f'Downloading Glove {version}B vectors')
                    for chunk in tqdm(request.iter_content(chunk_size=128),
                                      total=math.ceil(total_size//128)):
                        glove_zip_file.write(chunk)
                print('Unzipping word vector download')
                glove_zip_path = str(glove_zip_path.resolve())
                with zipfile.ZipFile(glove_zip_path, 'r') as glove_zip_file:
                    glove_zip_file.extractall(path=glove_folder)
            else:
                raise Exception(f'Glove Common Crawl {version}b vectors '
                                'not downloaded therefore cannot load them')
            glove_folder_files = list(glove_folder.iterdir())
            if not glove_txt_fp.is_file():
                raise Exception('Error in either downloading the glove vectors'
                                ' or file path names. Files in the glove '
                                f'folder {glove_folder_files} and where the '
                                f'golve file should be {str(glove_txt_fp)}')
        return self.glove_txt_binary(glove_txt_fp)

    def __init__(self, version=42, name=None, unit_length=False,
                 skip_conf=False, padding_value=None,
                 filter_words=None):
        '''
        :param version: Choice of either the 42 or 840 Billion token 300 \
        dimension common crawl glove vectors. The values can be only 42 or \
        840 and default is 42.
        :param skip_conf: Whether to skip the permission step for downloading \
        the word vectors as it requires user input. True to skip permission.
        :type version: int. Default 42.
        :type skip_conf: bool. Default False
        '''
        if version not in [42, 840]:
            raise ValueError('Common Crawl only come in two version the 840 '
                             'or 42 Billion tokens. Require to choose between '
                             f'42 and 840 and not {version}')

        if name is None:
            name = 'glove 300d {}b common crawl'.format(version)
        vector_file = self.download(skip_conf, version)
        print(f'Loading {name} from file')
        glove_key_vectors = KeyedVectors.load_word2vec_format(vector_file,
                                                              binary=True)
        super().__init__(glove_key_vectors, name=name, unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)

    def _unknown_vector(self):
        '''
        This is to be Overridden by sub classes if they want to return a custom
        unknown vector.

        :returns: A vector for all unknown words. In this case it is a zero
        vector.
        :rtype: numpy.ndarray
        '''

        return np.zeros(self.vector_size, dtype=np.float32)


class GloveWikiGiga(WordVectors):

    def download(self, skip_conf):
        '''
        This method checks if the
        `Glove Wikipedia Gigaword word vectors
        <https://nlp.stanford.edu/projects/glove/>`_
        are already in the repoistory if not it downloads and unzips the word
        vectors if permission is granted and converts them into a gensim
        KeyedVectors binary representation.

        :param skip_conf: Whether to skip the permission step as it requires \
        user input. True to skip permission.
        :type skip_conf: bool
        :returns: A dict containing word vector dimension as keys and the \
        absolute path to the vector file.
        :rtype: dict
        '''

        glove_folder = BELLA_VEC_DIR.joinpath(f'glove_wiki_giga')
        glove_folder.mkdir(parents=True, exist_ok=True)
        current_glove_files = set([glove_file.name for glove_file
                                   in glove_folder.iterdir()])
        all_glove_files = set(['glove.6B.50d.binary',
                               'glove.6B.100d.binary',
                               'glove.6B.200d.binary',
                               'glove.6B.300d.binary'])
        interset = all_glove_files.intersection(current_glove_files)
        # If the files in the folder aren't all the glove files that would be
        # downloaded re-download the zip and unzip the files.
        if interset != all_glove_files:
            can_download = 'yes'
            if not skip_conf:
                download_msg = 'We are going to download the glove vectors '\
                               'this is a large download of ~900MB and takes '\
                               '~2.1GB of disk space after being unzipped. '\
                               'Would you like to continue? If so type '\
                               '`yes`\n>> '
                can_download = input(download_msg)

            if can_download.strip().lower() == 'yes':
                download_link = 'http://nlp.stanford.edu/data/glove.6B.zip'
                glove_zip_path = glove_folder.joinpath('glove_zip.zip')
                # Reference:
                # http://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
                with glove_zip_path.open('wb') as glove_zip_file:
                    request = requests.get(download_link, stream=True)
                    total_size = int(request.headers.get('content-length', 0))
                    print('Downloading Glove Wikipedia Gigaword vectors')
                    for chunk in tqdm(request.iter_content(chunk_size=128),
                                      total=math.ceil(total_size//128)):
                        glove_zip_file.write(chunk)
                print('Unzipping word vector download')
                glove_zip_path = str(glove_zip_path.resolve())
                with zipfile.ZipFile(glove_zip_path, 'r') as glove_zip_file:
                    glove_zip_file.extractall(path=glove_folder)
            else:
                raise Exception('Glove Twitter vectors not downloaded '
                                'therefore cannot load them')

        def add_full_path(file_name):
            file_path = glove_folder.joinpath(file_name)
            return self.glove_txt_binary(file_path)

        return {50: add_full_path('glove.6B.50d.txt'),
                100: add_full_path('glove.6B.100d.txt'),
                200: add_full_path('glove.6B.200d.txt'),
                300: add_full_path('glove.6B.300d.txt')}

    def __init__(self, dimension, name=None, unit_length=False, skip_conf=False,
                 padding_value=None, filter_words=None):
        '''
        :param dimension: Dimension size of the word vectors you would like to \
        use. Choice: 50, 100, 200, 300
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
            name = 'glove wiki giga {}d'.format(dimension)
        vector_file = dimension_file[dimension]
        print(f'Loading {name} from file')
        glove_key_vectors = KeyedVectors.load_word2vec_format(vector_file,
                                                              binary=True)
        super().__init__(glove_key_vectors, name=name, unit_length=unit_length,
                         padding_value=padding_value,
                         filter_words=filter_words)

    def _unknown_vector(self):
        '''
        This is to be Overridden by sub classes if they want to return a custom
        unknown vector.

        :returns: A vector for all unknown words. In this case it is a zero
        vector.
        :rtype: numpy.ndarray
        '''

        return np.zeros(self.vector_size, dtype=np.float32)
