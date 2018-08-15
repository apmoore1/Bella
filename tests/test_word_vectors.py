'''
Unit test suite for the :py:mod:`bella.word_vectors` module.
'''
from pathlib import Path
import os
from unittest import TestCase
import tempfile

import pytest
import numpy as np

from bella.helper import read_config
import bella.tokenisers as tokenisers
from bella.word_vectors import WordVectors
from bella.word_vectors import GensimVectors
from bella.word_vectors import PreTrained
from bella.word_vectors import GloveTwitterVectors
from bella.word_vectors import GloveCommonCrawl
from bella.word_vectors import VoVectors
from bella.word_vectors import SSWE

CONFIG_FP = Path(__file__).parent.joinpath('..', 'config.yaml')

class TestWordVectors(TestCase):
    '''
    Contains the following functions:

    1. test_wordvector_methods
    2. test_gensim_word2vec
    3. test_pre_trained
    '''

    def test_wordvector_methods(self):
        '''
        Tests the :py:class:`bella.word_vectors.WordVectors`
        '''

        hello_vec = np.asarray([0.5, 0.3, 0.4], dtype=np.float32)
        another_vec = np.asarray([0.3333, 0.2222, 0.1111])
        test_vectors = {'hello' : hello_vec,
                        'another' : another_vec}
        word_vector = WordVectors(test_vectors)

        vec_size = word_vector.vector_size
        self.assertEqual(vec_size, 3, msg='Vector size should be 3 not {}'\
                         .format(vec_size))

        # Testing the methods
        hello_lookup = word_vector.lookup_vector('hello')
        self.assertEqual(True, np.array_equal(hello_lookup, hello_vec), msg='{} '\
                         'should equal {}'.format(hello_lookup, hello_vec))

        zero_vec = np.zeros(3)
        nothing_vec = word_vector.lookup_vector('nothing')
        self.assertEqual(True, np.array_equal(zero_vec, nothing_vec), msg='{} '\
                         'should be a zero vector'.format(nothing_vec))

        index2word = word_vector.index2word
        word2index = word_vector.word2index
        index2vector = word_vector.index2vector
        embedding_matrix = word_vector.embedding_matrix

        another_index = word2index['another']
        self.assertEqual('another', index2word[another_index], msg='index2word '\
                         'and word2index do not match on word `another`')
        index_correct = np.array_equal(index2vector[another_index], another_vec)
        self.assertEqual(True, index_correct, msg='index2vector does not return '\
                         'the correct vector for `another`')

        # Check that it returns 0 index for unknown words
        self.assertEqual(0, word2index['nothing'], msg='All unknown words should '\
                         'be mapped to the zero index')
        # Check that unknown words are mapped to the <unk> token
        self.assertEqual('<unk>', index2word[0], msg='All zero index should be '\
                         'mapped to the <unk> token')
        # Check that the unkown words map to the unknown vector
        self.assertEqual(True, np.array_equal(np.zeros(3), index2vector[0]),
                         msg='Zero index should map to the unknown vector')

        # Test the embedding matrix
        hello_index = word2index['hello']
        is_hello_vector = np.array_equal(hello_vec, embedding_matrix[hello_index])
        self.assertEqual(True, is_hello_vector, msg='The embedding matrix lookup'\
                         ' is wrong for the `hello` word')
        unknown_index = word2index['nothing']
        is_nothing_vector = np.array_equal(zero_vec, embedding_matrix[unknown_index])
        self.assertEqual(True, is_nothing_vector, msg='The embedding matrix lookup'\
                         ' is wrong for the unknwon word')

    def test_unit_norm(self):
        '''
        Testing the unit_length of WordVectors
        '''

        hello_vec = np.asarray([0.5, 0.3, 0.4], dtype=np.float32)
        another_vec = np.asarray([0.3333, 0.2222, 0.1111], dtype=np.float32)
        test_vectors = {'hello' : hello_vec,
                        'another' : another_vec}
        word_vector = WordVectors(test_vectors, unit_length=True)
        # Tests the normal case
        unit_hello_vec = np.asarray([0.70710677, 0.4242641,
                                     0.56568545], dtype=np.float32)
        unit_is_equal = np.array_equal(unit_hello_vec,
                                       word_vector.lookup_vector('hello'))
        self.assertEqual(True, unit_is_equal, msg='Unit vector is not working')
        # Test the l2 norm of a unit vector is 1
        test_unit_mag = np.linalg.norm(word_vector.lookup_vector('hello'))
        self.assertEqual(1.0, test_unit_mag, msg='l2 norm of a unit vector '\
                         'should be 1 not {}'.format(test_unit_mag))

        # Test that it does not affect zero vectors these should still be zero
        unknown_vector = word_vector.lookup_vector('nothing')
        self.assertEqual(True, np.array_equal(np.zeros(3), unknown_vector),
                         msg='unknown vector should be a zero vector and not {}'\
                         .format(unknown_vector))

        hello_index = word_vector.word2index['hello']
        hello_embedding = word_vector.embedding_matrix[hello_index]
        self.assertEqual(True, np.array_equal(unit_hello_vec, hello_embedding),
                         msg='The embedding matrix is not applying the unit '\
                         'normalization {} should be {}'\
                         .format(hello_embedding, unit_hello_vec))

    def test_padded_vector(self):
        hello_vec = np.asarray([0.5, 0.3, 0.4], dtype=np.float32)
        another_vec = np.asarray([0.3333, 0.2222, 0.1111])
        test_vectors = {'hello' : hello_vec,
                        'another' : another_vec}

        pad_vec = np.asarray([-1, -1, -1], dtype=np.float32)
        word_vector = WordVectors(test_vectors, padding_value=-1)
        self.assertEqual('<pad>', word_vector.index2word[0])
        self.assertEqual('<unk>', word_vector.index2word[3])
        anno_unk_vec = np.array_equal(np.zeros(3),
                                      word_vector.lookup_vector('anno'))
        self.assertEqual(3, word_vector.word2index['anno'])
        self.assertEqual(True, anno_unk_vec)
        embedding_matrix = word_vector.embedding_matrix
        pad_emb_vec = np.array_equal(pad_vec, embedding_matrix[0])
        unk_emb_vec = np.array_equal(np.zeros(3), embedding_matrix[3])
        hello_emb_vec = np.array_equal(hello_vec, embedding_matrix[1])
        self.assertEqual(True, pad_emb_vec)
        self.assertEqual(True, unk_emb_vec)
        self.assertEqual(True, hello_emb_vec)
        self.assertEqual('<pad>', word_vector.index2word[0])
        self.assertEqual('<unk>', word_vector.index2word[3])
        self.assertEqual(3, word_vector.unknown_index)

        pad_vec = np.asarray([-1]*100, dtype=np.float32)
        vo_zhang = VoVectors(skip_conf=True, padding_value=-1)
        vo_zhang_unk_index = vo_zhang.unknown_index
        embedding_matrix = vo_zhang.embedding_matrix
        pad_emb_vec = np.array_equal(pad_vec, embedding_matrix[0])
        unk_emb_vec = np.array_equal(vo_zhang.unknown_vector, embedding_matrix[vo_zhang_unk_index])
        unk_not_equal_pad = np.array_equal(embedding_matrix[0], vo_zhang.unknown_vector)
        self.assertEqual(True, pad_emb_vec, msg='{} {}'.format(pad_vec, embedding_matrix[0]))
        self.assertEqual(True, unk_emb_vec)
        self.assertEqual(True, hello_emb_vec)
        self.assertEqual(True, vo_zhang_unk_index != 0)
        self.assertEqual(True, vo_zhang_unk_index != 0)
        self.assertEqual(False, unk_not_equal_pad)
        self.assertEqual('<pad>', vo_zhang.index2word[0])
        self.assertEqual('<unk>', vo_zhang.index2word[vo_zhang_unk_index])

        # Ensure that padding does not affect word vectors that do not state
        # it is required
        word_vector = WordVectors(test_vectors)
        self.assertEqual('<unk>', word_vector.index2word[0])
        self.assertEqual('<unk>', word_vector.index2word[word_vector.unknown_index])
        self.assertEqual('<unk>', word_vector.padding_word)



    def test_gensim_word2vec(self):
        '''
        Tests the :py:class:`bella.word_vectors.GensimVectors`
        '''

        # Test loading word vectors from a file
        vo_zhang = VoVectors(skip_conf=True)

        self.assertEqual(vo_zhang.vector_size, 100, msg='Vector size should be equal'\
                         ' to 100 not {}'.format(vo_zhang.vector_size))
        # Check zero vectors work for OOV words
        zero_vector = np.zeros(100)
        oov_word = 'thisssssdoesssssnotexists'
        oov_vector = vo_zhang.lookup_vector(oov_word)
        self.assertEqual(True, np.array_equal(oov_vector, zero_vector),
                         msg='This word {} should not exists and have a zero '\
                         'vector and not {}'.format(oov_word, oov_vector))
        # Check it does get word vectors
        the_vector = vo_zhang.lookup_vector('the')
        self.assertEqual(False, np.array_equal(the_vector, zero_vector),
                         msg='The word `the` should have a non-zero vector.')

        with self.assertRaises(ValueError, msg='Should raise a value for any param'\
                               'that is not a String and this is a list'):
            vo_zhang.lookup_vector(['the'])

        # Check if the word, index and vector lookups match
        index_word = vo_zhang.index2word
        word_index = vo_zhang.word2index
        the_index = word_index['the']
        self.assertEqual('the', index_word[the_index], msg='index2word and '\
                         'word2index do not match for the word `the`')
        index_vector = vo_zhang.index2vector
        the_vectors_match = np.array_equal(index_vector[the_index],
                                           vo_zhang.lookup_vector('the'))
        self.assertEqual(True, the_vectors_match, msg='index2vector does not match'\
                         ' lookup_vector func for the word `the`')

        # Test the constructor
        test_file_path = 'this'
        with self.assertRaises(Exception, msg='The file path should have no saved '\
                               'word vector file {} and there is no training data'\
                               .format(test_file_path)):
            GensimVectors(test_file_path, 'fake data', model='word2vec')
        with self.assertRaises(Exception, msg='Should not accept neither no saved '\
                               'word vector model nor no training data'):
            GensimVectors(None, None, model='word2vec')
        with self.assertRaises(Exception, msg='Should only accept the following models'\
                               ' {}'.format(['word2vec', 'fasttext'])):
            GensimVectors(None, [['hello', 'how', 'are']], model='nothing',
                          min_count=1)

        # Test creating vectors from data
        data_path = os.path.abspath(read_config('sherlock_holmes_test', 
                                                CONFIG_FP))
        with open(data_path, 'r') as data:
            data = map(tokenisers.whitespace, data)
            with tempfile.NamedTemporaryFile() as temp_file:
                data_vector = GensimVectors(temp_file.name, data, model='word2vec',
                                            size=200, name='sherlock')
                d_vec_size = data_vector.vector_size
                self.assertEqual(d_vec_size, 200, msg='Vector size should be 200 not'\
                                 ' {}'.format(d_vec_size))
                sherlock_vec = data_vector.lookup_vector('sherlock')
                self.assertEqual(False, np.array_equal(zero_vector, sherlock_vec),
                                 msg='Sherlock should be a non-zero vector')
                # Test that it saved the trained model
                saved_vector = GensimVectors(temp_file.name, None, model='word2vec')
                s_vec_size = saved_vector.vector_size
                self.assertEqual(s_vec_size, 200, msg='Vector size should be 200 not'\
                                 ' {}'.format(s_vec_size))
                equal_sherlocks = np.array_equal(sherlock_vec,
                                                 saved_vector.lookup_vector('sherlock'))
                self.assertEqual(True, equal_sherlocks, msg='The saved model and '\
                                 'the trained model should have the same vectors')
                # Ensure the name attributes works
                self.assertEqual('sherlock', data_vector.name, msg='The name '\
                                 'of the instance should be sherlock and not {}'\
                                 .format(data_vector.name))
    def test_pre_trained(self):
        '''
        Tests the :py:class:`bella.word_vectors.PreTrained`
        '''

        # Test constructor
        with self.assertRaises(TypeError, msg='Should not accept a list when '\
                               'file path is expect to be a String'):
            PreTrained(['a fake file path'])
        with self.assertRaises(ValueError, msg='Should not accept strings that '\
                               'are not file paths'):
            PreTrained('file.txt')
        # Test if model loads correctly
        sswe_model = SSWE(skip_conf=True)
        sswe_vec_size = sswe_model.vector_size
        self.assertEqual(sswe_vec_size, 50, msg='Vector size should be 50 not '\
                         '{}'.format(sswe_vec_size))
        unknown_word = '$$$ZERO_TOKEN$$$'
        unknown_vector = sswe_model.lookup_vector(unknown_word)
        zero_vector = np.zeros(sswe_vec_size)

        # Test the unknown vector value
        self.assertEqual(False, np.array_equal(zero_vector, unknown_vector),
                         msg='The unknown vector should not be zeros')
        # Ensure the name attributes works
        self.assertEqual('sswe', sswe_model.name, msg='The name of the instance '\
                         'should be sswe and not {}'.format(sswe_model.name))
        # Test that the unknown word is mapped to the zero vector
        self.assertEqual(0, sswe_model.word2index[unknown_word], msg='unknown word '\
                         'should be mapped to the 0 index')
        # Test that the unknown index maps the unknown word
        self.assertEqual('<unk>', sswe_model.index2word[0], msg='unknown index '\
                         'should map to the <unk> word')
        # Test that the unknown word maps to the unknown vector
        is_zero = np.array_equal(sswe_model.index2vector[0], np.zeros(sswe_vec_size))
        self.assertEqual(False, is_zero, msg='The unknwon vector should not be '\
                         'zeros')

    @pytest.mark.skip(reason="Takes a long time to test only add on large tests")
    def test_glove_twitter_download(self):
        '''
        Tests that the Glove Twitter word vectors are downloaded correctly
        '''
        # Creating an instance of the vectors ensures they are downloaded
        glove_twitter_vectors = GloveTwitterVectors(25, skip_conf=True)

        current_dir = os.path.abspath(os.path.dirname(__file__))
        glove_twitter_folder = os.path.join(current_dir, os.pardir, 'data',
                                            'word_vectors', 'glove_twitter')
        self.assertEqual(True, os.path.isdir(glove_twitter_folder), msg='Glove '\
                         'Twitter vectors directory should be here {}'\
                         .format(glove_twitter_folder))

        glove_twitter_files = ['glove.twitter.27B.25d.binary',
                               'glove.twitter.27B.50d.binary',
                               'glove.twitter.27B.100d.binary',
                               'glove.twitter.27B.200d.binary']
        for glove_twitter_file in glove_twitter_files:
            glove_file_path = os.path.join(glove_twitter_folder, glove_twitter_file)
            self.assertEqual(True, os.path.isfile(glove_file_path), msg='Glove '\
                             'Twitter vector file {} should be here {}'\
                             .format(glove_twitter_file, glove_file_path))

    @pytest.mark.skip(reason="Takes a long time to test only add on large tests")
    def test_glove_common_download(self):
        '''
        Tests that the Glove Common Crawl 840 and 42 Billion token vectors are
        downloaded correctly
        '''

        for version in [42, 840]:
            GloveCommonCrawl(version=version, skip_conf=True)

            current_dir = os.path.abspath(os.path.dirname(__file__))

            glove_common_folder = 'glove_common_crawl_{}b'.format(version)
            glove_common_folder = os.path.join(current_dir, os.pardir, 'data',
                                               'word_vectors',
                                               glove_common_folder)
            self.assertEqual(True, os.path.isdir(glove_common_folder),
                             msg='Glove Common Crawl vector directory should '\
                             'be here {}'.format(glove_common_folder))
            glove_common_file_name = 'glove.{}B.300d.binary'.format(version)
            glove_common_file_path = os.path.join(glove_common_folder,
                                                  glove_common_file_name)
            self.assertEqual(True, os.path.isfile(glove_common_file_path),
                             msg='Glove common vector file {} should be here {}'\
                             .format(glove_common_file_name,
                                     glove_common_file_path))
