'''
Unit test suite for the :py:mod:`tdparse.word_vectors` module.
'''
import os
from unittest import TestCase
import tempfile

import numpy as np

from tdparse.helper import read_config
import tdparse.tokenisers as tokenisers
from tdparse.word_vectors import WordVectors
from tdparse.word_vectors import GensimVectors
from tdparse.word_vectors import PreTrained

class TestWordVectors(TestCase):
    '''
    Contains the following functions:

    1. test_wordvector_methods
    2. test_gensim_word2vec
    3. test_pre_trained
    '''

    def test_wordvector_methods(self):
        '''
        Tests the :py:class:`tdparse.word_vectors.WordVectors`
        '''
        # Testing the constructor
        with self.assertRaises(TypeError, msg='Should only accept dicts not lists '\
                               'for word2vector parameter'):
            WordVectors([1, 2, 3], [0])
        with self.assertRaises(TypeError, msg='Should only accept lists'):
            WordVectors({'something' : np.asarray([5, 6])}, 'something')

        hello_vec = np.asarray([0.5, 0.3, 0.4])
        another_vec = np.asarray([0.3333, 0.2222, 0.1111])
        test_vectors = {'hello' : hello_vec,
                        'another' : another_vec}
        word_vector = WordVectors(test_vectors, list(test_vectors.keys()))

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

        another_index = word2index['another']
        self.assertEqual('another', index2word[another_index], msg='index2word '\
                         'and word2index do not match on word `another`')
        index_correct = np.array_equal(index2vector[another_index], another_vec)
        self.assertEqual(True, index_correct, msg='index2vector does not return '\
                         'the correct vector for `another`')


    def test_gensim_word2vec(self):
        '''
        Tests the :py:class:`tdparse.word_vectors.GensimVectors`
        '''

        # Test loading word vectors from a file
        vo_zhang_path = read_config('word2vec_files')['vo_zhang']
        vo_zhang = GensimVectors(vo_zhang_path, None, model='word2vec')

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
        data_path = os.path.abspath(read_config('test_data')['sherlock_holmes'])
        with open(data_path, 'r') as data:
            data = map(tokenisers.whitespace, data)
            with tempfile.NamedTemporaryFile() as temp_file:
                data_vector = GensimVectors(temp_file.name, data, model='word2vec',
                                            size=200)
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
    def test_pre_trained(self):
        '''
        Tests the :py:class:`tdparse.word_vectors.PreTrained`
        '''

        # Test constructor
        with self.assertRaises(TypeError, msg='Should not accept a list when '\
                               'file path is expect to be a String'):
            PreTrained(['a fake file path'])
        with self.assertRaises(ValueError, msg='Should not accept strings that '\
                               'are not file paths'):
            PreTrained('file.txt')
        # Test if model loads correctly
        sswe_path = read_config('sswe_files')['vo_zhang']
        sswe_model = PreTrained(sswe_path)
        sswe_vec_size = sswe_model.vector_size
        self.assertEqual(sswe_vec_size, 50, msg='Vector size should be 50 not '\
                         '{}'.format(sswe_vec_size))
        unknown_word = '$$$ZERO_TOKEN$$$'
        unknown_vector = sswe_model.lookup_vector(unknown_word)
        zero_vector = np.zeros(sswe_vec_size)

        # Test the unknown vector value
        self.assertEqual(False, np.array_equal(zero_vector, unknown_vector),
                         msg='The unknown vector should not be zeros')
