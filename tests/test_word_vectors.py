'''
Unit test suite for the :py:mod:`tdparse.word_vectors` module.
'''
import os
from unittest import TestCase
import tempfile

import numpy as np

from tdparse.helper import read_config
import tdparse.tokenisers as tokenisers
from tdparse.word_vectors import GensimVectors

class TestWordVectors(TestCase):
    '''
    Contains the following functions:

    1. test_gensim_word2vec
    '''

    def test_gensim_word2vec(self):
        '''
        Tests the :py:class:`tdparse.word_vectors.GensimVectors`
        '''

        # Test loading word vectors from a file
        vo_zhang_path = os.path.abspath(read_config('word2vec_files')['vo_zhang'])
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
