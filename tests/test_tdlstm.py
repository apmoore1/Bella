'''
Unit test suite for the :py:mod:`tdparse.helper` module.
'''
from unittest import TestCase

import numpy as np

from tdparse.helper import read_config
from tdparse.word_vectors import PreTrained
from tdparse.word_vectors import GensimVectors
from tdparse import lexicons
from tdparse.dependency_parsers import tweebo
from tdparse.tokenisers import whitespace
from tdparse.models.tdlstm import LSTM

class TestTDLstm(TestCase):
    '''
    Contains the following functions:
    '''

    def test_process_text(self):
        '''
        Test process_text
        '''
        sswe_path = read_config('sswe_files')['vo_zhang']
        sswe_model = PreTrained(sswe_path, name='sswe')

        test_texts = ['hello how are you', 'I am very good thank you nottttthingword',
                      'i am very good thank you']
        valid_return = np.asarray([[0, 0, 0, 394, 86, 45, 8],
                                   [0, 131, 199, 58, 88, 8, 0],
                                   [0, 5, 131, 199, 58, 88, 8]])
        lstm = LSTM(whitespace, sswe_model)
        seq_len, test_return = lstm.process_text(test_texts, -1)
        are_equal = np.array_equal(valid_return, test_return)
        self.assertEqual(True, are_equal, msg='valid sequences {} '\
                         'return {}'.format(valid_return, test_return))
        self.assertEqual(7, seq_len, msg='Max sequence length must = 7 not '\
                         '{}'.format(seq_len))

        # Testing pre-truncating
        valid_return = np.asarray([[86, 45, 8],
                                   [88, 8, 0],
                                   [58, 88, 8]])
        test_seq_len, test_return = lstm.process_text(test_texts, 3)
        are_equal = np.array_equal(valid_return, test_return)
        self.assertEqual(True, are_equal, msg='Cannot pre-truncate data valid {} '\
                         'test {}'.format(valid_return, test_return))
        self.assertEqual(test_seq_len, 3, msg='The sequence length should be 3 '\
                         'not {}'.format(test_seq_len))
        # Testing post-truncating
        valid_return = np.asarray([[0, 394, 86, 45, 8],
                                   [0, 131, 199, 58, 88],
                                   [5, 131, 199, 58, 88]])
        test_seq_len, test_return = lstm.process_text(test_texts, 5, truncate='post')
        are_equal = np.array_equal(valid_return, test_return)
        self.assertEqual(True, are_equal, msg='Cannot post-truncate data valid {} '\
                         'test {}'.format(valid_return, test_return))
        self.assertEqual(test_seq_len, 5, msg='The sequence length should be 5 '\
                         'not {}'.format(test_seq_len))

        # Testing post-padding, Pre padding already test by functions above
        valid_return = np.asarray([[394, 86, 45, 8, 0],
                                   [199, 58, 88, 8, 0],
                                   [131, 199, 58, 88, 8]])
        test_seq_len, test_return = lstm.process_text(test_texts, 5, padding='post')
        are_equal = np.array_equal(valid_return, test_return)
        self.assertEqual(True, are_equal, msg='Cannot post-pad data valid {} '\
                         'test {}'.format(valid_return, test_return))
        self.assertEqual(test_seq_len, 5, msg='The sequence length should be 5 '\
                         'not {}'.format(test_seq_len))
