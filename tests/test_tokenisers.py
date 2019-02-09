'''
Unit test suite for the :py:mod:`bella.tokenisers` module.
'''
from unittest import TestCase

from bella.tokenisers import whitespace
from bella.tokenisers import ark_twokenize, spacy_tokeniser

class TestTokenisers(TestCase):
    '''
    Contains the following functions:
    1. :py:func:`bella.tokenisers.whitespace`
    '''

    test_sentences = ['The fox    jumped over the         MOON.',
                      'lol ly x0x0,:D',
                      '    ']

    def test_whitespace(self):
        '''
        Tests :py:func:`bella.tokenisers.whitespace`
        '''

        with self.assertRaises(ValueError, msg='It should not accept a list'):
            whitespace(['words to be tested'])

        expected_results = [['The', 'fox', 'jumped', 'over', 'the', 'MOON.'],
                            ['lol', 'ly', 'x0x0,:D'],
                            []]
        for index, test_sentence in enumerate(self.test_sentences):
            test_result = whitespace(test_sentence)
            expected_result = expected_results[index]
            self.assertIsInstance(test_result, list, msg='The returned result is of '\
                                  'the wrong type {} should be a list'.format(type(test_result)))
            self.assertEqual(expected_result, test_result, msg='Did not return the '\
                             'expected result {} returned this {}'\
                             .format(expected_result, test_result))

    def test_ark_twokenize(self):
        '''
        Tests :py:func:`bella.tokenisers.ark_twokenize`
        '''

        with self.assertRaises(ValueError, msg='It should not accept a list'):
            ark_twokenize(['words to be tested'])

        expected_results = [['The', 'fox', 'jumped', 'over', 'the', 'MOON', '.'],
                            ['lol', 'ly', 'x0x0', ',', ':D'],
                            []]
        for index, test_sentence in enumerate(self.test_sentences):
            test_result = ark_twokenize(test_sentence)
            expected_result = expected_results[index]
            self.assertIsInstance(test_result, list, msg='The returned result is of '\
                                  'the wrong type {} should be a list'.format(type(test_result)))
            self.assertEqual(expected_result, test_result, msg='Did not return the '\
                             'expected result {} returned this {}'\
                             .format(expected_result, test_result))

    def test_spacy_tokeniser(self):
        '''
        Tests :py:func:`bella.tokenisers.spacy_tokeniser`
        '''

        with self.assertRaises(ValueError, msg='It should not accept a list'):
            spacy_tokeniser(['words to be tested'])

        expected_results =[['The', 'fox', 'jumped', 'over', 'the', 'MOON', '.'], 
                           ['lol', 'ly', 'x0x0,:D'], []]
        for index, test_sentence in enumerate(self.test_sentences):
            test_result = spacy_tokeniser(test_sentence)
            expected_result = expected_results[index]
            self.assertIsInstance(test_result, list, msg='The returned result is of '\
                                  'the wrong type {} should be a list'.format(type(test_result)))
            self.assertEqual(expected_result, test_result, msg='Did not return the '\
                             'expected result {} returned this {}'\
                             .format(expected_result, test_result))