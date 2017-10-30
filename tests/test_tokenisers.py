'''
Unit test suite for the :py:mod:`tdparse.tokenisers` module.
'''
from unittest import TestCase

from tdparse.tokenisers import whitespace

class TestTokenisers(TestCase):
    '''
    Contains the following functions:
    1. :py:func:`tdparse.tokenisers.whitespace`
    '''

    def test_whitespace(self):
        '''
        Tests :py:func:`tdparse.tokenisers.whitespace`
        '''
        with self.assertRaises(ValueError, msg='It should not accept a list'):
            whitespace(['words to be tested'])

        expected_result = ['The', 'fox', 'jumped', 'over', 'the', 'MOON.']
        test_result = whitespace('The fox    jumped over the         MOON.')
        self.assertIsInstance(test_result, list, msg='The returned result is of '\
                              'the wrong type {} should be a list'.format(type(test_result)))
        self.assertEqual(expected_result, test_result, msg='Did not return the '\
                         'expected result {} returned this {}'\
                         .format(expected_result, test_result))
