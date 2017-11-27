''''
Unit test suite for the :py:mod:`tdparse.dependency_tokens` module.
'''
from collections import defaultdict
from unittest import TestCase

from tdparse.dependency_tokens import DependencyToken

class TestDependencyTokens(TestCase):
    '''
    Contains the following functions:
    '''

    def test_init(self):
        '''
        Tests the DependencyToken constructor.
        '''

        def dict_test(valid_dict, test_dict):
            '''
            Given a valid dictionary and a test dictionary, tests that keys in
            the valid_dict are in the test_dict and all values are equal.

            :param valid_dict: A dictionary of valid keys and values
            :param test_dict: A dictionary to be tested to ensure they have the \
            same keys and values as the valid_dict parameter
            :type valid_dict: dict
            :type test_dict: dict
            :returns: Nothing but performs a series of assertion tests
            :rtype: None
            '''

            for key, value in valid_dict.items():
                has_key = key in test_dict
                self.assertEqual(True, has_key, msg='The relations dict should '\
                                 'have the following key {}'.format(key))
                test_value = test_dict[key]
                self.assertEqual(value, test_value, msg='The value for the key {}'\
                                 ' should have the following value {} and not {}'\
                                 .format(key, value, test_value))
        # Basic valid test
        relations = {1 : ['something', 'anything'], 2 : ['anything']}
        token = 'another'
        connected_words = ['something', 'anything']
        test_token = DependencyToken(token, relations, connected_words)
        self.assertEqual(test_token.token, token, msg='The token attribute should '\
                         'be `another` and not {}'.format(test_token.token))
        dict_test(relations, test_token.relations)

        with self.assertRaises(TypeError, msg='token parameter has to always be '\
                               'of type String'):
            DependencyToken(('not a string',), relations, connected_words)
        with self.assertRaises(TypeError, msg='relations parameter has to always '\
                               'be of type dict'):
            DependencyToken(token, ('not a dict'), connected_words)
        with self.assertRaises(TypeError, msg='The keys in the relations parameter'\
                               ' has to be always of type int'):
            DependencyToken(token, {0.1 : ['something']}, connected_words)
        with self.assertRaises(TypeError, msg='The values in the relations '\
                               'parameter has to be of type list'):
            DependencyToken(token, {1 : {'something'}}, connected_words)
        with self.assertRaises(ValueError, msg='The keys in relations parameter '\
                               'lowest value has to be 1'):
            DependencyToken(token, {2 : ['something']}, connected_words)
        with self.assertRaises(ValueError, msg='The keys in relations parameter '\
                               'lowest value has to be 1'):
            DependencyToken(token, {-2 : ['something'], 1 : ['something']}, connected_words)
        with self.assertRaises(ValueError, msg='The keys in relations parameter '\
                               'lowest value has to be 1'):
            DependencyToken(token, {0 : ['something'], 1 : ['something'],
                                    1000 : ['something']}, connected_words)
        with self.assertRaises(ValueError, msg='The keys in relations parameter '\
                               'has to have incremental range of 1'):
            DependencyToken(token, {1 : ['something'], 2 : ['another'],
                                    4 : ['invalid']}, connected_words)
        with self.assertRaises(ValueError, msg='The keys in relations parameter '\
                               'has to have incremental range of 1'):
            DependencyToken(token, {1 : ['something'], 2 : ['another'],
                                    4 : ['invalid'], 5 : ['wrong']}, connected_words)
        with self.assertRaises(TypeError, msg='connected words has to be of type list'):
            DependencyToken(token, relations, set(['anything']))
    def test_get_n_relations(self):
        '''
        Tests DependencyToken().get_n_relations function
        '''

        connected_words = ['test']
        relations = {1 : ['test', 'anything'], 2 : ['something', 'another'],
                     3 : ['you', 'the'], 4 : ['last', 'pass']}
        token = 'nothing'

        # Check that it works when the token has no dependencies
        dep_token = DependencyToken(token, defaultdict(list), connected_words)
        # Check normal situation
        dep_token = DependencyToken(token, relations, connected_words)

        with self.assertRaises(ValueError, msg='Should not accept negative values'\
                               ' where the first value > second value'):
            dep_token.get_n_relations((-2, -3))
        with self.assertRaises(ValueError, msg='Should not accept zero values'):
            dep_token.get_n_relations((0, 2))
        with self.assertRaises(ValueError, msg='Should not accept zero values'):
            dep_token.get_n_relations((0, 0))
        with self.assertRaises(TypeError, msg='Should only accept tuples'):
            dep_token.get_n_relations((1))
        with self.assertRaises(TypeError, msg='Should only accept tuples'):
            dep_token.get_n_relations(range(1, 2))
        with self.assertRaises(ValueError, msg='The tuple has to of length 2'):
            dep_token.get_n_relations((1, 3, 1))
        with self.assertRaises(ValueError, msg='The first value has to be less '\
                               'than the second value'):
            dep_token.get_n_relations((3, 1))
        with self.assertRaises(ValueError, msg='The values in the tuple have to '\
                               'be of type int'):
            dep_token.get_n_relations((0.5, 3))
        with self.assertRaises(ValueError, msg='The values in the tuple have to '\
                               'be of type int'):
            dep_token.get_n_relations((1, 3.5))

        valid_1 = ['test', 'anything']
        test_1 = dep_token.get_n_relations((1, 1))
        self.assertEqual(valid_1, test_1, msg='Should return the first depth of '\
                         'realtions {} and not {}'.format(valid_1, test_1))
        valid_2 = ['something', 'another']
        test_2 = dep_token.get_n_relations((2, 2))
        self.assertEqual(valid_2, test_2, msg='Should return the second depth of '\
                         'realtions {} and not {}'.format(valid_2, test_2))
        valid_1_2 = ['test', 'anything', 'something', 'another']
        test_1_2 = dep_token.get_n_relations((1, 2))
        self.assertEqual(valid_1_2, test_1_2, msg='Should return the first and '\
                         'second depth of relations {} and not {}'\
                         .format(valid_1_2, test_1_2))
        valid_2_4 = ['something', 'another', 'you', 'the', 'last', 'pass']
        test_2_4 = dep_token.get_n_relations((2, 4))
        self.assertEqual(valid_2_4, test_2_4, msg='Should return the second, third'\
                         ' and fourth relations {} and not {}'\
                         .format(valid_2_4, test_2_4))
        valid_2_7 = ['something', 'another', 'you', 'the', 'last', 'pass']
        test_2_7 = dep_token.get_n_relations((2, 7))
        self.assertEqual(valid_2_7, test_2_7, msg='Should return the second, third'\
                         ' and fourth relations and no more as the relations dict'\
                         ' has been exhausted valid: {} and not {}'\
                         .format(valid_2_7, test_2_7))

        valid_neg_1 = ['last', 'pass']
        test_neg_1 = dep_token.get_n_relations((-1, -1))
        self.assertEqual(valid_neg_1, test_neg_1, msg='Should return the last'\
                         'realtions {} and not {}'.format(valid_neg_1, test_neg_1))
        valid_neg_2 = ['you', 'the', 'last', 'pass']
        test_neg_2 = dep_token.get_n_relations((-2, -1))
        self.assertEqual(valid_neg_2, test_neg_2, msg='Should return the last 2'\
                         'realtions {} and not {}'.format(valid_neg_2, test_neg_2))
        valid_neg_3 = ['something', 'another', 'you', 'the', 'last', 'pass']
        test_neg_3 = dep_token.get_n_relations((-3, -1))
        self.assertEqual(valid_neg_3, test_neg_3, msg='Should return the last 3'\
                         'realtions {} and not {}'.format(valid_neg_3, test_neg_3))
        valid_neg_4 = ['test', 'anything', 'something', 'another', 'you', 'the']
        test_neg_4 = dep_token.get_n_relations((-4, -2))
        self.assertEqual(valid_neg_4, test_neg_4, msg='Should return the first 3'\
                         'realtions {} and not {}'.format(valid_neg_4, test_neg_4))

        valid_comp_1 = ['something', 'another', 'you', 'the', 'last', 'pass']
        test_comp_1 = dep_token.get_n_relations((2, -1))
        self.assertEqual(valid_comp_1, test_comp_1, msg='Should return from the '\
                         'second to the last realtions {} and not {}'\
                         .format(valid_comp_1, test_comp_1))
        valid_comp_2 = ['test', 'anything', 'something', 'another', 'you',
                        'the', 'last', 'pass']
        test_comp_2 = dep_token.get_n_relations((1, -1))
        self.assertEqual(valid_comp_2, test_comp_2, msg='Should return from the '\
                         'first to the last realtions {} and not {}'\
                         .format(valid_comp_2, test_comp_2))
        valid_comp_3 = ['something', 'another', 'you', 'the']
        test_comp_3 = dep_token.get_n_relations((2, -2))
        self.assertEqual(valid_comp_3, test_comp_3, msg='Should return from the '\
                         'second to the second to last realtions {} and not {}'\
                         .format(valid_comp_3, test_comp_3))
