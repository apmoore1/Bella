'''
Unit test suite for the :py:mod:`tdparse.dependency_parsers` module.
'''
import os
import shutil
from unittest import TestCase

import pytest

from tdparse.dependency_tokens import DependencyToken
from tdparse.dependency_parsers import tweebo_install
from tdparse.dependency_parsers import tweebo

class TestDependencyParsers(TestCase):
    '''
    Contains the following functions:
    '''
    @pytest.mark.skip(reason="Takes a long time to test only add on large tests")
    def test_tweebo_install(self):
        '''
        Test for :py:func:`tdparse.dependency_parsers.tweebo_install`. Tests that
        it downloads the files correctly. Hard to test the functionality without
        testing the parser which is in the
        :py:func:`tdparse.tests.test_dependency_parsers.test_tweebo`
        '''

        def add(num1, num2):
            '''
            Adds two numbers together, used as a test function.
            '''
            return num1 + num2
        current_dir = os.path.abspath(os.path.dirname(__file__))
        tweebo_dir = os.path.abspath(os.path.join(current_dir, os.pardir, 'tools',
                                                  'TweeboParser'))
        model_dir = os.path.join(tweebo_dir, 'pretrained_models')
        model_file = os.path.join(tweebo_dir, 'pretrained_models.tar.gz')
        if os.path.isfile(model_file):
            os.remove(model_file)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        install_and_add = tweebo_install(add)
        self.assertEqual(True, os.path.isfile(model_file), msg='Did not download '\
                         'the tweebo models, file path {} Tweebo install error '\
                         'check the install.sh within tools/TweeboParser '\
                         .format(model_file))
        self.assertEqual(True, os.path.isdir(model_dir), msg='Did not unpack the '\
                         'tweebo models to the following dir {} Tweebo install error'\
                         ' check the install.sh within tools/TweeboParser '\
                         .format(model_dir))
        self.assertEqual(10, install_and_add(8, 2), msg='The tweebo install '\
                         'function does not wrap function properly')
    #@pytest.mark.skip(reason="Takes a long time to test only add on large tests")
    def test_tweebo(self):
        '''
        Tests tweebo function
        '''
        def check_dependencies(valid, test):
            '''
            Given two dictionaries where keys are relation depth and values are
            a list of words associated to that dependency depth. Creates tests
            to ensure test parameter is the same as valid if not fails tests.

            :param valid: The valid dependency dictionary
            :param test: The dependency dictionary to be tested
            :type valid: dict
            :type test: dict
            :returns: Nothing but performs a series of assertion tests
            :rtype: None
            '''

            for depth, words in valid.items():
                has_key = depth in test
                self.assertEqual(True, has_key, msg='The dependency dictionary '\
                                 'should contain the following key {}'\
                                 .format(depth))
                test_words = sorted(test[depth])
                valid_words = sorted(words)
                self.assertEqual(valid_words, test_words, msg='The dependecy '\
                                 'words at depth {} should be {} but are {}'\
                                 .format(depth, valid_words, test_words))
        def check_parser(parse_results, parser, correct_tokens, correct_deps,
                         correct_con_words):
            '''
            Checks if the dependency parser is performing correctly based on
            token values and dependecy connections.

            :param parse_results: List of dependecy tokens that are the output of \
            a dependecy parser that is being tested.
            :param parser: The name of the parser being tested.
            :param correct_tokens: The correct token values for the dependecy tokens
            :param correct_deps: Correct dependecy links between the dependecy tokens
            :param correct_con_words: Correct connected words for the dependecy tokens
            :type parse_results: list
            :type parser: String
            :type correct_tokens: list
            :type correct_deps: list
            :type correct_con_words: list
            :returns: Performs a series of assertion tests but returns None.
            :rtype: None
            '''

            # Goes through the results and compares them to the correct results above.
            if len(parse_results) != len(correct_tokens):
                raise ValueError('There should be as many results as there are '\
                                 'token lists output length error: correct {} '\
                                 'actual {}'.format(len(parse_results), len(correct_tokens)))
            for index, result in enumerate(parse_results):
                self.assertIsInstance(result, list, msg='The return of parser '\
                                      'function {} should be a list of lists '\
                                      'not list of {}'.format(parser, type(result)))
                rel_correct_tokens = correct_tokens[index]
                rel_correct_deps = correct_deps[index]
                rel_correct_con = correct_con_words[index]
                for token_index, dep_token in enumerate(result):

                    self.assertIsInstance(dep_token, DependencyToken, msg='The result '\
                                          'list should contain `DependencyToken` '\
                                          'instances and not {}'.format(type(dep_token)))
                    # Check that it tokenises correctly
                    valid_token = rel_correct_tokens[token_index]
                    test_token = dep_token.token
                    self.assertEqual(valid_token, test_token, msg='The token '\
                                     'value should be {} not {}'\
                                     .format(valid_token, test_token))
                    # Check that the dependencies are correct
                    valid_dep = rel_correct_deps[token_index]
                    test_dep = dep_token.relations
                    check_dependencies(valid_dep, test_dep)
                    # Check the connected words
                    valid_con_words = rel_correct_con[token_index]
                    test_con_words = dep_token.connected_words
                    self.assertEqual(valid_con_words, test_con_words,
                                     msg='Connected words are not correct valid {}'\
                                     ' test {} token {}'\
                                     .format(valid_con_words, test_con_words, test_token))


        test_sentences = ['To appear (EMNLP 2014): Detecting Non-compositional '\
                          'MWE Components using Wiktionary '\
                          'http://people.eng.unimelb.edu.au/tbaldwin/pubs/emn'\
                          'lp2014-mwe.pdf … #nlproc',
                          'Norm ish lookin sentence I think :)']

        test_tokens_1 = ['To', 'appear', '(', 'EMNLP', '2014', '):', 'Detecting',
                         'Non-compositional', 'MWE', 'Components', 'using',
                         'Wiktionary',
                         'http://people.eng.unimelb.edu.au/tbaldwin/pubs/emnlp2014-mwe.pdf',
                         '…', '#nlproc']
        test_tokens_2 = ['Norm', 'ish', 'lookin', 'sentence', 'I', 'think', ':)']
        test_tokens = [test_tokens_1, test_tokens_2]

        test_dep_1 = [{1 : ['appear']}, {}, {}, {1 : ['2014']}, {}, {}, {}, {}, {},
                      {1 : ['MWE', 'Non-compositional']},
                      {1 : ['Components', 'Wiktionary', 'Detecting'],
                       2 : ['MWE', 'Non-compositional']}, {}, {}, {}, {}]
        test_dep_2 = [{}, {1 : ['Norm']}, {1 : ['ish', 'sentence'], 2 : ['Norm']},
                      {}, {}, {1 : 'I'}, {}]
        test_deps = [test_dep_1, test_dep_2]

        test_connected_words_1 = [[('To', 'CURRENT'), ('appear', 'CONNECTED')],
                                  [('To', 'CONNECTED'), ('appear', 'CURRENT')],
                                  [('(', 'CURRENT')],
                                  [('EMNLP', 'CURRENT'), ('2014', 'CONNECTED')],
                                  [('EMNLP', 'CONNECTED'), ('2014', 'CURRENT')],
                                  [('):', 'CURRENT')],
                                  [('Detecting', 'CURRENT'), ('Non-compositional', 'CONNECTED'),
                                   ('MWE', 'CONNECTED'), ('Components', 'CONNECTED'),
                                   ('using', 'CONNECTED'), ('Wiktionary', 'CONNECTED')],
                                  [('Detecting', 'CONNECTED'), ('Non-compositional', 'CURRENT'),
                                   ('MWE', 'CONNECTED'), ('Components', 'CONNECTED'),
                                   ('using', 'CONNECTED'), ('Wiktionary', 'CONNECTED')],
                                  [('Detecting', 'CONNECTED'), ('Non-compositional', 'CONNECTED'),
                                   ('MWE', 'CURRENT'), ('Components', 'CONNECTED'),
                                   ('using', 'CONNECTED'), ('Wiktionary', 'CONNECTED')],
                                  [('Detecting', 'CONNECTED'), ('Non-compositional', 'CONNECTED'),
                                   ('MWE', 'CONNECTED'), ('Components', 'CURRENT'),
                                   ('using', 'CONNECTED'), ('Wiktionary', 'CONNECTED')],
                                  [('Detecting', 'CONNECTED'), ('Non-compositional', 'CONNECTED'),
                                   ('MWE', 'CONNECTED'), ('Components', 'CONNECTED'),
                                   ('using', 'CURRENT'), ('Wiktionary', 'CONNECTED')],
                                  [('Detecting', 'CONNECTED'), ('Non-compositional', 'CONNECTED'),
                                   ('MWE', 'CONNECTED'), ('Components', 'CONNECTED'),
                                   ('using', 'CONNECTED'), ('Wiktionary', 'CURRENT')],
                                  [('http://people.eng.unimelb.edu.au/tbaldwin/'
                                    'pubs/emnlp2014-mwe.pdf', 'CURRENT')],
                                  [('…', 'CURRENT')], [('#nlproc', 'CURRENT')]]

        test_connected_words_2 = [[('Norm', 'CURRENT'), ('ish', 'CONNECTED'),
                                   ('lookin', 'CONNECTED'), ('sentence', 'CONNECTED')],
                                  [('Norm', 'CONNECTED'), ('ish', 'CURRENT'),
                                   ('lookin', 'CONNECTED'), ('sentence', 'CONNECTED')],
                                  [('Norm', 'CONNECTED'), ('ish', 'CONNECTED'),
                                   ('lookin', 'CURRENT'), ('sentence', 'CONNECTED')],
                                  [('Norm', 'CONNECTED'), ('ish', 'CONNECTED'),
                                   ('lookin', 'CONNECTED'), ('sentence', 'CURRENT')],
                                  [('I', 'CURRENT'), ('think', 'CONNECTED')],
                                  [('I', 'CONNECTED'), ('think', 'CURRENT')],
                                  [(':)', 'CURRENT')]]
        test_connected_words = [test_connected_words_1, test_connected_words_2]

        results = tweebo(test_sentences)
        self.assertIsInstance(results, list, msg='The return of `tweebo` function '\
                              'should be of type list not {}'.format(type(results)))
        # Goes through the results and compares them to the correct results above.
        check_parser(results, 'twebbo', test_tokens, test_deps, test_connected_words)
        #
        # Testing when you can include an empty String
        #
        test_sentences = ['To appear (EMNLP 2014): Detecting Non-compositional '\
                          'MWE Components using Wiktionary '\
                          'http://people.eng.unimelb.edu.au/tbaldwin/pubs/emn'\
                          'lp2014-mwe.pdf … #nlproc',
                          '',
                          'Norm ish lookin sentence I think :)']

        test_tokens_1 = ['To', 'appear', '(', 'EMNLP', '2014', '):', 'Detecting',
                         'Non-compositional', 'MWE', 'Components', 'using',
                         'Wiktionary',
                         'http://people.eng.unimelb.edu.au/tbaldwin/pubs/emnlp2014-mwe.pdf',
                         '…', '#nlproc']
        test_tokens_2 = ['Norm', 'ish', 'lookin', 'sentence', 'I', 'think', ':)']
        test_tokens = [test_tokens_1, [{}], test_tokens_2]

        test_dep_1 = [{1 : ['appear']}, {}, {}, {1 : ['2014']}, {}, {}, {}, {}, {},
                      {1 : ['MWE', 'Non-compositional']},
                      {1 : ['Components', 'Wiktionary', 'Detecting'],
                       2 : ['MWE', 'Non-compositional']}, {}, {}, {}, {}]
        test_dep_2 = [{}, {1 : ['Norm']}, {1 : ['ish', 'sentence'], 2 : ['Norm']},
                      {}, {}, {1 : 'I'}, {}]
        test_deps = [test_dep_1, [], test_dep_2]
        test_connected_words = [test_connected_words_1, [], test_connected_words_2]

        results = tweebo(test_sentences)
        self.assertIsInstance(results, list, msg='The return of `tweebo` function '\
                              'should be of type list not {}'.format(type(results)))
        check_parser(results, 'twebbo', test_tokens, test_deps, test_connected_words)
