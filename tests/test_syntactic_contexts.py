'''
Unit test suite for the :py:mod:`tdparse.syntactic_contexts` module.
'''
from unittest import TestCase

from tdparse.data_types import Target
from tdparse.syntactic_contexts import context
from tdparse.syntactic_contexts import target_normalisation
from tdparse.syntactic_contexts import dependency_context
from tdparse.syntactic_contexts import dependency_relation_context
from tdparse.dependency_parsers import tweebo

class TestTarget(TestCase):
    '''
    Contains the following functions:
    '''

    def test_target_normlisation(self):
        '''
        Tests target_normalisation
        '''

        with self.assertRaises(TypeError, msg='target_dict parameter has to '\
                               'be of type dict only'):
            target_normalisation(['anything'])

        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful ssNews Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (85, 97)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(16, 19), (64, 67)]}]
        valid_results = [('This is a fake news_article dd that is to represent a '\
                          'Tweet!!!! and it was an awful ss news_article ss I think.',
                          'news_article', [], 2),
                         ('I had a great ss day however I did not get much '\
                          'work done in the day s',
                          'day', [0, 1], 1)]
        test_values = [Target(**test_value) for test_value in test_values]
        for index, test_value in enumerate(test_values):
            test_result = target_normalisation(test_value)
            valid_result = valid_results[index]
            self.assertEqual(valid_result, test_result, msg='Results is '\
                             '{} and should be {}'\
                             .format(valid_result, test_result))
    def test_dependency_relation_context(self):
        '''
        Tests dependency_relation_context
        '''

        # Test the normalise case
        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful ssNews Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (85, 97)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(16, 19), (64, 67)]},
                       {'target_id':str(2),
                        'sentiment':1,
                        'text':'Ten pop star heartthrobe is all the rage on '\
                        'social media',
                        'target':'is',
                        'spans':[(25, 27)]},
                       {'target_id':str(3),
                        'sentiment':1,
                        'text':'Ten pop star heartthrobe is all the rage on '\
                        'social media',
                        'target':'ten',
                        'spans':[(0, 3)]}]
        valid_results = [['a fake', 'an awful'], ['a great', 'the'],
                         ['heartthrobe all Ten pop star rage on the media '\
                          'social'], ['']]
        test_values = [Target(**test_value) for test_value in test_values]
        test_results = dependency_relation_context(test_values, tweebo,
                                                   n_relations=(1, -1))
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            self.assertEqual(valid_result, test_result, msg='Incorrect context'\
                             ' correct {} test {}'.format(valid_result, test_result))

        # Testing when we only want the first dependency relation
        valid_results = [['a fake', 'an awful'], ['a great', 'the'],
                         ['heartthrobe all'], ['']]
        test_results = dependency_relation_context(test_values, tweebo)
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            self.assertEqual(valid_result, test_result, msg='Incorrect context'\
                             ' correct {} test {}'.format(valid_result, test_result))

        # Testing to ensure it will lower case the words before processing
        valid_results = [['a fake', 'an awful'], ['a great', 'the'],
                         ['heartthrobe all ten pop star rage on the media '\
                          'social'], ['']]
        test_results = dependency_relation_context(test_values, tweebo, True,
                                                   (1, -1))
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            self.assertEqual(valid_result, test_result, msg='Incorrect context'\
                             ' correct {} test {}'.format(valid_result, test_result))

        # Testing for when a sentence mentions the target more than onece but we 
        # are only interested in the first mention
        test_values = [{'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the day',
                        'target':'day',
                        'spans':[(16, 19)]}]
        valid_results = [['a great']]
        test_values = [Target(**test_value) for test_value in test_values]
        test_results = dependency_relation_context(test_values, tweebo,
                                                   n_relations=(1, -1))
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            self.assertEqual(valid_result, test_result, msg='Incorrect context'\
                             ' for more than one mention correct {} test {}'\
                             .format(valid_result, test_result))

    def test_dependency_context(self):
        '''
        Tests dependency_context
        '''

        # Test the normalise case
        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful ssNews Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (85, 97)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(16, 19), (64, 67)]}]
        valid_results = [[{'text' : 'This is a fake news_article',
                           'span' : [15, 27]},
                          {'text' : 'dd that is to represent a Tweet and it was '\
                                    'an awful news_article',
                           'span' : [52, 64]}],
                         [{'text' : 'I had a great day however I did not get '\
                                    'much work done in the day',
                           'span' : [14, 17]},
                          {'text' : 'I had a great day however I did not get '\
                                    'much work done in the day',
                           'span' : [62, 65]}]]
        test_values = [Target(**test_value) for test_value in test_values]
        test_results = dependency_context(test_values, tweebo)
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            for dict_index, valid_dict in enumerate(valid_result):
                test_dict = test_result[dict_index]
                self.assertEqual(valid_dict['text'], test_dict['text'],
                                 msg='texts are different correct `{}` test `{}`'\
                                     .format(valid_dict['text'], test_dict['text']))
                self.assertEqual(valid_dict['span'], test_dict['span'],
                                 msg='spans are different correct `{}` test `{}`'\
                                     .format(valid_dict['span'], test_dict['span']))
        # Test the lower casing case
        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful ssNews Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (85, 97)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(16, 19), (64, 67)]}]
        valid_results = [[{'text' : 'this is a fake news_article',
                           'span' : [15, 27]},
                          {'text' : 'dd that is to represent a tweet and it was '\
                                    'an awful news_article',
                           'span' : [52, 64]}],
                         [{'text' : 'i had a great day however i did not get '\
                                    'much work done in the day',
                           'span' : [14, 17]},
                          {'text' : 'i had a great day however i did not get '\
                                    'much work done in the day',
                           'span' : [62, 65]}]]
        test_values = [Target(**test_value) for test_value in test_values]
        test_results = dependency_context(test_values, tweebo, lower=True)
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            for dict_index, valid_dict in enumerate(valid_result):
                test_dict = test_result[dict_index]
                self.assertEqual(valid_dict['text'], test_dict['text'],
                                 msg='texts are different correct `{}` test `{}`'\
                                     .format(valid_dict['text'], test_dict['text']))
                self.assertEqual(valid_dict['span'], test_dict['span'],
                                 msg='spans are different correct `{}` test `{}`'\
                                     .format(valid_dict['span'], test_dict['span']))
        # Test the case where the target is mentioned twice but is only relevant
        # to one of the mentions
        test_values = [{'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the day',
                        'target':'day',
                        'spans':[(16, 19)]},
                        {'spans': [(64, 97)], 
                         'target_id': '6541', 
                         'target': 'Core Processing Unit temperatures', 
                         'text': 'Temperatures on the outside were alright but i did '\
                                 'not track in Core Processing Unit temperatures.', 
                         'sentiment': 0}]
        valid_results = [[{'text' : 'I had a great day however I did not get '\
                                    'much work done in the day',
                           'span' : [14, 17]}]]
        test_values = [Target(**test_value) for test_value in test_values]
        test_results = dependency_context(test_values, tweebo)
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            for dict_index, valid_dict in enumerate(valid_result):
                test_dict = test_result[dict_index]
                self.assertEqual(valid_dict['text'], test_dict['text'],
                                 msg='texts are different correct `{}` test `{}`'\
                                     .format(valid_dict['text'], test_dict['text']))
                self.assertEqual(valid_dict['span'], test_dict['span'],
                                 msg='spans are different correct `{}` test `{}`'\
                                     .format(valid_dict['span'], test_dict['span']))

        Target()
    def test_context(self):
        '''
        Tests context
        '''

        def test_contexts(all_valid_results, all_test_results):
            '''
            :param all_valid_results: A list of a list of Strings that are the \
            correct values.
            :param all_test_results: A list of a list of Strings that are the \
            values being tested.
            :type all_valid_results: list
            :type all_test_results: list
            :returns: Nothing. It tests if the Strings found it the test results \
            are the same as those in the valid results else raises an assertion \
            error.
            :rtype: None
            '''

            for index, valid_results in enumerate(all_valid_results):
                for inner_index, valid_result in enumerate(valid_results):
                    test_result = all_test_results[index][inner_index]
                    self.assertEqual(valid_result, test_result, msg='context '\
                                     'should be {} and not {}'\
                                     .format(valid_result, test_result))

        test_values = [[{'text' : 'This is a fake news_article',
                         'span' : [15, 27]},
                        {'text' : 'dd that is to represent a Tweet and it was '\
                                  'an awful news_article',
                         'span' : [52, 64]}],
                       [{'text' : 'I had a great day however I did not get '\
                                  'much work done in the day',
                         'span' : [14, 17]},
                        {'text' : 'I had a great day however I did not get '\
                                  'much work done in the day',
                         'span' : [62, 65]}]]
        # Testing all four contexts: 1. left, 2. right, 3. target and 4. full
        valid_left_results = [['This is a fake ',
                               'dd that is to represent a Tweet and it was an '\
                               'awful '],
                              ['I had a great ', 'I had a great day however I '\
                               'did not get much work done in the ']]
        test_left_results = context(test_values, 'left')
        test_contexts(valid_left_results, test_left_results)

        valid_right_results = [['', ''],
                               [' however I did not get much work done in the day',
                                '']]
        test_right_results = context(test_values, 'right')
        test_contexts(valid_right_results, test_right_results)

        valid_target_results = [['news_article', 'news_article'], ['day', 'day']]
        test_target_results = context(test_values, 'target')
        test_contexts(valid_target_results, test_target_results)

        valid_full_results = [['This is a fake news_article',
                               'dd that is to represent a Tweet and it was '\
                               'an awful news_article'],
                              ['I had a great day however I did not get much '\
                               'work done in the day',
                               'I had a great day however I did not get much '\
                                'work done in the day']]
        test_full_results = context(test_values, 'full')
        test_contexts(valid_full_results, test_full_results)

        # Testing all contexts but including the target
        valid_left_results = [['This is a fake news_article',
                               'dd that is to represent a Tweet and it was an '\
                               'awful news_article'],
                              ['I had a great day', 'I had a great day however I '\
                               'did not get much work done in the day']]
        test_left_results = context(test_values, 'left', inc_target=True)
        test_contexts(valid_left_results, test_left_results)

        valid_right_results = [['news_article', 'news_article'],
                               ['day however I did not get much work '\
                                'done in the day', 'day']]
        test_right_results = context(test_values, 'right', inc_target=True)
        test_contexts(valid_right_results, test_right_results)

        valid_target_results = [['news_article', 'news_article'], ['day', 'day']]
        test_target_results = context(test_values, 'target', inc_target=True)
        test_contexts(valid_target_results, test_target_results)

        valid_full_results = [['This is a fake news_article',
                               'dd that is to represent a Tweet and it was '\
                               'an awful news_article'],
                              ['I had a great day however I did not get much '\
                               'work done in the day',
                               'I had a great day however I did not get much '\
                                'work done in the day']]
        test_full_results = context(test_values, 'full', inc_target=True)
        test_contexts(valid_full_results, test_full_results)
