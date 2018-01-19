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

        new_test_case = "#britneyspears Britney Spears 's new single -3' debuts "\
                        "at #1: video: congratulations are in order .."

        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful News Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (83, 95)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(16, 19), (64, 67)]},
                       {'target_id':str(2),
                        'sentiment':1,
                        'text':'I had a great ssDay however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(16, 19)]},
                       {'target_id':str(3),
                        'sentiment':1,
                        'text':'Day however I did not get much done',
                        'target':'day',
                        'spans':[(0, 3)]},
                       {'target_id':str(4),
                        'sentiment':1,
                        'text':'however I did not get much done in the day',
                        'target':'day',
                        'spans':[(39, 42)]},
                       {'spans': [(47, 80)],
                        'target_id': '2',
                        'target': 'Core Processing Unit temperatures',
                        'text': 'Temperatures were ok but I was not tracking'\
                                ' in Core Processing Unit temperatures.',
                        'sentiment': 0},
                       {'spans': [(1, 14), (15, 29)],
                        'target_id': '8',
                        'target': 'britney spears',
                        'text': "#britneyspears Britney Spears 's new single "\
                                "-3' debuts at #1: video: congratulations are "\
                                "in order ..",
                        'sentiment': 0},
                       {'spans': [(39, 50), (53, 64)],
                        'target_id': '9',
                        'target': 'Sarah Palin',
                        'text': "Letterman Apologizes to His Wife...and "\
                                "Sarah Palin - Sarah Palin , I'm terribly, "\
                                "terribly sorry. So there we g...",
                        'sentiment': 0}
                      ]
        valid_results = ['This is a fake {} dd that is to represent a '\
                         'Tweet!!!! and it was an awful {} ss I think.'\
                         .format('$News_Article$', '$News_Article$'),
                         'I had a great ss {} however I did not get much '\
                         'work done in the {} s'\
                         .format('$Day$', '$Day$'),
                         'I had a great ss {} however I did not get much '\
                         'work done in the days'.format('$Day$'),
                         '{} however I did not get much done'\
                         .format('$Day$'),
                         'however I did not get much done in the {}'\
                         .format('$Day$'),
                         'Temperatures were ok but I was not tracking'\
                         ' in {} .'.format('$Core_ProcessingUnitTemperatures$'),
                         "# {} {} 's new single -3' debuts at #1: video: "\
                         "congratulations are in order .."\
                         .format('$Britney_Spears$', '$Britney_Spears$'),
                         "Letterman Apologizes to His Wife...and "\
                         "{} - {} , I'm terribly, terribly sorry. So there "\
                         "we g...".format("$Sarah_Palin$", "$Sarah_Palin$")]
        test_values = [Target(**test_value) for test_value in test_values]
        for index, test_value in enumerate(test_values):
            test_result = target_normalisation(test_value)
            valid_result = valid_results[index]
            self.assertEqual(valid_result, test_result, msg='Results is '\
                             '{} and should be {}. Test value {}'\
                             .format(test_result, valid_result, test_value))
    def test_dependency_relation_context(self):
        '''
        Tests dependency_relation_context
        '''

        # Test the normalise case
        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful News Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (83, 95)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great Day however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(14, 17), (62, 65)]},
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
                         ['on heartthrobe rage Ten pop star the media social all'],
                         ['']]
        test_values = [Target(**test_value) for test_value in test_values]
        test_results = dependency_relation_context(test_values, tweebo,
                                                   n_relations=(1, -1))
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            self.assertEqual(valid_result, test_result, msg='Incorrect context'\
                             ' correct {} test {}'.format(valid_result, test_result))

        # Testing when we only want the first dependency relation
        valid_results = [['a fake', 'an awful'], ['a great', 'the'],
                         ['on heartthrobe rage'], ['']]
        test_results = dependency_relation_context(test_values, tweebo)
        for index, valid_result in enumerate(valid_results):
            test_result = test_results[index]
            self.assertEqual(valid_result, test_result, msg='Incorrect context'\
                             ' correct {} test {}'.format(valid_result, test_result))

        # Testing to ensure it will lower case the words before processing
        valid_results = [['a fake', 'an awful'], ['a great', 'the'],
                         ['on heartthrobe rage ten pop star the media social all'],
                         ['']]
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
                        'text':'I had a great Day however I did not get much '\
                        'work done in the day',
                        'target':'day',
                        'spans':[(14, 17)]}]
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
                        'Tweet!!!! and it was an awful News Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (83, 95)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great Day however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(14, 17), (62, 65)]},
                       {'spans': [(1, 14), (15, 29)],
                        'target_id': '8',
                        'target': 'britney spears',
                        'text': "#britneyspears Britney Spears 's new single "\
                                "-3' debuts at #1: video: congratulations are "\
                                "in order ..",
                        'sentiment': 0},
                       {'spans': [(39, 50), (53, 64)],
                        'target_id': '9',
                        'target': 'Sarah Palin',
                        'text': "Letterman Apologizes to His Wife...and Sarah Palin - Sarah Palin , I'm terribly, terribly sorry. So there we g...",
                                #"Sarah Palin - Sarah Palin , I'm terribly, "\
                                #"terribly sorry. So there we g...",
                        'sentiment': 0}]
        valid_results = [[{'text' : 'This is a fake  news article  ',
                           'span' : (16, 28)},
                          {'text' : 'dd that is to represent a Tweet and it was '\
                                    'an awful  news article  ',
                           'span' : (53, 65)}],
                         [{'text' : 'I had a great  day  however I did not get '\
                                    'much work done in the  day  ',
                           'span' : (15, 18)},
                          {'text' : 'I had a great  day  however I did not get '\
                                    'much work done in the  day  ',
                           'span' : (65, 68)}],
                         [{'text' : "  britney spears   britney spears  ",
                           'span' : (2, 16)},
                          {'text' : "  britney spears   britney spears  ",
                           'span' : (19, 33)}],
                         [{'text' : "Letterman Apologizes to His Wife and  Sarah Palin  ",
                           'span' : (38, 49)},
                          {'text' : "  Sarah Palin  ",
                           'span' : (2, 13)}]]
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
                                     ' text `{}`'.format(valid_dict['span'],
                                                         test_dict['span'],
                                                         test_dict['text']))
        # Test the lower casing case
        test_values = [{'target_id':str(0),
                        'sentiment':-1,
                        'text':'This is a fake news articledd that is to represent a '\
                        'Tweet!!!! and it was an awful News Articless I think.',
                        'target':'news article',
                        'spans':[(15, 27), (83, 95)]},
                       {'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great Day however I did not get much '\
                        'work done in the days',
                        'target':'day',
                        'spans':[(14, 17), (62, 65)]}]
        valid_results = [[{'text' : 'this is a fake  news article  ',
                           'span' : (16, 28)},
                          {'text' : 'dd that is to represent a tweet and it was '\
                                    'an awful  news article  ',
                           'span' : (53, 65)}],
                         [{'text' : 'i had a great  day  however i did not get '\
                                    'much work done in the  day  ',
                           'span' : (15, 18)},
                          {'text' : 'i had a great  day  however i did not get '\
                                    'much work done in the  day  ',
                           'span' : (65, 68)}]]
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
        # Test the case where the target is mentioned twice but only 1 is relevant
        # to one of the mentions
        test_values = [{'target_id':str(1),
                        'sentiment':1,
                        'text':'I had a great Day however I did not get much '\
                        'work done in the day',
                        'target':'day',
                        'spans':[(14, 17)]},
                       {'target_id':str(3),
                        'sentiment':1,
                        'text':'I had a great Day however I did not get much '\
                        'work done in the Day',
                        'target':'day',
                        'spans':[(14, 17)]},
                        {'spans': [(47, 80)],
                         'target_id': '2',
                         'target': 'Core Processing Unit temperatures',
                         'text': 'Temperatures were ok but I was not tracking'\
                                 ' in Core Processing Unit temperatures.',
                         'sentiment': 0}]
        valid_results = [[{'text' : 'I had a great  day  however I did not get '\
                                    'much work done in the day',
                           'span' : (15, 18)}],
                          [{'text' : 'I had a great  day  however I did not get '\
                                     'much work done in the Day',
                            'span' : (15, 18)}],
                          [{'text' : 'Temperatures were ok but I was not tracking'\
                                     ' in  Core Processing Unit temperatures  ',
                            'span' : (48, 81)}]]
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
