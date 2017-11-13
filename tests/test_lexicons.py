'''
Unit test suite for :py:mod:`tdparse.lexicons` module.
'''
from unittest import TestCase

import pytest

from tdparse.lexicons import combine_lexicons
from tdparse.lexicons import hu_liu
from tdparse.lexicons import nrc_emotion
from tdparse.lexicons import mpqa

class TestLexicons(TestCase):
    '''
    Contains the following functions:

    1. :py:func:`tdparse.lexicons.hu_liu`
    2. :py:func:`tdparse.lexicons.nrc_emotion`
    3. :py:func:`tdparse.lexicons.mpqa`
    '''

    def test_combine_lexicons(self):
        '''
        Tests the :py:func:`tdparse.lexicons.combine_lexicons`
        '''
        with self.assertRaises(TypeError, msg='Should only accept lists and not '\
                               'sets'):
            combine_lexicons(set([('another', 'positive')]), set([('best', 'positive')]))

        wrong_values1 = [('hate', 'negative')]
        wrong_values2 = [('happy', 'positive')]
        with self.assertRaises(ValueError, msg='The two lists of tuples should '\
                               'have the same values.'):
            combine_lexicons(wrong_values1, wrong_values2)

        fake_lexicon1 = [('happy', 'positive'), ('unfair', 'positive'),
                         ('nothing', 'negative')]
        fake_lexicon2 = [('great', 'positive'), ('unfair', 'negative'),
                         ('hate', 'negative')]
        combined_fake = set(combine_lexicons(fake_lexicon1, fake_lexicon2))
        valied_combined = set([('happy', 'positive'), ('great', 'positive'),
                               ('hate', 'negative'), ('nothing', 'negative')])
        self.assertEqual(combined_fake, valied_combined, msg='These should be '\
                         'equal {} {}'.format(combined_fake, valied_combined))
        fake_lexicon3 = [('happy', 'negative'), ('great', 'negative'),
                         ('fantastic', 'positive')]
        combined_fake1 = set(combine_lexicons(list(combined_fake), fake_lexicon3))
        valid_combined1 = set([('hate', 'negative'), ('nothing', 'negative'),
                               ('fantastic', 'positive')])
        self.assertEqual(combined_fake1, valid_combined1, msg='These should be '\
                         'equal {} {}'.format(combined_fake1, valid_combined1))



    def test_subset_values(self):
        '''
        Tests the :py:func:`tdparse.lexicons.parameter_check` decorator works.
        '''

        with self.assertRaises(TypeError, msg='Should not allow list types for '\
                               'the `subset_values` parameter type found only set'):
            hu_liu(['positive'])

        valid_subset_size = 2006
        subset_size = len(hu_liu({'positive'}))
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'positive hu and liu lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
        valid_subset_size = 4783
        subset_size = len(hu_liu({'negative'}))
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'negative hu and liu lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))

        valid_subset_size = 2312
        subset_size = len(nrc_emotion({'positive'}))
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'positive nrc emotion lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
        valid_subset_size = 3324
        subset_size = len(nrc_emotion({'negative'}))
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'negative nrc emotion lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))

        valid_subset_size = 2304
        subset_size = len(mpqa({'positive'}))
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'positive mpqa lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
        valid_subset_size = 4154
        subset_size = len(mpqa({'negative'}))
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'negative mpqa lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
    #@pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
    #                  "not own the lexicons therefore cannot release them.")
    def test_hu_liu(self):
        '''
        Tests the :py:class:`tdparse.lexicons.hu_liu`
        '''
        word_sentiment = hu_liu()
        self.assertIsInstance(word_sentiment, list, msg='The return type should '\
                              'of type list not {}'.format(type(hu_liu)))
        self.assertIsInstance(word_sentiment[0], tuple, msg='The list should be a'\
                              ' list of tuples not {}'.format(type(word_sentiment[0])))

        value_types = set([sentiment for word, sentiment in word_sentiment])
        valid_value_types = {'positive', 'negative'}
        self.assertEqual(value_types, valid_value_types, msg='The values associated'\
                         ' to the words in the lexicons should be only `positive`'\
                         ' and `negative` and not {}'.format(value_types))

        lexicon_length = len(word_sentiment)
        valid_lexicon_length = 6789
        self.assertEqual(valid_lexicon_length, lexicon_length, msg='The number of '\
                         'words in the lexicon should be {} and not {}'\
                         .format(valid_lexicon_length, lexicon_length))

    #@pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
    #                  "not own the lexicons therefore cannot release them.")
    def test_nrc_emotion_sentiment(self):
        '''
        Tests the :py:class:`tdparse.lexicons.nrc_emotion`
        '''
        word_sentiment = nrc_emotion()
        self.assertIsInstance(word_sentiment, list, msg='The return type should '\
                              'of type list not {}'.format(type(hu_liu)))
        self.assertIsInstance(word_sentiment[0], tuple, msg='The list should be a'\
                              ' list of tuples not {}'.format(type(word_sentiment[0])))

        value_types = set([sentiment for word, sentiment in word_sentiment])
        valid_value_types = {'positive', 'negative', 'anger', 'fear', 'anticipation',
                             'trust', 'surprise', 'sadness', 'joy', 'disgust'}
        self.assertEqual(value_types, valid_value_types, msg='The values associated'\
                         ' to the words in the lexicons should be only {}'\
                         ' and not {}'.format(valid_value_types, value_types))

    #@pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
    #                  "not own the lexicons therefore cannot release them.")
    def test_mpqa(self):
        '''
        Tests the :py:class:`tdparse.lexicons.mpqa`
        '''
        word_sentiment = mpqa()
        self.assertIsInstance(word_sentiment, list, msg='The return type should '\
                              'of type list not {}'.format(type(hu_liu)))
        self.assertIsInstance(word_sentiment[0], tuple, msg='The list should be a'\
                              ' list of tuples not {}'.format(type(word_sentiment[0])))

        value_types = set([sentiment for word, sentiment in word_sentiment])
        valid_value_types = {'positive', 'negative', 'both', 'neutral'}
        self.assertEqual(value_types, valid_value_types, msg='The values associated'\
                         ' to the words in the lexicons should be only {}'\
                         ' and not {}'.format(valid_value_types, value_types))

        lexicon_length = len(word_sentiment)
        valid_lexicon_length = 6905
        self.assertEqual(valid_lexicon_length, lexicon_length, msg='The number of '\
                         'words in the lexicon should be {} and not {}'\
                         .format(valid_lexicon_length, lexicon_length))
