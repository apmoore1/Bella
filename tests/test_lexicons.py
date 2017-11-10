'''
Unit test suite for :py:mod:`tdparse.lexicons` module.
'''
from unittest import TestCase

import pytest

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

    @pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
                      "not own the lexicons therefore cannot release them.")
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

    @pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
                      "not own the lexicons therefore cannot release them.")
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

    @pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
                      "not own the lexicons therefore cannot release them.")
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
        valid_lexicon_length = 8222
        self.assertEqual(valid_lexicon_length, lexicon_length, msg='The number of '\
                         'words in the lexicon should be {} and not {}'\
                         .format(valid_lexicon_length, lexicon_length))
