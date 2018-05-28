'''
Unit test suite for :py:mod:`bella.lexicons` module.
'''
from unittest import TestCase

import pytest

from bella.lexicons import Mpqa
from bella.lexicons import HuLiu
from bella.lexicons import NRC
from bella.lexicons import Lexicon

class TestLexicons(TestCase):
    '''
    Tests the :py:class:`bella.lexicons.Lexicon` class and all of it's
    subclasses.
    '''

    def test_lexicon_combo(self):
        '''
        Tests combining two Lexicon classes together. Ensures that words are
        removed if they have conflicting categories.
        '''
        with self.assertRaises(TypeError, msg='Should only accept type of '\
                               'Lexicon not lists'):
            Lexicon.combine_lexicons([('great', 'positive')],
                                     [('best', 'positive')])

        fake_lexicon1 = [('happy', 'positive'), ('unfair', 'positive'),
                         ('nothing', 'negative')]
        fake_lexicon2 = [('great', 'positive'), ('unfair', 'negative'),
                         ('hate', 'negative')]
        fake_lex1 = Lexicon(lexicon=fake_lexicon1, name='fake 1')
        fake_lex2 = Lexicon(lexicon=fake_lexicon2, name='fake 2')

        combined_fake = Lexicon.combine_lexicons(fake_lex1, fake_lex2)
        combined_lexicon = set(combined_fake.lexicon)
        valied_combined = set([('happy', 'positive'), ('great', 'positive'),
                               ('hate', 'negative'), ('nothing', 'negative')])
        self.assertEqual(combined_lexicon, valied_combined, msg='These should be '\
                         'equal {} {}'.format(combined_lexicon, valied_combined))

        self.assertEqual('fake 1 fake 2', combined_fake.name, msg='Name of the '\
                         'combined lexicon should be {} not {}'\
                         .format('fake 1 fake 2', combined_fake.name))

        fake_lexicon3 = [('happy', 'negative'), ('great', 'negative'),
                         ('fantastic', 'positive')]
        fake_lex3 = Lexicon(lexicon=fake_lexicon3, name='fake 3')

        combined_fake1 = Lexicon.combine_lexicons(combined_fake, fake_lex3)
        combined_lexicon1 = set(combined_fake1.lexicon)
        valid_combined1 = set([('hate', 'negative'), ('nothing', 'negative'),
                               ('fantastic', 'positive')])
        self.assertEqual(combined_lexicon1, valid_combined1, msg='These should be '\
                         'equal {} {}'.format(combined_lexicon1, valid_combined1))
        self.assertEqual('fake 1 fake 2 fake 3', combined_fake1.name,
                         msg='Name of the combined lexicon should be {} not {}'\
                         .format('fake 1 fake 2 fake 3', combined_fake1.name))

    def test_lexicon_values(self):
        '''
        Tests that :py:func:`bella.lexicons.Lexicon` checks the types of
        self.lexicon and if setting the lexicon in the constructor works.
        '''
        with self.assertRaises(TypeError, msg='Should not allow lexicon that is'\
                               ' of type set'):
            Lexicon(lexicon={'something'})
        with self.assertRaises(TypeError, msg='Should not allow lexicon that is '\
                               'not made up of tuples in the list'):
            Lexicon(lexicon=['anything'])
        # Should allow the following
        valid_words = {'great', 'fantastic'}
        valid_lexicon = Lexicon(lexicon=[('great', 'positive'),
                                         ('fantastic', 'positive')])
        self.assertEqual(valid_words, valid_lexicon.words, msg='These should be '\
                         'equal {} {}'.format(valid_words, valid_lexicon.words))

    @pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
                      "not own the lexicons therefore cannot release them.")
    def test_subset_values_class(self):
        '''
        Tests that :py:func:`bella.lexicons.Lexicon._process_lexicon` function
        works correctly. The functionality is lower casing the lexicon words and
        removing words from the lexicon if they are not within
        '''

        with self.assertRaises(TypeError, msg='Should not allow list types for '\
                               'the `subset_values` parameter type found only set'):
            HuLiu(['positive'])

        with self.assertRaises(TypeError, msg='Should not allow 1 as a substitute'\
                               ' for True boolean'):
            HuLiu({'positive'}, 1)

        # Ensure the lower cases works
        mpqa_upper = Mpqa(lower=False).lexicon
        normal_mpqa = Mpqa().lexicon
        mpqa_lower = Mpqa(lower=True).lexicon
        false_value = mpqa_lower == mpqa_upper
        self.assertEqual(False, false_value, msg='The upper case version of mpqa '\
                         'should not equal the lower case')
        self.assertEqual(normal_mpqa, mpqa_upper, msg='Default values of lower '\
                         'should be False')

        valid_subset_size = 2006
        subset_size = len(HuLiu(subset_cats={'positive'}, lower=True).lexicon)
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'positive hu and liu lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
        valid_subset_size = 4783
        subset_size = len(HuLiu(subset_cats={'negative'}, lower=True).lexicon)
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'negative hu and liu lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))

        valid_subset_size = 2312
        subset_size = len(NRC(subset_cats={'positive'}, lower=True).lexicon)
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'positive nrc emotion lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
        valid_subset_size = 3324
        subset_size = len(NRC(subset_cats={'negative'}, lower=True).lexicon)
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'negative nrc emotion lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))

        valid_subset_size = 2304
        subset_size = len(Mpqa(subset_cats={'positive'}, lower=True).lexicon)
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'positive mpqa lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))
        valid_subset_size = 4154
        subset_size = len(Mpqa(subset_cats={'negative'}, lower=True).lexicon)
        self.assertEqual(valid_subset_size, subset_size, msg='The size of the '\
                         'negative mpqa lexicon should be {} not {}'\
                         .format(valid_subset_size, subset_size))

    @pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
                      "not own the lexicons therefore cannot release them.")
    def test_nrc(self):
        '''
        Tests the :py:class:`bella.lexicons.NRC`
        '''

        nrc_lex = NRC()
        word_sentiment = nrc_lex.lexicon
        self.assertIsInstance(word_sentiment, list, msg='The return type should '\
                              'of type list not {}'.format(type(word_sentiment)))
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
    def test_huliu(self):
        '''
        Tests the :py:class:`bella.lexicons.HuLiu`
        '''

        hu_liu_lex = HuLiu()
        word_sentiment = hu_liu_lex.lexicon
        self.assertIsInstance(word_sentiment, list, msg='The return type should '\
                              'of type list not {}'.format(type(word_sentiment)))
        self.assertIsInstance(word_sentiment[0], tuple, msg='The list should be a'\
                              ' list of tuples not {}'.format(type(word_sentiment[0])))

        value_types = set([sentiment for word, sentiment in word_sentiment])
        valid_value_types = {'positive', 'negative'}
        self.assertEqual(value_types, valid_value_types, msg='The values associated'\
                         ' to the words in the lexicons should be only {}'\
                         ' and not {}'.format(valid_value_types, value_types))

        lexicon_length = len(word_sentiment)
        valid_lexicon_length = 6789
        self.assertEqual(valid_lexicon_length, lexicon_length, msg='The number of '\
                         'words in the lexicon should be {} and not {}'\
                         .format(valid_lexicon_length, lexicon_length))

    @pytest.mark.skip(reason="Requires the files to be downloaded and we do "\
                      "not own the lexicons therefore cannot release them.")
    def test_mpqa(self):
        '''
        Tests the :py:class:`bella.lexicons.Mpqa`
        '''

        mpqa_lex = Mpqa()
        word_sentiment = mpqa_lex.lexicon
        self.assertIsInstance(word_sentiment, list, msg='The return type should '\
                              'of type list not {}'.format(type(word_sentiment)))
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
