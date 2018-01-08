'''
Unit test suite for the :py:mod:`tdparse.data_types` module.
'''
import copy
from unittest import TestCase

from tdparse.data_types import Target
from tdparse.data_types import TargetCollection

class TestTarget(TestCase):
    '''
    Contains the following functions:
    '''

    def test_target_constructor(self):
        '''
        Test that target constructor
        '''

        # Testing the spans types
        with self.assertRaises(TypeError, msg='Spans should be of type list'):
            Target('span', '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(TypeError, msg='Spans should be list of tuples'):
            Target([1, 2], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(ValueError, msg='Spans should contain tuples of '\
                               'length 2'):
            Target([(1, 2, 3), (3, 4, 5)], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(ValueError, msg='Spans should contain tuples of '\
                               'length 2'):
            Target([(1, 2), (3, 4, 5)], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(TypeError, msg='Spans should contain tuples of '\
                               'length 2 and are Ints'):
            Target([('1', '2')], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(TypeError, msg='Spans should contain tuples of '\
                               'length 2 and are Ints'):
            Target([(1, '2')], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(TypeError, msg='Spans should contain tuples of '\
                               'length 2 and are Ints'):
            Target([('1', 2)], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(ValueError, msg='Spans should contain tuples of '\
                               'Ints where the first Int < second Int'):
            Target([(7, 5)], '1', 'Iphone', 'text with Iphone', 'Pos')
        with self.assertRaises(ValueError, msg='Spans should contain tuples of '\
                               'Ints where the first Int < second Int'):
            Target([(3, 5), (6, 6)], '1', 'Iphone', 'text with Iphone', 'Pos')
        # Testing that the spans work in a normal case
        Target([(3, 5), (6, 8)], '1', 'Iphone', 'text with Iphone', 'Pos')

        # Testing the target ID type
        with self.assertRaises(TypeError, msg='Target ID should be a String'):
            Target([(3, 5), (6, 8)], 1, 'Iphone', 'text with Iphone', 'Pos')

        # Testing target type
        with self.assertRaises(TypeError, msg='Target should be a String'):
            Target([(3, 5), (6, 8)], '1', ('Iphone',), 'text with Iphone', 'Pos')

        # Testing text type
        with self.assertRaises(TypeError, msg='Text should be a String'):
            Target([(3, 5), (6, 8)], '1', 'Iphone', ('text with Iphone',), 'Pos')

        # Testing sentiment type
        with self.assertRaises(TypeError, msg='Sentiment should be a String or '\
                               'Int'):
            Target([(3, 5), (6, 8)], '1', 'Iphone', 'text with Iphone', ('Pos',))
        # Testing the sentence_id type
        with self.assertRaises(TypeError, msg='sentence_id should be a String'):
            Target([(3, 5), (6, 8)], '1', 'Iphone', 'text with Iphone', 'pos',
                   sentence_id=1)
        # Testing the sentiment type works as an Integer (Normal case)
        span = [(3, 5)]
        target = 'Iphone'
        sentiment = 1
        text = 'text with Iphone'
        target_id = '210#1'
        predicted = 0
        sentence_id = '210'
        target_example = Target(span, target_id, target, text, sentiment, 
                                predicted, sentence_id)

        # Testing that the dictionary mapping is correct
        self.assertEqual(target_id, target_example['target_id'],
                         msg='The target ID should {} and not {}'\
                         .format(target_id, target_example['target_id']))
        self.assertEqual(text, target_example['text'],
                         msg='The text should be {} and not {}'\
                         .format(text, target_example['text']))
        self.assertEqual(sentiment, target_example['sentiment'],
                         msg='The sentiment should be {} and not {}'\
                         .format(sentiment, target_example['sentiment']))
        self.assertEqual(target, target_example['target'],
                         msg='The target should be {} and not {}'\
                         .format(target, target_example['target']))
        self.assertEqual(span, target_example['spans'],
                         msg='The spans should be {} and not {}'\
                         .format(span, target_example['spans']))
        self.assertEqual(predicted, target_example['predicted'],
                         msg='The predicted sentiment should be {} and not {}'\
                         .format(predicted, target_example['predicted']))
        self.assertEqual(sentence_id, target_example['sentence_id'],
                         msg='The sentence_id should be {} and not {}'\
                         .format(sentence_id, target_example['sentence_id']))

    def test_target_eq(self):
        '''
        Test the Target __eq__ method
        '''

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 1)
        target_example_dup0 = Target([(1, 2)], '1', 'S8',
                                     'text with Samsung S8', 1)
        # Targets with the same ID should be True even if they have different
        # data
        self.assertEqual(target_example_0, target_example_dup0,
                         msg='Should be equal as they have the same ID `1`')
        # Targets with the same minimum keys should be True
        target_example_dup1 = copy.deepcopy(target_example_1)
        del target_example_dup1['target_id']
        self.assertEqual(target_example_1, target_example_dup1, msg='Should be'\
                         ' equal as they have the same minimum keys')
        # Normal case
        target_example_dup2 = copy.deepcopy(target_example_2)
        self.assertEqual(target_example_2, target_example_dup2, msg='Copies of'\
                         ' the same Target instance should be equal')
        # Test that it won't accept dicts with the same minimum keys
        dict_example_1 = {'target' : 'Iphone', 'sentiment' : 1,
                          'spans' : [(1, 5)], 'text' : 'text with Iphone'}
        test_equality = dict_example_1 == target_example_1
        self.assertEqual(False, test_equality, msg='Should not accept dicts '\
                         'even with the same minimum_keys')

    def test_target_set(self):
        '''
        Tests Target set function
        '''

        # Test normal cases
        target_example_int = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                    'text with Iphone', 1)
        predicted_sentiment = 0
        target_example_int['predicted'] = 0
        self.assertEqual(predicted_sentiment, target_example_int['predicted'], 
                         msg='Predicted sentiment value should be 0 and '\
                         'not {}'.format(target_example_int['predicted']))
        
        target_example_string = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                       'text with Iphone', 'pos')
        predicted_sentiment = 'neg'
        target_example_string['predicted'] = 'neg'
        self.assertEqual(predicted_sentiment, target_example_string['predicted'],
                         msg='Predicted sentiment value should be `neg` and '\
                         'not {}'.format(target_example_string['predicted']))

        # Testing the errors
        with self.assertRaises(KeyError, msg='Should not allow you to set keys '\
                               'other than `predicted`'):
            target_example_int['sentiment'] = 0
        with self.assertRaises(TypeError, msg='Should not allow you to set '\
                               'predicted sentiment to a data type that is not '\
                               'the same as the sentiment value data type'):
            target_example_int['predicted'] = 'pos'
        with self.assertRaises(TypeError, msg='Should not allow you to set '\
                               'predicted sentiment to a data type that is not '\
                               'the same as the sentiment value data type'):
            target_example_string['predicted'] = 1






    def test_targetcoll_constructor(self):
        '''
        Tests TargetCollection constructor
        '''

        target_example = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                'text with Iphone', 1)
        # Test that it will not accept anything but Target instances
        with self.assertRaises(TypeError, msg='The constructor should only '\
                               'accept an interator as an argument'):
            TargetCollection(1)
        with self.assertRaises(TypeError, msg='The constructor should only '\
                               'accept an interator of Target instances'):
            TargetCollection([1, 2, 3, 4])
        # Should accept the following without any problems
        TargetCollection([])
        TargetCollection([target_example])
        TargetCollection()

        # Testing the case where the list of Targets contains duplicate keys
        another_example = Target([(3, 4)], '2', 'Keys',
                                 'text with Keys', -1)
        dup_example = Target([(3, 10)], '1', 'Pixel',
                             'text with Pixels', 0)
        with self.assertRaises(KeyError, msg='Should raise an error as two of '\
                               'the target instances have the same key'):
            TargetCollection([target_example, another_example, dup_example])

    def test_targetcoll_get(self):
        '''
        Test the __getitem__ function of TargetCollection
        '''

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 1)
        target_col = TargetCollection([target_example_0, target_example_1,
                                       target_example_2])

        # Test normal case
        self.assertEqual(target_example_2, target_col['2'], msg='Cannot access '\
                         'data using keys. key used {} collection {}'\
                         .format('2', target_col))
        self.assertEqual(target_example_1, target_col.get('3'), msg='Cannot '\
                         'access data using the get method.')
        self.assertEqual(None, target_col.get('5'), msg='Default value for '\
                         'get not working should be None not {}'\
                         .format(target_col.get('5')))
        # Test that it raises a KeyError when key does not exist
        with self.assertRaises(KeyError, msg='Should produce a key error when '\
                               'the data does not exist'):
            target_col['5']

    def test_targetcoll_set(self):
        '''
        Test the __setitem__ function of TargetCollection
        '''

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 1)
        target_col = TargetCollection([target_example_0, target_example_1,
                                       target_example_2])
        target_example_3 = Target([(2, 4)], '5', 'new', 'new text', 0)
        target_example_4 = Target([(1, 3)], '6', 'another', 'another text', 1)
        target_example_5 = Target([(1, 3)], '7', 'another', 'another text', 1)
        target_diff_1 = Target([(4, 5)], '3', 'test', 'test text', 0)

        # Normal case adding a new value
        target_col['5'] = target_example_3
        self.assertEqual(target_col['5'], target_example_3, msg='Cannot add '\
                         'new value. store {} value added {}'\
                         .format(target_col, target_example_3))
        # If key already exists it cannot be added
        with self.assertRaises(KeyError, msg='Should not be able to add value '\
                               '{} as its key {} already exists {}'\
                               .format(target_diff_1, '3', target_col)):
            target_col['3'] = target_diff_1
        with self.assertRaises(KeyError, msg='Value with a different `id` to '\
                               'the key should fail. Key {} Value {}'\
                               .format('7', target_example_4)):
            target_col['7'] = target_example_4
        # Should accept Target instance with no `id`
        del target_example_5['target_id']
        if 'target_id' in target_example_5:
            raise KeyError('{} should not contain `id` key'\
            .format(target_example_5))
        target_col['8'] = target_example_5

    def test_targetcoll_add(self):
        '''
        Test the add function of TargetCollection
        '''

        target_col = TargetCollection()
        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        # Ensure the normal case works
        target_col.add(target_example_0)
        self.assertEqual(target_col['1'], target_example_0, msg='Test that {}' \
                         ' has been added to {}'\
                         .format(target_example_0, target_col))

        with self.assertRaises(TypeError, msg='Should not be able to add a dict'):
            target_col.add({'target_id' : '2'})

        with self.assertRaises(ValueError, msg='Should not be able to add a '\
                               'Target that has no `id`'):
            del target_example_1['target_id']
            if 'target_id' in target_example_1:
                raise KeyError('{} should not contain `id` key'\
                .format(target_example_1))
            target_col.add(target_example_1)

    def test_targetcoll_data(self):
        '''
        Test the data function of TargetCollection
        '''

        target_col = TargetCollection()
        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_col.add(target_example_0)
        target_col.add(target_example_1)

        all_data = target_col.data()
        self.assertEqual(target_example_0, all_data[0], msg='First data '\
                         'returned should be the first inserted {} and not '\
                         '{}'.format(target_example_0, all_data[0]))
        self.assertEqual(target_example_1, all_data[1], msg='Second data '\
                         'returned should be the second inserted {} and not '\
                         '{}'.format(target_example_1, all_data[1]))

        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 1)
        del target_col['1']
        target_col.add(target_example_2)
        all_data = target_col.data()
        self.assertEqual(target_example_1, all_data[0], msg='First data '\
                         'returned should be the second inserted {} and not '\
                         '{} as the first has been removed'\
                         .format(target_example_1, all_data[0]))
        self.assertEqual(target_example_2, all_data[1], msg='Second data '\
                         'returned should be the third inserted {} and not '\
                         '{} as the first has been removed'\
                         .format(target_example_2, all_data[1]))
        self.assertEqual(2, len(all_data), msg='The length of the data returned'\
                         'shoudl be 2 and not {}'.format(len(all_data)))

    def test_targetcoll_stored_sent(self):
        '''
        Test the stored_sentiments function of TargetCollection
        '''

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', -1)
        target_col = TargetCollection([target_example_0, target_example_1,
                                       target_example_2])
        valid_sentiments = set([1, -1])
        test_sentiments = target_col.stored_sentiments()
        self.assertEqual(valid_sentiments, test_sentiments, msg='The unique '\
                         'sentiments in the TargetCollection should be {} and '\
                         'not {}'.format(valid_sentiments, test_sentiments))

    def test_targetcoll_sent_data(self):
        '''
        Test the sentiment_data function of TargetCollection
        '''

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', -1)
        target_col_int = TargetCollection([target_example_0, target_example_1,
                                           target_example_2])

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 'pos')
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 'pos')
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 'neg')
        target_col_str = TargetCollection([target_example_0, target_example_1,
                                           target_example_2])
        # Testing the basic example
        test_sentiments = target_col_int.sentiment_data()
        valid_sentiments = [1, 1, -1]
        self.assertEqual(valid_sentiments, test_sentiments, msg='The Integer '\
                         'sentiments returned should be {} and not {}'\
                         .format(valid_sentiments, test_sentiments))
        test_sentiments = target_col_str.sentiment_data()
        valid_sentiments = ['pos', 'pos', 'neg']
        self.assertEqual(valid_sentiments, test_sentiments, msg='The String '\
                         'sentiments returned should be {} and not {}'\
                         .format(valid_sentiments, test_sentiments))

        # Testing the mapping function
        str_mapper = {'pos' : 1, 'neg' : -1}
        test_sentiments = target_col_str.sentiment_data(mapper=str_mapper)
        valid_sentiments = [1, 1, -1]
        self.assertEqual(valid_sentiments, test_sentiments, msg='The String '\
                         'sentiments should be mapped to Integers. Valid {} '\
                         'not {}'.format(valid_sentiments, test_sentiments))
        int_mapper = {1 : 'pos', -1 : 'neg'}
        test_sentiments = target_col_int.sentiment_data(mapper=int_mapper)
        valid_sentiments = ['pos', 'pos', 'neg']
        self.assertEqual(valid_sentiments, test_sentiments, msg='The Integer '\
                         'sentiments should be mapped to String. Valid {} '\
                         'not {}'.format(valid_sentiments, test_sentiments))

        with self.assertRaises(TypeError, msg='Should only accept dict mapper'):
            target_col_int.sentiment_data(mapper=[(1, 'pos'), (-1, 'neg')])
        with self.assertRaises(ValueError, msg='Mapper should refuse dicts that'\
                               ' may have valid mappings but not all the mappings'):
            target_col_int.sentiment_data(mapper={1 : 'pos'})
        with self.assertRaises(ValueError, msg='Mapper should refuse dicts that'\
                               ' contain the correct number of mappings but not '\
                               'the correct mappings'):
            target_col_int.sentiment_data({0 : 'pos', -1 : 'neg'})
        with self.assertRaises(ValueError, msg='Mapper should refuse dicts that '\
                               'have all the correct mappings but contain some '\
                               'in-correct mappings'):
            target_col_int.sentiment_data(mapper={1 : 'pos', -1 : 'neg',
                                                  0 : 'neu'})

        # Testing the sentiment_field parameter
        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 'pos', 'neg')
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 'pos', 'neu')
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 'neg', 'pos')
        target_col = TargetCollection([target_example_0, target_example_1,
                                       target_example_2])
        test_sentiments = target_col.sentiment_data(sentiment_field='predicted')
        valid_sentiments = ['neg', 'neu', 'pos']
        self.assertEqual(valid_sentiments, test_sentiments, msg='The predicted '\
                         'sentiments returned should be {} and not {}'\
                         .format(valid_sentiments, test_sentiments))


    def test_targetcoll_add_preds(self):
        '''
        Tests the add_pred_sentiment function of TargetCollection
        '''
        
        target_example_int_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 1)
        target_example_int_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 1)
        target_example_int_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', -1)
        target_col_int = TargetCollection([target_example_int_0, 
                                           target_example_int_1,
                                           target_example_int_2])

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 'pos')
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 'pos')
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 'neg')
        target_col_str = TargetCollection([target_example_0, target_example_1,
                                           target_example_2])

        pred_sents = [2, 5, 4]
        target_col_int.add_pred_sentiment(pred_sents)
        self.assertEqual(5, target_col_int['3']['predicted'], msg='Predicted '\
                         'sentiment not set correctly for `id` 3 should be 5 '\
                         'and not {}'.format(target_col_int))
        with self.assertRaises(KeyError, msg='The original Target instances '\
                               'predicted sentiment should not be set as the '\
                               'TargetCollection should have copied them.'):
            target_example_int_1['predicted']

        pred_sents = ['neg', 'neu', 'pos']
        target_col_str.add_pred_sentiment(pred_sents)
        self.assertEqual('neu', target_col_str['3']['predicted'], msg='Predicted '\
                         'sentiment not set correctly for `id` 3 should be `neu` '\
                         'and not {}'.format(target_col_str))
        pred_sents = [1, 0, -1]
        target_col_str.add_pred_sentiment(pred_sents, mapper={1 : 'pos', 0 : 'neu',
                                                         -1 : 'neg'})
        self.assertEqual('neu', target_col_str['3']['predicted'], msg='Predicted '\
                         'sentiment not set correctly for `id` 3 should be `neu` '\
                         'and not {} using the mapper'.format(target_col_str))

        with self.assertRaises(KeyError, msg='The original Target instances '\
                               'predicted sentiment should not be set as the '\
                               'TargetCollection should have copied them.'):
            target_example_1['predicted']

        with self.assertRaises(TypeError, msg='Should only accept list type '\
                               'not tuples'):
            target_col_int.add_pred_sentiment((2, 5, 4))
        with self.assertRaises(ValueError, msg='Should accept lists that are '\
                               ' the same size as the TargetCollection'):
            target_col_int.add_pred_sentiment([1, 2, 3, 4])
        with self.assertRaises(TypeError, msg='When using the mapper should not '\
                               'accept Strings when expecting Integer'):
            target_col_int.add_pred_sentiment([1, 1, 0], mapper={1 : 'pos', 0: 'neg'})

    def test_target_coll_subset_by_sent(self):
        '''
        Test the subset_by_sentiment function of TargetCollection
        '''

        target_example_0 = Target([(3, 5), (6, 8)], '1', 'Iphone',
                                  'text with Iphone', 'pos', sentence_id='4')
        target_example_1 = Target([(1, 5)], '3', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='4')
        target_example_2 = Target([(1, 2)], '2', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='5')
        target_example_3 = Target([(1, 2)], '4', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='5')
        target_example_4 = Target([(1, 2)], '5', 'Iphone',
                                  'text with Iphone', 'pos', sentence_id='6')
        target_example_5 = Target([(1, 2)], '6', 'Iphone',
                                  'text with Iphone', 'neu', sentence_id='6')
        target_example_6 = Target([(1, 2)], '7', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='6')
        target_example_7 = Target([(1, 2)], '8', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='7')
        target_example_8 = Target([(1, 2)], '9', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='8')
        target_example_9 = Target([(1, 2)], '10', 'Iphone',
                                  'text with Iphone', 'neg', sentence_id='8')
        target_example_10 = Target([(1, 2)], '11', 'Iphone',
                                  'text with Iphone', 'pos', sentence_id='8')
        all_targets = [target_example_0, target_example_1, target_example_2,
                       target_example_3, target_example_4, target_example_5,
                       target_example_6, target_example_7, target_example_8,
                       target_example_9, target_example_10]
        target_col = TargetCollection(all_targets)

        # Test for 2 unique sentiments per sentence
        test_col = target_col.subset_by_sentiment(2)
        valid_col = TargetCollection([target_example_0, target_example_1, 
                                      target_example_8, target_example_9, 
                                      target_example_10])
        self.assertEqual(valid_col, test_col, msg='Should only return these {}'\
                         ' but has returned this {}'.format(valid_col, test_col))
        # Test for 1 unique sentiments per sentence
        test_col = target_col.subset_by_sentiment(1)
        valid_col = TargetCollection([target_example_7, target_example_2, 
                                      target_example_3])
        self.assertEqual(valid_col, test_col, msg='Should only return these {}'\
                         ' but has returned this {}'.format(valid_col, test_col))
        # Test for 3 unique sentiments per sentence
        test_col = target_col.subset_by_sentiment(3)
        valid_col = TargetCollection([target_example_4, target_example_5,
                                      target_example_6])
        self.assertEqual(valid_col, test_col, msg='Should only return these {}'\
                         ' but has returned this {}'.format(valid_col, test_col))




