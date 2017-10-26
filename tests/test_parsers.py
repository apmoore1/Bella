'''
Unit test suite for the :py:mod:`tdparse.helper` module.
'''
from unittest import TestCase


from tdparse.parsers import dong
from tdparse.helper import read_config

class TestParsers(TestCase):
    '''
    Contains the following functions:
    :py:func:`tdparse.parsers.test_dong`
    :py:func:`tdparse.helper.test_package_dir`
    '''

    def test_dong(self):
        '''
        Tests :py:func:`tdparse.parsers.dong`
        '''

        test_file_path = 'anything'
        with self.assertRaises(FileNotFoundError, msg='there should be no file named {}'\
                               .format(test_file_path)):
            dong(test_file_path)

        test_file_path = './tests/test_data/dong_test_data.txt'
        expected_results = [{'id':'dong_train_0',
                             'sentiment':-1,
                             'text':'This is a fake news article that is to represent a Tweet!!!!',
                             'target':'news article',
                             'span':(15, 26)},
                            {'id':'dong_train_1',
                             'sentiment':1,
                             'text':'I had a great day however I did not get much work done',
                             'target':'day',
                             'span':(14, 17)},
                            {'id':'dong_train_2',
                             'sentiment':0,
                             'text':'I cycled in today and it was ok as it was not raining.',
                             'target':'cycled',
                             'span':(2, 7)}]
        # Ensures it extracts the correct dictionary structure
        test_results = dong(test_file_path)
        for index, test_result in test_results:
            expected_result = expected_results[index]
            for key, test_value in test_result.items():
                expected_value = expected_result[key]
                self.assertEqual(expected_value, test_value,
                                 msg='Expected {} returned {}'.format(expected_value,
                                                                      test_value))

        # Ensure the Sentiment is on a 1, 0, -1 scale in the train and test data
        # and it can be read from the config file
        bad_sent_path = read_config('unit_test_dong_bad_sent_data')
        with self.assertRaises(ValueError, msg='It should not accept sentiment '\
                               'values that are not 1, 0, or -1'):
            dong(bad_sent_path)

        # Ensure that it can handle the same target with multiple spans
        test_multiple_path = read_config('unit_test_dong_bad_multiple_data')
