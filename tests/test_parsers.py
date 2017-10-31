'''
Unit test suite for the :py:mod:`tdparse.helper` module.
'''
from unittest import TestCase


from tdparse.parsers import dong
from tdparse.helper import read_config

class TestParsers(TestCase):
    '''
    Contains the following functions:
    1. :py:func:`tdparse.parsers.test_dong`
    '''

    def test_dong(self):
        '''
        Tests :py:func:`tdparse.parsers.dong`
        '''

        def check_results(expected_results, test_results):
            '''
            Given the expected results and the results from the function beign
            tested it will test that they are both equal. It will return nothing
            but will test if they are correct else it fails the tests.

            :param expected_results: A list of dictionaries containing expected
            values
            :param test_results: A list of dictionaries containing results from
            the function that is being tested
            :type expected_results: list
            :type test_results: list
            :returns: Nothing but checks if the results are to be expected
            :rtype: None
            '''

            for index, expected_result in enumerate(expected_results):
                test_result = test_results[index]
                for key, expected_value in expected_result.items():
                    test_value = test_result[key]
                    self.assertIsInstance(expected_value, type(test_value),
                                          msg='The expected value : {} is not of the '\
                                          'same type as the tested value : {}'\
                                          .format(type(expected_value), type(test_value)))
                    if key == 'spans':
                        test_value = sorted(test_value, key=lambda x: x[0])
                        expected_value = sorted(expected_value, key=lambda x: x[0])

                    self.assertEqual(expected_value, test_value,
                                     msg='Expected {} returned {}'.format(expected_value,
                                                                          test_value))


        test_file_path = 'anything'
        with self.assertRaises(FileNotFoundError, msg='there should be no file named {}'\
                               .format(test_file_path)):
            dong(test_file_path)

        test_file_path = './tests/test_data/dong_test_data.txt'
        expected_results = [{'id':0,
                             'sentiment':-1,
                             'text':'This is a fake news article that is to represent a Tweet!!!!',
                             'target':'news article',
                             'spans':[[15, 27]]},
                            {'id':1,
                             'sentiment':1,
                             'text':'I had a great day however I did not get much work done',
                             'target':'day',
                             'spans':[[14, 17]]},
                            {'id':2,
                             'sentiment':0,
                             'text':'I cycled in today and it was ok as it was not raining.',
                             'target':'cycled',
                             'spans':[[2, 8]]}]
        check_results(expected_results, dong(test_file_path))

        bad_sent_path = read_config('test_data')['dong_bad_sent_data']
        with self.assertRaises(ValueError, msg='It should not accept sentiment '\
                               'values that are not 1, 0, or -1'):
            dong(bad_sent_path)

        # Ensure that it can handle the same target with multiple spans
        test_multiple_path = read_config('test_data')['dong_multiple_offsets_data']

        multi_expected = [{'id':0,
                           'sentiment':-1,
                           'text':'This is a fake news article that is to represent a '\
                           'Tweet!!!! and it was an awful News Article I think.',
                           'target':'news article',
                           'spans':[[15, 27], [81, 93]]},
                          {'id':1,
                           'sentiment':1,
                           'text':'I had a great Day however I did not get much '\
                           'work done in the day',
                           'target':'day',
                           'spans':[[14, 17], [62, 65]]}]
        check_results(multi_expected, dong(test_multiple_path))
