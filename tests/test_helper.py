'''
Unit test suite for :py:mod:`tdparse.helper` module.
'''
import os

from unittest import TestCase


from tdparse.helper import read_config
from tdparse.helper import package_dir

class TestHelper(TestCase):
    '''
    Contains the following functions:
    1. :py:func:`tdparse.helper.test_read_config`
    2. :py:func:`tdparse.helper.test_package_dir`
    '''

    def test_read_config(self):
        '''
        Tests :py:func:`tdparse.helper.read_config`
        '''

        # Check if it can handle nested values - which should be converted to
        # a dictionary
        self.assertIsInstance(read_config('word2vec_files'), dict)

        self.assertEqual(read_config('test_data')['dong_data'],
                         './tests/test_data/dong_test_data.txt')
        with self.assertRaises(ValueError,
                               msg='nothing here should not be in the config.yaml'):
            read_config('nothing here')
        test_config_name = 'doesnotexist'
        with self.assertRaises(FileNotFoundError,
                               msg='there should be no file named {}'\
                                   .format(test_config_name)):
            read_config('test_data', config_file_name=test_config_name)



    def test_package_dir(self):
        '''
        Tests :py:func:`tdparse.helper.package_dir`
        '''

        self.assertIsInstance(package_dir(), str, msg='The return should be a String')
        self.assertEqual(package_dir().split(os.sep)[-1], 'tdparse',
                         msg='The last folder should be tdparse')
