'''
Unit test suite for :py:mod:`bella.helper` module.
'''
from pathlib import Path

from unittest import TestCase

from bella.helper import read_config

CONFIG_FP = Path(__file__).parent.joinpath('..', 'config.yaml')


class TestHelper(TestCase):
    '''
    Contains the following functions:
    1. :py:func:`bella.helper.test_read_config`
    2. :py:func:`bella.helper.test_package_dir`
    '''

    def test_read_config(self):
        '''
        Tests :py:func:`bella.helper.read_config`
        '''

        # Check if it can handle nested values - which should be converted to
        # a dictionary
        self.assertIsInstance(read_config('word2vec_files', CONFIG_FP), dict)

        self.assertEqual(read_config('test_data', CONFIG_FP)['dong_data'],
                         './tests/test_data/dong_test_data.txt')
        with self.assertRaises(ValueError,
                               msg='nothing here should not be in the '
                                   'config.yaml'):
            read_config('nothing here', CONFIG_FP)
        test_config_name = Path('./doesnotexist')
        with self.assertRaises(FileNotFoundError,
                               msg='there should be no file named '
                                   f'{test_config_name}'):
            read_config('test_data', test_config_name)
