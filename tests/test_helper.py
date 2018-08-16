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
    '''

    def test_read_config(self):
        '''
        Tests :py:func:`bella.helper.read_config`
        '''

        dong_test_fp = 'tests/test_data/dong_test_data.txt'
        assert dong_test_fp in read_config('dong_data_test', CONFIG_FP)
        with self.assertRaises(ValueError,
                               msg='nothing here should not be in the '
                                   'config.yaml'):
            read_config('nothing here', CONFIG_FP)
        test_config_name = Path('./doesnotexist')
        with self.assertRaises(FileNotFoundError,
                               msg='there should be no file named '
                                   f'{test_config_name}'):
            read_config('test_data', test_config_name)
