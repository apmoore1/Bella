'''
Unit tests for the :py:mod:`tdparse.helper` module.
'''
import os

from unittest import TestCase


from tdparse.helper import read_config
from tdparse.helper import package_dir

class TestHelper(TestCase):
    '''
    Contains the following functions:
    :py:func:`tdparse.helper.read_config`
    '''
    def test_read_config(self):
        '''
        Tests :py:func:`tdparse.helper.read_config`
        '''

        self.assertEqual(read_config('unit_test_dong_data'),
                         './tests/data/dong_sent.txt')
        with self.assertRaises(ValueError,
                               msg='nothing here should not be in the config.yaml'):
            read_config('nothing here')
        test_config_name = 'doesnotexist'
        with self.assertRaises(FileNotFoundError,
                               msg='there should be no file named {}'\
                                   .format(test_config_name)):
            read_config('unit_test_dong_data', config_file_name=test_config_name)

    def test_package_dir(self):
        '''
        Tests :py:func:`tdparse.helper.package_dir`
        '''

        self.assertIsInstance(package_dir(), str, msg='The return should be a String')
        self.assertEqual(package_dir().split(os.sep)[-1], 'tdparse',
                         msg='The last folder should be tdparse')
