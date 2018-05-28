'''
Unit test suite for :py:mod:`bella.helper` module.
'''
import os

from unittest import TestCase


from bella.helper import read_config
from bella.helper import package_dir
from bella.helper import full_path

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
        Tests :py:func:`bella.helper.package_dir`
        '''

        self.assertIsInstance(package_dir(), str, msg='The return should be a String')
        self.assertEqual(package_dir().split(os.sep)[-1], 'bella',
                         msg='The last folder should be bella')

    def test_full_path(self):
        '''
        Tests :py:func:`bella.helper.full_path`
        '''

        path = '../aspect datasets'
        example = full_path(path)
        self.assertIsInstance(example, str, msg='The return should always be a '\
                              'String')
        example_folders = example.split(os.sep)
        self.assertEqual('aspect datasets', example_folders[-1],
                         msg='The last directory should be `aspect datasets` '\
                         'and not {}'.format(example_folders[-1]))
        all_dirs = os.path.abspath(__file__).split(os.sep)
        parent_example = all_dirs[-4]
        self.assertEqual(parent_example, example_folders[-2], msg='Parent of '\
                         'the example path should be {} not {}'\
                         .format(parent_example, example_folders[-2]))
