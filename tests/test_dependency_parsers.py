'''
Unit test suite for the :py:mod:`tdparse.dependency_parsers` module.
'''
import os
import shutil
from unittest import TestCase

import pytest

from tdparse.dependency_parsers import tweebo_install
from tdparse.dependency_parsers import tweebo

class TestDependencyParsers(TestCase):
    '''
    Contains the following functions:
    '''

    #@pytest.mark.skip(reason="Takes a long time to test only add on large tests")
    def test_tweebo_install(self):
        '''
        Test for :py:func:`tdparse.dependency_parsers.tweebo_install`. Tests that
        it downloads the files correctly. Hard to test the functionality without
        testing the parser which is in the
        :py:func:`tdparse.tests.test_dependency_parsers.test_tweebo`
        '''

        def add(num1, num2):
            '''
            Adds two numbers together, used as a test function.
            '''
            return num1 + num2
        current_dir = os.path.abspath(os.path.dirname(__file__))
        tweebo_dir = os.path.abspath(os.path.join(current_dir, os.pardir, 'tools',
                                                  'TweeboParser'))
        model_dir = os.path.join(tweebo_dir, 'pretrained_models')
        model_file = os.path.join(tweebo_dir, 'pretrained_models.tar.gz')
        if os.path.isfile(model_file):
            os.remove(model_file)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        install_and_add = tweebo_install(add)
        self.assertEqual(True, os.path.isfile(model_file), msg='Did not download '\
                         'the tweebo models, file path {} Tweebo install error '\
                         'check the install.sh within tools/TweeboParser '\
                         .format(model_file))
        self.assertEqual(True, os.path.isdir(model_dir), msg='Did not unpack the '\
                         'tweebo models to the following dir {} Tweebo install error'\
                         ' check the install.sh within tools/TweeboParser '\
                         .format(model_dir))
        self.assertEqual(10, install_and_add(8, 2), msg='The tweebo install '\
                         'function does not wrap function properly')

    def test_tweebo(self):
        pass
