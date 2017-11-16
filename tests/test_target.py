'''
Unit test suite for the :py:mod:`tdparse.helper` module.
'''
from unittest import TestCase

from tdparse.helper import read_config
from tdparse.word_vectors import PreTrained
from tdparse.word_vectors import GensimVectors
from tdparse.models.target import TargetInd
from tdparse.models.target import TargetDepC

class TestTarget(TestCase):
    '''
    Contains the following functions:
    '''

    def test_target_get_params_dict(self):
        '''
        Tests:

        1. :py:func:`tdparse.models.target.TargetInd.get_params_dict`
        2. :py:func:`tdparse.models.target.TargetDepC.get_params_dict`
        '''

        def key_value_test(valid_k_v, test_k_v, k_error, v_error):
            for key, value in valid_k_v.items():
                contains_key = key in test_k_v
                self.assertEqual(True, contains_key, msg=k_error)
                if contains_key:
                    self.assertEqual(value, test_k_v[key], msg=v_error)

        sswe_path = read_config('sswe_files')['vo_zhang']
        sswe_model = PreTrained(sswe_path, name='sswe')
        vo_zhang_path = read_config('word2vec_files')['vo_zhang']
        vo_zhang = GensimVectors(vo_zhang_path, None, model='word2vec')

        # Testing TargetInd get_params_dict
        test_model = TargetInd()
        test_params = test_model.get_params_dict([sswe_model])
        valid_params = {'word_vectors__vector' : [sswe_model]}
        self.assertEqual(valid_params, test_params, msg='Failed param dict')

        grid_test_params = test_model.get_params_dict([[sswe_model]])
        valid_grid_params = [{'word_vectors__vector' : [[sswe_model]]}]
        self.assertEqual(valid_grid_params, grid_test_params,
                         msg='Failed grid search param dict')

        with self.assertRaises(TypeError, msg='Failed to assert the that input '\
                               'has to be a list'):
            test_model.get_params_dict('anything')
        with self.assertRaises(TypeError, msg='Failed to assert the type of '\
                               'word vectors for normal parameter input'):
            test_model.get_params_dict(['anything'])
        with self.assertRaises(TypeError, msg='Failed to assert the type of '\
                               'word vectors for grid searching'):
            test_model.get_params_dict([['anything']])

        # Testing TargetDepC get_params_dict
        test_model = TargetDepC()
        test_params = test_model.get_params_dict([sswe_model])
        valid_params = {'union__left__word_vectors__vector' : [sswe_model],
                        'union__right__word_vectors__vector' : [sswe_model],
                        'union__target__word_vectors__vector' : [sswe_model]}
        key_error = 'Not getting the param names '\
                    'correctly check `_get_word_vector_names`'
        value_error = 'param word vector not set properly'
        key_value_test(valid_params, test_params, key_error, value_error)

        test_params = test_model.get_params_dict([sswe_model, vo_zhang])
        valid_params = {'union__left__word_vectors__vector' : [sswe_model, vo_zhang],
                        'union__right__word_vectors__vector' : [sswe_model, vo_zhang],
                        'union__target__word_vectors__vector' : [sswe_model, vo_zhang]}
        key_error = 'Not getting the param names '\
                    'correctly check `_get_word_vector_names`'
        value_error = 'Cannot handle multiple word vectors'
        key_value_test(valid_params, test_params, key_error, value_error)
        # Testing the grid search functions
        test_params = test_model.get_params_dict([[sswe_model], [vo_zhang]])
        valid_params = [{'union__left__word_vectors__vector' : [[sswe_model]],
                         'union__right__word_vectors__vector' : [[sswe_model]],
                         'union__target__word_vectors__vector' : [[sswe_model]]},
                        {'union__left__word_vectors__vector' : [[vo_zhang]],
                         'union__right__word_vectors__vector' : [[vo_zhang]],
                         'union__target__word_vectors__vector' : [[vo_zhang]]}]
        for index, valid_param in enumerate(valid_params):
            test_param = test_params[index]
            key_value_test(valid_param, test_param, key_error,
                           'Cannot get parameters for grid search')
