'''
Unit test suite for the :py:mod:`bella.models.tdparse` module.
'''
from unittest import TestCase

from bella.helper import read_config
from bella.word_vectors import PreTrained
from bella.word_vectors import GensimVectors
from bella import lexicons
from bella.dependency_parsers import tweebo
from bella.tokenisers import whitespace
from bella.tokenisers import ark_twokenize
from bella.models.tdparse import TDParseMinus
from bella.models.tdparse import TDParse
from bella.models.tdparse import TDParsePlus

class TestTarget(TestCase):
    '''
    Contains the following functions:
    '''

    @staticmethod
    def _get_name(test_obj, valid_obj):
        '''
        Given two unknown objects returns the name of the objects or the
        values of the objects if the objects are basic python type (str, int or
        tuple).
        Returns a tuple of 2 comprabale objects.

        :param test_obj: A python object ranging from functions to \
        instances of classes.
        :param valid_obj: A python object ranging from functions to \
        instances of classes.
        :type test_obj: Python object
        :type valid_obj: Python object
        :returns: A tuple of two Python objects that can be compared. e.g. str \
        or int.
        :rtype: tuple
        '''
        if hasattr(test_obj, '__name__'):
            if hasattr(test_obj, '__name__'):
                return (test_obj.__name__, valid_obj.__name__)
            else:
                raise ValueError('test_obj has attr __name__ but valid '\
                                 'does not. Test {} Valid {}'\
                                 .format(test_obj, valid_obj))
        elif hasattr(test_obj, 'name'):
            if hasattr(valid_obj, 'name'):
                return (test_obj.name, valid_obj.name)
            else:
                raise ValueError('test_obj has attr name but valid '\
                                 'does not. Test {} Valid {}'\
                                 .format(test_obj, valid_obj))
        elif isinstance(test_obj, (int, str, tuple)):
            if isinstance(test_obj, type(valid_obj)):
                return (test_obj, valid_obj)
            else:
                raise ValueError('test_obj is not of the same type as the valid '\
                                 '{} {}'.format(type(test_obj), type(valid_obj)))

        else:
            raise ValueError('Cannot determine the name of value or test'\
                             ' object please insert a rule into `get_name`'\
                             ' function to resolve this problem. Valid {} '\
                             'test {}'.format(valid_obj, test_obj))

    def _compare_types_list(self, valid_list, test_list):
        '''
        Given a list of values or a list of a list of values it compares
        the two lists to see if they are the same by comparing the values
        types. Returns True if all values in the test_list are the same as
        the value_list else False.

        It has to compare types as the functions copy values therefore cannot
        use the equalls functions however the type comparison is not ideal
        and needs changing.

        :param valid_list: list of (of a list) of values
        :param test_list: list of (of a list) of values
        :type valid_list: list
        :type test_list: list
        :returns: True if test values have the same as the valid values.
        :rtype: bool
        '''

        if not isinstance(valid_list, list) or not isinstance(test_list, list):
            raise ValueError('Valid and test lists has to be of type list '\
                             'and not {} {}'\
                             .format(type(valid_list), type(test_list)))
        if len(valid_list) != len(test_list):
            return False
        for index, valid_obj in enumerate(valid_list):
            test_obj = test_list[index]
            if isinstance(valid_obj, list):
                for inner_index, valid_list_obj in enumerate(valid_obj):
                    t_obj_name, v_obj_name = self._get_name(test_obj[inner_index],
                                                            valid_list_obj)
                    if t_obj_name != v_obj_name:
                        return False
            else:
                t_obj_name, v_obj_name = self._get_name(test_obj, valid_obj)
                if t_obj_name != v_obj_name:
                    return False
        return True

    def _compare_types(self, valid, test):
        '''
        Given a value or a list of values it compares the two to check if they
        are the same values. The values can be any python object and to determine
        they are the same it tests if they have the same name. This checking needs
        improving as not all objects have a name therefore the rule base may need
        expanding in the future.

        :param valid: Any python object ranging from function to class instance
        :param test: Any python object ranging from function to class instance
        :type valid: Python object
        :type test: Python object
        :returns: True if test is the same as value
        :rtype: bool
        '''

        if isinstance(valid, list):
            for inner_index, valid_list_obj in enumerate(valid):
                t_obj_name, v_obj_name = self._get_name(test[inner_index],
                                                        valid_list_obj)
                if t_obj_name != v_obj_name:
                    return False
        else:
            t_name, v_name = self._get_name(test, valid)
            if t_name != v_name:
                return False
        return True

    def _key_value_test(self, valid_k_v, test_k_v, grid_test=True):
        '''
        Checks that the two input dictionaries are equal. Returns None.

        NOTE: That if the dictionaries are more than one layer deep this
        method will not work. One layer deep means it can not check if
        a dict within dict is equal between the two input dictionaries.

        :param valid_k_v: A dictionary that contains the correct values
        :param test_k_v: A dictionary that contains the values that are to \
        be compared with the valid_k_v dictionary
        :type valid_k_v: dict
        :type test_k_v: dict
        :returns: Nothing but will fail the tests.
        :rtype: None
        '''

        for key, value in valid_k_v.items():
            contains_key = key in test_k_v
            k_error = 'The test dict does not contain the following key {}'\
                      .format(key)
            self.assertEqual(True, contains_key, msg=k_error)
            if contains_key:
                v_error = 'The test dict has value: {} for {} where as valid'\
                          ' has {}'.format(test_k_v[key], key, value)
                the_same = False
                if grid_test:
                    the_same = self._compare_types_list(value, test_k_v[key])
                else:
                    the_same = self._compare_types(value, test_k_v[key])
                self.assertEqual(True, the_same, msg=v_error)


    def test_tdparse_cv_params(self):
        '''
        Tests:

        1. `bella.models.target.TargetInd.get_cv_params`
        2. `bella.models.target.TargetDepC.get_params`
        '''

        def grid_key_value_test(test_dict_list, valid_dict_list):
            '''
            Given two list of dictionaries as input it compares each dictionary
            in the list to ensure that both dicts are the same using the
            :py:func:`key_value_test` function. Returns None

            :param test_dict_list: A list of dicts to compared to the valid \
            dict list.
            :param valid_dict_list: A list of dicts.
            :type test_dict_list: list
            :type valid_dict_list: list
            :returns: None but will fail the tests if the two lists are not equal.
            :rtype: None
            '''

            if len(test_dict_list) != len(valid_dict_list):
                raise ValueError('test and valid dict lists should be the same '\
                                 'size. Test size {} valid {}'\
                                 .format(len(test_dict_list), len(valid_dict_list)))
            for index, valid_dict in enumerate(valid_dict_list):
                test_dict = test_dict_list[index]
                self._key_value_test(valid_dict, test_dict)


        sswe_path = read_config('sswe_files')['vo_zhang']
        sswe_model = PreTrained(sswe_path, name='sswe')
        vo_zhang_path = read_config('word2vec_files')['vo_zhang']
        vo_zhang = GensimVectors(vo_zhang_path, None, model='word2vec')

        # Testing TDParse_minus class
        #
        # Simple example
        test_model = TDParseMinus()
        valid_example = [{'word_vectors__vectors' : [[sswe_model]],
                          'dependency_context__parser' : [tweebo]},
                         {'word_vectors__vectors' : [[vo_zhang]],
                          'dependency_context__parser' : [tweebo]}]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model],
                                                              [vo_zhang]],
                                                parsers=[tweebo])
        grid_key_value_test(test_example, valid_example)
        # More complex has to hadle more than one parameter type
        valid_example = [{'word_vectors__vectors' : [[sswe_model]],
                          'tokens__tokeniser' : [whitespace],
                          'dependency_context__parser' : [tweebo]},
                         {'word_vectors__vectors' : [[vo_zhang]],
                          'tokens__tokeniser' : [whitespace],
                          'dependency_context__parser' : [tweebo]}]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace],
                                                parsers=[tweebo])
        grid_key_value_test(test_example, valid_example)
        # More complex same number of parameter types but now has to expand to
        # get all the unique combinations
        valid_example = [{'word_vectors__vectors' : [[sswe_model, vo_zhang]],
                          'tokens__tokeniser' : [whitespace],
                          'dependency_context__parser' : [tweebo]},
                         {'word_vectors__vectors' : [[vo_zhang]],
                          'tokens__tokeniser' : [whitespace],
                          'dependency_context__parser' : [tweebo]},
                         {'word_vectors__vectors' : [[sswe_model, vo_zhang]],
                          'tokens__tokeniser' : [ark_twokenize],
                          'dependency_context__parser' : [tweebo]},
                         {'word_vectors__vectors' : [[vo_zhang]],
                          'tokens__tokeniser' : [ark_twokenize],
                          'dependency_context__parser' : [tweebo]}]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model, vo_zhang],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace,
                                                            ark_twokenize],
                                                parsers=[tweebo])
        grid_key_value_test(test_example, valid_example)
        # Testing if child relations work
        test_model = TDParseMinus(child_relations=True)
        valid_example = [{'word_vectors__vectors' : [[sswe_model, vo_zhang]],
                          'tokens__tokeniser' : [whitespace],
                          'dependency_context__parser' : [tweebo],
                          'dependency_context__rel_depth' : [(1, 3), (1, 1)]},
                         {'word_vectors__vectors' : [[vo_zhang]],
                          'tokens__tokeniser' : [whitespace],
                          'dependency_context__parser' : [tweebo],
                          'dependency_context__rel_depth' : [(1, 3), (1, 1)]},
                         {'word_vectors__vectors' : [[sswe_model, vo_zhang]],
                          'tokens__tokeniser' : [ark_twokenize],
                          'dependency_context__parser' : [tweebo],
                          'dependency_context__rel_depth' : [(1, 3), (1, 1)]},
                         {'word_vectors__vectors' : [[vo_zhang]],
                          'tokens__tokeniser' : [ark_twokenize],
                          'dependency_context__parser' : [tweebo],
                          'dependency_context__rel_depth' : [(1, 3), (1, 1)]}]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model, vo_zhang],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace,
                                                            ark_twokenize],
                                                parsers=[tweebo],
                                                rel_depths=[(1, 3), (1, 1)])
        grid_key_value_test(test_example, valid_example)
        # Testing TDParse
        test_model = TDParse()
        valid_example = [{'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__dependency__context__parser' : [tweebo]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__dependency__context__parser' : [tweebo]},
                         {'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__dependency__context__parser' : [tweebo]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__dependency__context__parser' : [tweebo]}
                        ]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model, vo_zhang],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace,
                                                            ark_twokenize],
                                                parsers=[tweebo])
        grid_key_value_test(test_example, valid_example)
        # Testing if child relations work
        test_model = TDParse(child_relations=True)
        valid_example = [{'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' :
                          [(1, 2), (1, 1)]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' :
                          [(1, 2), (1, 1)]},
                         {'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' :
                          [(1, 2), (1, 1)]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' :
                          [(1, 2), (1, 1)]}
                        ]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model, vo_zhang],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace,
                                                            ark_twokenize],
                                                parsers=[tweebo],
                                                rel_depths=[(1, 2), (1, 1)])
        grid_key_value_test(test_example, valid_example)
        # Testing TDParsePlus
        test_lexicon = lexicons.Lexicon(name='test', lexicon=[('example', 'pos')])
        test_model = TDParsePlus()
        valid_example = [{'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__left_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__left_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo]},
                         {'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__tokens__tokeniser' : [ark_twokenize],
                          'union__left_s__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__left_s__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo]}
                        ]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model, vo_zhang],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace,
                                                            ark_twokenize],
                                                parsers=[tweebo],
                                                senti_lexicons=[test_lexicon])
        grid_key_value_test(test_example, valid_example)
        # Testing TDParsePlus with child_relations
        test_lexicon = lexicons.Lexicon(name='test', lexicon=[('example', 'pos')])
        test_model = TDParsePlus(child_relations=True)
        valid_example = [{'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__left_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' : [(1, 3)]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [whitespace],
                          'union__left__tokens__tokeniser' : [whitespace],
                          'union__right__tokens__tokeniser' : [whitespace],
                          'union__target__tokens__tokeniser' : [whitespace],
                          'union__left_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__tokens__tokeniser' : [whitespace],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' : [(1, 3)]},
                         {'union__dependency__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[sswe_model, vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__tokens__tokeniser' : [ark_twokenize],
                          'union__left_s__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' : [(1, 3)]},
                         {'union__dependency__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__target__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__left_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__right_s__word_vectors__vectors' :
                          [[vo_zhang]],
                          'union__dependency__tokens__tokeniser' : [ark_twokenize],
                          'union__left__tokens__tokeniser' : [ark_twokenize],
                          'union__right__tokens__tokeniser' : [ark_twokenize],
                          'union__target__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__tokens__tokeniser' : [ark_twokenize],
                          'union__left_s__tokens__tokeniser' : [ark_twokenize],
                          'union__right_s__filter__lexicon' : [test_lexicon],
                          'union__left_s__filter__lexicon' : [test_lexicon],
                          'union__dependency__context__parser' : [tweebo],
                          'union__dependency__context__rel_depth' : [(1, 3)]}
                        ]
        test_example = test_model.get_cv_params(word_vectors=[[sswe_model, vo_zhang],
                                                              [vo_zhang]],
                                                tokenisers=[whitespace,
                                                            ark_twokenize],
                                                parsers=[tweebo],
                                                senti_lexicons=[test_lexicon],
                                                rel_depths=[(1, 3)])
        grid_key_value_test(test_example, valid_example)

    def test_target_params(self):
        '''
        Tests:

        1. `bella.models.target.TargetInd.get_params`
        2. `bella.models.target.TargetDepC.get_params`
        '''

        sswe_path = read_config('sswe_files')['vo_zhang']
        sswe_model = PreTrained(sswe_path, name='sswe')
        vo_zhang_path = read_config('word2vec_files')['vo_zhang']
        vo_zhang = GensimVectors(vo_zhang_path, None, model='word2vec')

        # Testing TDParseMinus class
        test_model = TDParseMinus()
        valid_example = {'word_vectors__vectors' : [sswe_model, vo_zhang],
                         'tokens__tokeniser' : whitespace,
                         'dependency_context__parser' : tweebo}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             parser=tweebo)
        self._key_value_test(valid_example, test_example, grid_test=False)

        # Testing on arguments that do not call a function to get the arguments
        # names
        valid_example = {'word_vectors__vectors' : [sswe_model, vo_zhang],
                         'tokens__tokeniser' : whitespace,
                         'svm__random_state' : 42,
                         'dependency_context__parser' : tweebo}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             random_state=42,
                                             parser=tweebo)
        self._key_value_test(valid_example, test_example, grid_test=False)
        # Testing child relations
        test_model = TDParseMinus(child_relations=True)
        valid_example = {'word_vectors__vectors' : [sswe_model, vo_zhang],
                         'tokens__tokeniser' : whitespace,
                         'svm__random_state' : 42,
                         'dependency_context__parser' : tweebo,
                         'dependency_context__rel_depth' : (1, -1)}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             random_state=42,
                                             parser=tweebo,
                                             rel_depth=(1, -1))
        self._key_value_test(valid_example, test_example, grid_test=False)

        # Testing TDParse
        test_model = TDParse()
        valid_example = {'union__dependency__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__right__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__left__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__target__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__dependency__tokens__tokeniser' : whitespace,
                         'union__left__tokens__tokeniser' : whitespace,
                         'union__right__tokens__tokeniser' : whitespace,
                         'union__target__tokens__tokeniser' : whitespace,
                         'svm__random_state' : 42,
                         'union__dependency__context__parser' : tweebo}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             random_state=42,
                                             parser=tweebo)
        self._key_value_test(valid_example, test_example, grid_test=False)
        # Testing child relations
        test_model = TDParse(child_relations=True)
        valid_example = {'union__dependency__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__right__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__left__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__target__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__dependency__tokens__tokeniser' : whitespace,
                         'union__left__tokens__tokeniser' : whitespace,
                         'union__right__tokens__tokeniser' : whitespace,
                         'union__target__tokens__tokeniser' : whitespace,
                         'svm__random_state' : 42,
                         'union__dependency__context__parser' : tweebo,
                         'union__dependency__context__rel_depth' : (1, -1)}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             random_state=42,
                                             parser=tweebo,
                                             rel_depth=(1, -1))
        self._key_value_test(valid_example, test_example, grid_test=False)
        # Testing TDParsePlus
        test_model = TDParsePlus()
        test_lexicon = lexicons.Lexicon(name='test', lexicon=[('example', 'pos')])
        valid_example = {'union__dependency__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__right__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__left__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__target__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__left_s__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__right_s__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__dependency__tokens__tokeniser' : whitespace,
                         'union__left__tokens__tokeniser' : whitespace,
                         'union__right__tokens__tokeniser' : whitespace,
                         'union__target__tokens__tokeniser' : whitespace,
                         'union__right_s__tokens__tokeniser' : whitespace,
                         'union__left_s__tokens__tokeniser' : whitespace,
                         'union__left_s__filter__lexicon' : test_lexicon,
                         'union__right_s__filter__lexicon' : test_lexicon,
                         'svm__random_state' : 42,
                         'union__dependency__context__parser' : tweebo}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             random_state=42,
                                             parser=tweebo,
                                             senti_lexicon=test_lexicon)
        self._key_value_test(valid_example, test_example, grid_test=False)
        # Testing TDParsePlus with child_relations
        test_model = TDParsePlus(child_relations=True)
        test_lexicon = lexicons.Lexicon(name='test', lexicon=[('example', 'pos')])
        valid_example = {'union__dependency__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__right__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__left__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__target__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__left_s__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__right_s__word_vectors__vectors' :
                         [sswe_model, vo_zhang],
                         'union__dependency__tokens__tokeniser' : whitespace,
                         'union__left__tokens__tokeniser' : whitespace,
                         'union__right__tokens__tokeniser' : whitespace,
                         'union__target__tokens__tokeniser' : whitespace,
                         'union__right_s__tokens__tokeniser' : whitespace,
                         'union__left_s__tokens__tokeniser' : whitespace,
                         'union__left_s__filter__lexicon' : test_lexicon,
                         'union__right_s__filter__lexicon' : test_lexicon,
                         'svm__random_state' : 42,
                         'union__dependency__context__parser' : tweebo,
                         'union__dependency__context__rel_depth' : (1, -2)}
        test_example = test_model.get_params(word_vector=[sswe_model, vo_zhang],
                                             tokeniser=whitespace,
                                             random_state=42,
                                             parser=tweebo,
                                             senti_lexicon=test_lexicon,
                                             rel_depth=(1, -2))
        self._key_value_test(valid_example, test_example, grid_test=False)
