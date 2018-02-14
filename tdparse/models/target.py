'''
Contains classes of models that can be found in `Vo and Zhang 2015 paper \
<https://www.ijcai.org/Proceedings/15/Papers/194.pdf>`_.

Classes:

1. :py:class:`tdparse.models.target.TargetInd` - Target indepdent model
'''
from collections import defaultdict
import copy
import types

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tdparse.tokenisers import ark_twokenize
from tdparse.neural_pooling import matrix_max, matrix_min, matrix_avg,\
matrix_median, matrix_prod, matrix_std

from tdparse.scikit_features.context import Context
from tdparse.scikit_features.tokeniser import ContextTokeniser
from tdparse.scikit_features.word_vector import ContextWordVectors
from tdparse.scikit_features.lexicon_filter import LexiconFilter
from tdparse.scikit_features.neural_pooling import NeuralPooling
from tdparse.scikit_features.join_context_vectors import JoinContextVectors

class TargetInd():
    def __init__(self):
        self.model = None
        self.pipeline = Pipeline([
            ('contexts', Context('full')),
            ('tokens', ContextTokeniser(ark_twokenize, True)),
            ('word_vectors', ContextWordVectors()),
            ('pool_funcs', FeatureUnion([
                ('max_pipe', Pipeline([
                    ('max', NeuralPooling(matrix_max)),
                    ('join', JoinContextVectors(matrix_median))
                ])),
                ('min_pipe', Pipeline([
                    ('min', NeuralPooling(matrix_min)),
                    ('join', JoinContextVectors(matrix_median))
                ])),
                ('avg_pipe', Pipeline([
                    ('avg', NeuralPooling(matrix_avg)),
                    ('join', JoinContextVectors(matrix_median))
                ])),
                ('prod_pipe', Pipeline([
                    ('min', NeuralPooling(matrix_prod)),
                    ('join', JoinContextVectors(matrix_median))
                ])),
                ('std_pipe', Pipeline([
                    ('min', NeuralPooling(matrix_std)),
                    ('join', JoinContextVectors(matrix_median))
                ]))
            ])),
            ('scale', MinMaxScaler()),
            ('svm', LinearSVC(C=0.01))
        ])
    def find_best_c(self, train_data, train_y, grid_params, **kwargs):
        '''
        :param train_data: Training instances to grid search over
        :param train_y: Training True values to grid search over
        :param grid_params: parameters for the model, all parameters can be \
        found from the `get_cv_params` function. The C value parameter will be \
        ignored if given.
        :param kwargs: keywords arguments to give as arguments to the scikit learn \
        `GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.\
        model_selection.GridSearchCV.html>`_ object e.g. cv=10.
        :type train_data: array/list
        :type train_y: array/list
        :type grid_params: dict
        :type kwargs: dict
        :returns: Searches through two sets of C values a coarse grain values \
        then a fine grain. Grid searches over these values to return the best \
        C value without doing a full exhaustive search. This method inspired by \
        `Hsu et al. SVM guide \
        <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>`_
        :rtype: float
        '''

        def grid_res_to_dict(grid_results):
            c_score = {}
            c_scores = grid_results[['param_svm__C', 'mean_test_score']]
            for i in c_scores.index:
                c_result = c_scores.loc[i]
                c_value = c_result['param_svm__C']
                test_score = c_result['mean_test_score']
                c_score[c_value] = test_score
            return c_score

        # If C value given in grid_params remove it
        if 'C' in grid_params:
            del grid_params['C']

        # Coarse grain search
        coarse_range = []
        start = 0.00001
        stop = 10
        while True:
            coarse_range.append(start)
            start *= 10
            if start > stop:
                break
        grid_params['C'] = coarse_range
        cv_params = self.get_cv_params(**grid_params)
        c_scores = {}
        coarse_results = self.grid_search(train_data, train_y,
                                          params=cv_params, **kwargs)
        c_scores = {**grid_res_to_dict(coarse_results), **c_scores}
        best_coarse_c = self.model.best_params_['svm__C']

        # Fine grain search
        fine_range = [(best_coarse_c / 10) * 3.5,
                      (best_coarse_c / 10) * 7, best_coarse_c,
                      best_coarse_c * 3.5, best_coarse_c * 7]
        grid_params['C'] = fine_range
        cv_params = self.get_cv_params(**grid_params)

        fine_results = self.grid_search(train_data, train_y,
                                        params=cv_params, **kwargs)
        c_scores = {**grid_res_to_dict(fine_results), **c_scores}
        best_c = self.model.best_params_['svm__C']
        return best_c, c_scores



    @staticmethod
    def _get_word_vector_names():
        '''
        Method to be overidden by subclasses as each pipeline will be different
        and will have different parameter names for where the word vectors are
        set.

        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names():
        '''
        Method to be overidden by subclasses as each pipeline will be different
        and will have different parameter names for where the tokenisers are
        set.

        :returns: A list of of parameter names where the tokenisers are set in \
        the pipeline.
        :rtype: list
        '''

        return ['tokens']

    @staticmethod
    def _add_to_params_dict(params_dict, keys, value):
        '''
        Given a dictionary it adds the value to each key in the list of keys
        into the dictionary. Returns the updated dictionary.

        :param params_dict: Dictionary to be updated
        :param keys: list of keys
        :param value: value to be added to each key in the list of keys.
        :type params_dict: dict
        :type keys: list
        :type value: Python object
        :returns: The dictionary updated
        :rtype: dict
        '''

        if not isinstance(keys, list):
            raise ValueError('The keys parameter has to be of type list and not {}'\
                             .format(type(keys)))
        for key in keys:
            params_dict[key] = value
        return params_dict

    def get_params(self, word_vector, tokeniser=None, lower=None, C=None,
                   random_state=None, scale=True):
        '''
        This method is to be overidden when more values than those listed in the
        attributes are required for the model. E.g. a lexicon.

        If values are not required e.g. lower then the model has a defualt value
        for it which will be used when the user does not set a value here.

        :param word_vector: A list of `tdparse.word_vectors.WordVectors` \
        instances e.g. [WordVectors(), AnotherWordVector()]
        :param tokeniser: A tokeniser method from `tdparse.tokenisers` \
        or a method that conforms to the same output as `tdparse.tokenisers`
        :param lower: A bool which indicate wether to lower case the input words.
        :param C: A float which indicates the C value of the SVM classifier.
        :param random_state: A int which defines the random number to generate \
        to shuffle the data. Used to ensure reproducability.
        :param scale: bool indicating to use scaling or not. Default is to scale.
        :type word_vector: list
        :type tokeniser: function
        :type lower: bool
        :type C: float
        :type random_state: int
        :type scale: bool Default True
        :return: A parameter dict which indicates the parameters the model should \
        use. The return of this function can be used as the params attribute in \
        the `fit` method.
        :rtype: dict
        '''
        params_dict = {}
        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_word_vector_names(),
                                               word_vector)
        if tokeniser is not None:
            tokenisers_names = [param_name + '__tokeniser'
                                for param_name in self._get_tokeniser_names()]
            params_dict = self._add_to_params_dict(params_dict, tokenisers_names,
                                                   tokeniser)
        if lower is not None:
            lower_names = [param_name + '__lower'
                           for param_name in self._get_tokeniser_names()]
            params_dict = self._add_to_params_dict(params_dict, lower_names, lower)
        if C is not None:
            params_dict = self._add_to_params_dict(params_dict, ['svm__C'], C)
        if random_state is not None:
            params_dict = self._add_to_params_dict(params_dict,
                                                   ['svm__random_state'], random_state)
        if scale:
            params_dict = self._add_to_params_dict(params_dict, ['scale'],
                                                   MinMaxScaler())
        else:
            params_dict = self._add_to_params_dict(params_dict, ['scale'], None)
        return params_dict

    @staticmethod
    def _add_to_params(params_list, to_add, to_add_names):
        '''
        Used to add parameters that are stated multiple times in the same
        pipeline that must have the same value therefore to add them you
        have to copy the current parameter list N amount of times where N is
        the length of the to_add list. Returns the updated parameter list.
        Method to add parameters that are set in multiple parts of the pipeline
        but should contain the same value.

        :params_list: A list of dicts where each dict contains parameters and \
        corresponding values that are to be searched for. All dict are part of \
        the search space.
        :param to_add: List of values that are to be added to the search space.
        :param to_add_names: List of names that are associated to the values.
        :type params_list: list
        :type to_add: list
        :type to_add_names: list
        :returns: The updated params_list
        :rtype: list
        '''
        num_params = len(params_list)
        num_to_add = len(to_add)
        new_param_list = []
        # Catch the case that params_list was originally empty
        if num_params == 0:
            for _ in range(num_to_add):
                new_param_list.append([defaultdict(list)])
        else:
            for _ in range(num_to_add):
                new_param_list.append(copy.deepcopy(params_list))

        for index, param in enumerate(to_add):
            for param_name in to_add_names:
                for sub_list in new_param_list[index]:
                    sub_list[param_name].append(param)
        params_list = [param_dict for sub_list in new_param_list
                       for param_dict in sub_list]
        return params_list

    @staticmethod
    def _add_to_all_params(params_list, param_name, param_value):
        '''
        Used to add param_name and its values to each dictionary of parameters
        in the params_list. Returns the updated params_list.

        :param params_list: A list of dicts where each dict contains parameters and \
        corresponding values that are to be searched for. All dict are part of \
        the search space.
        :param param_name: The name associated to the parameter value to be added \
        to the params_list.
        :param param_value: The list of values associated to the param_name that are \
        added to the params_list linked to the associated name.
        :type param_list: list
        :type param_name: String
        :type param_value: list
        :returns: The updated params_list
        :rtype: list
        '''
        for param_dict in params_list:
            param_dict[param_name] = param_value
        return params_list


    def get_cv_params(self, word_vectors, tokenisers=None, lowers=None, C=None,
                      scale=None, random_state=None):
        '''
        Each attribute has to be a list which contains parameters that are to be
        tunned.

        This method is to be overidden when more values than those listed in the
        attributes are required for the model. E.g. a lexicon.

        :param word_vectors: A list of a list of `tdparse.word_vectors.WordVectors` \
        instances e.g. [[WordVectors()], [WordVectors(), AnotherWordVector()]]
        :param tokenisers: A list of tokenisers methods from `tdparse.tokenisers` \
        or a list of methods that conform to the same output as `tdparse.tokenisers`
        :param lowers: A list of bool values which indicate wether to lower case \
        the input words.
        :param C: A list of floats which indicate the C value on the SVM classifier.
        :param random_state: A int which defines the random number to generate \
        to shuffle the data. Used to ensure reproducability.
        :param scale: Can only be the value None to not scale the data. Do not \
        include this
        :type word_vectors: list
        :type tokenisers: list
        :type lowers: list
        :type C: list
        :type random_state: int
        :type scale: None
        :return: A list of dicts where each dict represents a different \
        parameter space to search. Used as the params attribute to grid_search \
        function.
        :rtype: list
        '''

        params_list = []
        params_list = self._add_to_params(params_list, word_vectors,
                                          self._get_word_vector_names())
        if tokenisers is not None:
            tokenisers_names = [param_name + '__tokeniser'
                                for param_name in self._get_tokeniser_names()]
            params_list = self._add_to_params(params_list, tokenisers,
                                              tokenisers_names)
        if lowers is not None:
            lower_names = [param_name + '__lower'
                           for param_name in self._get_tokeniser_names()]
            params_list = self._add_to_params(params_list, lowers, lower_names)
        if C is not None:
            params_list = self._add_to_all_params(params_list, 'svm__C', C)
        if random_state is not None:
            random_state = [random_state]
            params_list = self._add_to_all_params(params_list, 'svm__random_state',
                                                  random_state)
        if scale is not None:
            scale_params = []
            if len(scale) > 2:
                raise ValueError('Scale has to be a list, that can only '\
                                 'contain two values False to not scale and '\
                                 'True to scale your list contains more than '\
                                 'two values {}'.format(scale))
            for value in scale:
                if value:
                    scale_params.append(MinMaxScaler())
                else:
                    scale_params.append(None)
            params_list = self._add_to_all_params(params_list, 'scale', scale_params)
        return params_list

    def fit(self, train_data, train_y, params=None):
        if params is None:
            raise ValueError('params attribute has to have at least a value for '\
                             'the word vectors used')
        temp_pipeline = copy.deepcopy(self.pipeline)
        temp_pipeline.set_params(**params)
        temp_pipeline.fit(train_data, train_y)
        self.model = temp_pipeline

    def grid_search(self, train_data, train_y, params=None, **kwargs):
        if params is None:
            raise ValueError('params attribute is the `param_grid` attribute'\
                             ' given to `sklearn.model_selection.GridSearchCV` '\
                             'function.')
        grid_search = GridSearchCV(self.pipeline, param_grid=params, **kwargs)
        self.model = grid_search.fit(train_data, train_y)
        cross_val_results = pd.DataFrame(grid_search.cv_results_)
        return cross_val_results

    def predict(self, test_data):
        if self.model is not None:
            return self.model.predict(test_data)
        else:
            raise ValueError('self.model is not fitted please fit the model '\
                             'using the fit function')
    @staticmethod
    def score(true_values, pred_values, scorer, *args, **kwargs):
        '''
        Performs predicitions on the test_data and then scores the predicitons
        based on the scorer function using the true values. Returns the output
        of scorer function.

        :param true_values: Correct Target values
        :param pred_values: Predicted Target values
        :param scorer: Scoring function. The function must take the true \
        targets as the first parameter and predicted targets as the second \
        parameter. e.g sklearn.metrics.f1_score
        :param args: Additional arguments to the scorer function
        :param kwargs: Additional key word arguments to the scorer function
        :type true_values: array
        :type pred_values: array
        :type scorer: function. Default sklearn.metrics.accuracy_score
        :returns: The output from the scorer based on the true and predicted \
        values normally a float.
        :rtype: scorer output
        '''

        return scorer(true_values, pred_values, *args, **kwargs)

    def __repr__(self):
        return 'Target Indepdent'

class TargetDepC(TargetInd):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('right', Pipeline([
                    ('contexts', Context('right')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('target', Pipeline([
                    ('contexts', Context('target')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ]))
            ])),
            ('scale', MinMaxScaler()),
            ('svm', LinearSVC(C=0.025))
        ])

    @staticmethod
    def _get_word_vector_names():
        '''
        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors']
    @staticmethod
    def _get_tokeniser_names():
        '''
        :returns: A list of of parameter names where the tokenisers are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__tokens',
                'union__right__tokens',
                'union__target__tokens']

    def __repr__(self):
        return 'Target Dependent Context'


class TargetDep(TargetInd):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('right', Pipeline([
                    ('contexts', Context('right')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('target', Pipeline([
                    ('contexts', Context('target')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('full', Pipeline([
                    ('contexts', Context('full')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ]))
            ])),
            ('scale', MinMaxScaler()),
            ('svm', LinearSVC(C=0.01))
        ])

    @staticmethod
    def _get_word_vector_names():
        '''
        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors',
                'union__full__word_vectors__vectors']
    @staticmethod
    def _get_tokeniser_names():
        '''
        :returns: A list of of parameter names where the tokenisers are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__tokens',
                'union__right__tokens',
                'union__target__tokens',
                'union__full__tokens']

    def __repr__(self):
        return 'Target Dependent'

class TargetDepSent(TargetInd):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('left_s', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('filter', LexiconFilter()),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('right', Pipeline([
                    ('contexts', Context('right')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('right_s', Pipeline([
                    ('contexts', Context('right')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('filter', LexiconFilter()),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('target', Pipeline([
                    ('contexts', Context('target')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ])),
                ('full', Pipeline([
                    ('contexts', Context('full')),
                    ('tokens', ContextTokeniser(ark_twokenize, True)),
                    ('word_vectors', ContextWordVectors()),
                    ('pool_funcs', FeatureUnion([
                        ('max_pipe', Pipeline([
                            ('max', NeuralPooling(matrix_max)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('min_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_min)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('avg_pipe', Pipeline([
                            ('avg', NeuralPooling(matrix_avg)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('prod_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_prod)),
                            ('join', JoinContextVectors(matrix_median))
                        ])),
                        ('std_pipe', Pipeline([
                            ('min', NeuralPooling(matrix_std)),
                            ('join', JoinContextVectors(matrix_median))
                        ]))
                    ]))
                ]))
            ])),
            ('scale', MinMaxScaler()),
            ('svm', LinearSVC(C=0.01))
        ])

    @staticmethod
    def _get_word_senti_names():
        '''
        :returns: A list of of parameter names where the sentiment lexicons are \
        set in the pipeline.
        :rtype: list
        '''

        return ['union__left_s__filter__lexicon',
                'union__right_s__filter__lexicon']

    @staticmethod
    def _get_word_vector_names():
        '''
        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__word_vectors__vectors',
                'union__left_s__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__right_s__word_vectors__vectors',
                'union__target__word_vectors__vectors',
                'union__full__word_vectors__vectors']
    @staticmethod
    def _get_tokeniser_names():
        '''
        :returns: A list of of parameter names where the tokenisers are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__tokens',
                'union__left_s__tokens',
                'union__right__tokens',
                'union__right_s__tokens',
                'union__target__tokens',
                'union__full__tokens']

    def get_params(self, word_vector, senti_lexicon, tokeniser=None, lower=None,
                   C=None, scale=True, random_state=None):
        '''
        Overrides the base version and adds the lexicons parameter.
        :param senti_lexicon: Lexicon of words you want to use has to be an instance \
        of `tdparse.lexicons.Lexicon`.
        :type senti_lexicon: instance of `tdparse.lexicons.Lexicon`
        :returns: A parameter dictionary that can be used as the param attribute \
        in the `fit` function.
        :rtype: dict
        '''

        params_dict = super().get_params(word_vector, tokeniser=tokeniser,
                                         lower=lower, C=C, scale=scale,
                                         random_state=random_state)
        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_word_senti_names(),
                                               senti_lexicon)
        return params_dict

    def get_cv_params(self, word_vectors, senti_lexicons, tokenisers=None,
                      lowers=None, C=None, scale=None, random_state=None):
        '''
        Overrides the base version and adds the lexicons parameter.

        :param senti_lexicons: List of instance `tdparse.lexicons.Lexicon` where \
        each lexicon is used to sub select interested words.
        :type senti_lexicon: list
        :returns: A list of dicts where each dict represents a different \
        parameter space to search. Used as the params attribute to grid_search \
        function.
        :rtype: list
        '''

        params_list = super().get_cv_params(word_vectors, tokenisers=tokenisers,
                                            lowers=lowers, C=C, scale=scale,
                                            random_state=random_state)
        params_list = self._add_to_params(params_list, senti_lexicons,
                                          self._get_word_senti_names())
        return params_list

    def __repr__(self):
        return 'Target Dependent Plus'
