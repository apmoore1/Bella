'''
Contains classes of models that can be found in `Vo and Zhang 2015 paper \
<https://www.ijcai.org/Proceedings/15/Papers/194.pdf>`_.

Classes:

1. :py:class:`bella.models.target.TargetInd` - Target indepdent model
'''
from collections import defaultdict
import copy
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from bella.tokenisers import ark_twokenize
from bella.neural_pooling import matrix_max, matrix_min, matrix_avg,\
matrix_median, matrix_prod, matrix_std
from bella.notebook_helper import get_json_data, write_json_data

from bella.scikit_features.context import Context
from bella.scikit_features.tokeniser import ContextTokeniser
from bella.scikit_features.word_vector import ContextWordVectors
from bella.scikit_features.lexicon_filter import LexiconFilter
from bella.scikit_features.neural_pooling import NeuralPooling
from bella.scikit_features.join_context_vectors import JoinContextVectors

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

    def save_model(self, model_file, verbose=0):
        if self.model is None:
            raise ValueError('Model is not fitted please fit the model '\
                             'using the fit function')
        time_taken = time.time()
        joblib.dump(self.model, model_file)
        if verbose == 1:
            time_taken = round(time.time() - time_taken, 2)
            print('Model saved to {}. Save time {}'\
                  .format(model_file, time_taken))

    def load_model(self, model_file, verbose=0):
        if verbose == 1:
            time_taken = time.time()
            print('Loading model from {}'.format(model_file))
            self.model = joblib.load(model_file)
            time_taken = round(time.time() - time_taken, 2)
            print('Model successfully loaded. Load time {}'.format(time_taken))
        else:
            self.model = joblib.load(model_file)

    def find_best_c(self, train_data, train_y, grid_params, save_file=None,
                    dataset_name=None, re_write=False, **kwargs):
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

        def best_c_value(c_scores):
            best = 0
            best_c = 0
            for c_value, acc in c_scores.items():
                if acc > best:
                    best_c = c_value
                    best = acc
            return float(best_c)

        def grid_res_to_dict(grid_results):
            c_score = {}
            c_scores = grid_results[['param_svm__C', 'mean_test_score']]
            for i in c_scores.index:
                c_result = c_scores.loc[i]
                c_value = c_result['param_svm__C']
                test_score = c_result['mean_test_score']
                c_score[c_value] = test_score
            return c_score

        save_file_given = save_file is not None and dataset_name is not None

        # If C value given in grid_params remove it
        if 'C' in grid_params:
            del grid_params['C']
        if save_file_given and not re_write:
            c_scores = get_json_data(save_file, dataset_name)
            if c_scores != {}:
                return best_c_value(c_scores), c_scores

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
        c_2_string = self.c_param_name(c_scores.keys())
        c_scores = {c_2_string[c] : value for c, value in c_scores.items()}
        if save_file_given:
            write_json_data(save_file, dataset_name, c_scores)
        return best_c, c_scores

    def c_param_name(self, c_values):
        '''
        :param c_values: A list of floats representing C values to be mapped to \
        String values
        :type c_values: list
        :returns: A dict of float to String values where the float represents \
        the true C value and the String is it's String representation.
        :rtype: dict
        '''
        return {c_value : str(c_value) for c_value in c_values}

    def senti_lexicon_param_name(self, senti_lexicons):
        '''
        :param all_word_vectors: A list of Lexicon instances
        :type word_vectors: list
        :returns: A dict mapping Lexicon instance with the String name of the \
        lexicon
        :rtype: dict
        '''

        return {senti_lexicon : senti_lexicon.name \
                for senti_lexicon in senti_lexicons}

    def word_vector_param_name(self, all_word_vectors):
        '''
        :param all_word_vectors: A list of a list of WordVector instances
        :type word_vectors: list
        :returns: A dict of tuples containing WordVector instances and there \
        String representation found using their name attribute.
        :rtype: dict
        '''

        word_vector_2_name = {}
        for word_vectors in all_word_vectors:
            word_vectors_tuple = tuple(word_vectors)
            word_vectors_name = [word_vector.name for word_vector in word_vectors]
            word_vectors_name = ' '.join(word_vectors_name)
            word_vector_2_name[word_vectors_tuple] = word_vectors_name
        return word_vector_2_name

    def tokeniser_param_name(self, tokenisers):
        '''
        :param tokenisers: A list of tokeniser functions
        :type tokenisers: list
        :returns: A dict of tokeniser function to the name of the tokeniser \
        function as a String
        :rtype: dict
        '''
        return {tokeniser : tokeniser.__name__ for tokeniser in tokenisers}

    def param_name_function(self, param_name):
        '''
        :param param_name: Name of the only parameter being searched for in \
        the grid search
        :type param_name: String
        :returns: A function that can map the parameter values of the parameter \
        name to meaningful String values
        :rtype: function
        '''
        if param_name == 'word_vectors':
            return self.word_vector_param_name
        elif param_name == 'tokenisers':
            return self.tokeniser_param_name
        elif param_name == 'C':
            return self.c_param_name
        elif param_name == 'senti_lexicons':
            return self.senti_lexicon_param_name
        elif param_name == 'parsers':
            return self.tokeniser_param_name
        else:
            raise ValueError('param_name has to be on of the following values:'\
                             'word_vectors, tokenisers or C not {}'\
                             .format(param_name))






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

        :param word_vector: A list of `bella.word_vectors.WordVectors` \
        instances e.g. [WordVectors(), AnotherWordVector()]
        :param tokeniser: A tokeniser method from `bella.tokenisers` \
        or a method that conforms to the same output as `bella.tokenisers`
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

        :param word_vectors: A list of a list of `bella.word_vectors.WordVectors` \
        instances e.g. [[WordVectors()], [WordVectors(), AnotherWordVector()]]
        :param tokenisers: A list of tokenisers methods from `bella.tokenisers` \
        or a list of methods that conform to the same output as `bella.tokenisers`
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
            if not isinstance(random_state, int):
                raise TypeError('random_state should be of type int and not {}'\
                                .format(type(random_state)))
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
            params_list = self._add_to_all_params(params_list,
                                                  scale_params, ['scale'])
        return params_list

    def fit(self, train_data, train_y, params):
        temp_pipeline = copy.deepcopy(self.pipeline)
        temp_pipeline.set_params(**params)
        temp_pipeline.fit(train_data, train_y)
        self.model = temp_pipeline

    def grid_search(self, train_data, train_y, params, **kwargs):

        grid_search = GridSearchCV(self.pipeline, param_grid=params, **kwargs)
        self.model = grid_search.fit(train_data, train_y)
        cross_val_results = pd.DataFrame(grid_search.cv_results_)
        return cross_val_results

    def save_grid_search(self, train_data, train_y, grid_params, save_param,
                         dataset_name, file_name, re_write=False,
                         **kwargs):
        '''
        write_json_data(word_vector_file_path, name, word_vector_results)
        Saves the results of the grid search to json file where the result is
        stored per dataset and then per parameter being searched. Note that
        only one parameter type can be searched and saved per time e.g.
        searching for different tokenisers but you cannot search for different
        tokeniser and word vectors at the moment. If you would like those
        results without saving and caching please use grid_search function.
        '''

        if save_param not in grid_params:
            raise ValueError('save_param {} has to be a key in the grid_params'\
                             ' dict {}'.fromat(save_param, grid_params))

        param_values = grid_params[save_param]
        param_name = self.param_name_function(save_param)(param_values)
        name_score = {}
        name_param = {}
        if not re_write:
            name_score = get_json_data(file_name, dataset_name)
        temp_grid_params = copy.deepcopy(grid_params)
        for param in param_values:
            if isinstance(param, list):
                param = tuple(param)
            name = param_name[param]
            name_param[name] = param
            if not re_write:
                if name in name_score:
                    continue
            temp_grid_params[save_param] = [param]
            cv_params = self.get_cv_params(**temp_grid_params)
            grid_res = self.grid_search(train_data, train_y, cv_params, **kwargs)
            if grid_res.shape[0] != 1:
                raise ValueError('Searching over more than one parameter this '\
                                 'cannot be allowed as only one value can be '\
                                 'can be associated to a search parameter at a '\
                                 'time. Grid results {} parameter being searched'\
                                 ' {}'.format(grid_res, save_param))
            name_score[name] = grid_res['mean_test_score'][0]
        write_json_data(file_name, dataset_name, name_score)
        sorted_name_score = sorted(name_score.items(), key=lambda x: x[1],
                                   reverse=True)
        best_param = None
        for name, score in sorted_name_score:
            if name not in name_param:
                continue
            best_param = name_param[name]
            break
        if best_param is None:
            raise ValueError('best_param cannot be None this should only happen'\
                             ' if no parameters are being searched for. Param '\
                             ' names that have been searched {}'\
                             .formated(name_param))
        if isinstance(best_param, tuple):
            return list(best_param)
        return best_param

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
        of `bella.lexicons.Lexicon`.
        :type senti_lexicon: instance of `bella.lexicons.Lexicon`
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

        :param senti_lexicons: List of instance `bella.lexicons.Lexicon` where \
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
