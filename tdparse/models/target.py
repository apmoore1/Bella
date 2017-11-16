'''
Contains classes of models that can be found in `Vo and Zhang 2015 paper \
<https://www.ijcai.org/Proceedings/15/Papers/194.pdf>`_.

Classes:

1. :py:class:`tdparse.models.target.TargetInd` - Target indepdent model
'''
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler

from tdparse.tokenisers import ark_twokenize
from tdparse.word_vectors import WordVectors
from tdparse.neural_pooling import matrix_max, matrix_min, matrix_avg,\
matrix_median, matrix_prod, matrix_std

from tdparse.scikit_features.context import Context
from tdparse.scikit_features.tokeniser import ContextTokeniser
from tdparse.scikit_features.word_vector import ContextWordVectors
from tdparse.scikit_features.neural_pooling import NeuralPooling
from tdparse.scikit_features.join_context_vectors import JoinContextVectors

class TargetInd():
    def __init__(self):
        self.model = None
        self.pipeline = Pipeline([
            ('contexts', Context({'f'})),
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
            ('scale', MaxAbsScaler()),
            ('svm', LinearSVC(C=0.01))
        ])
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

        return ['word_vectors__vector']

    def get_params_dict(self, word_vectors):
        '''
        Paramter setting:
        Given a list of :py:class:`tdparse.word_vectors.WordVectors` instances
        where if more than one instances is given indicates to concatenate the
        word vectors it will return a dict which is input to the param attribute
        for the `fit` function.

        Grid Search:
        Given a list of a list of :py:class:`tdparse.word_vectors.WordVectors`
        instances it will return a list of dicts of which this list can be
        given as the param attribute to the `grid_search` function.

        This function is expect to be overidden when the pipeline is changed.

        :param word_vector: The word vectors you want to use in the model.
        :type word_vector: :list of :py:class:`tdparse.word_vectors.WordVectors` \
        instances
        :returns: Returns a dict or a list of parameters to be given to either \
        `fit` or `grid_search` params attribute.
        :rtype: dict or list
        '''

        if not isinstance(word_vectors, list):
            raise TypeError('word vectors has to be a list of `tdparse.word_'\
                            'vectors.WordVectors` instances or a list of a list of '\
                            '`tdparse.word_vectors.WordVectors` for grid searching')

        if isinstance(word_vectors[0], list):
            params_dict = []
            for word_vector in word_vectors:
                for vector in word_vector:
                    if not isinstance(vector, WordVectors):
                        raise TypeError('each list in the grid search should contain'\
                                        ' instances of {} and {}'\
                                        .format(type(WordVectors), type(vector)))
                param_dict = defaultdict(list)
                for word_vec_name in self._get_word_vector_names():
                    param_dict[word_vec_name].append(word_vector)
                params_dict.append(param_dict)
            return params_dict
        for word_vector in word_vectors:
            if not isinstance(word_vector, WordVectors):
                raise TypeError('The list should contain instances of {} and not {}'\
                                .format(type(WordVectors), type(word_vector)))
        param_dict = {}
        for word_vec_name in self._get_word_vector_names():
            param_dict[word_vec_name] = word_vectors
        return param_dict

    def fit(self, train_data, train_y, params=None):
        if params is None:
            raise ValueError('params attribute has to have at least a value for '\
                             'the word vectors used')
        self.pipeline.set_params(**params)
        self.pipeline.fit(train_data, train_y)
        self.model = self.pipeline

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
class TargetDepC(TargetInd):
    def __init__(self):
        super().__init__()
        print(type(self.model))
        self.pipeline = Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context({'l'})),
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
                    ('contexts', Context({'r'})),
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
                    ('contexts', Context({'t'})),
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
            ('scale', MaxAbsScaler()),
            ('svm', LinearSVC(C=0.01))
        ])

    @staticmethod
    def _get_word_vector_names():
        '''
        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__left__word_vectors__vector',
                'union__right__word_vectors__vector',
                'union__target__word_vectors__vector']
