'''
Module contains all of the classes that represent Machine Learning models
that are within `Vo and Zhang 2015 paper \
<https://ijcai.org/Proceedings/15/Papers/194.pdf>`_:

1. :py:class:`bella.models.target.TargetInd` -- Target Indepdent model
2. :py:class:`bella.models.target.TargetDepMinus` -- Target Dependent Minus
   model
3. :py:class:`bella.models.target.TargetDep` -- Target Dependent model
4. :py:class:`bella.models.target.TargetDepPlus` -- Target Dependent Plus model
'''

from typing import Any, List, Callable, Dict

import sklearn
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

import bella
from bella.tokenisers import ark_twokenize
from bella.neural_pooling import matrix_max, matrix_min, matrix_avg
from bella.neural_pooling import matrix_median, matrix_prod, matrix_std
from bella.scikit_features.context import Context
from bella.scikit_features.lexicon_filter import LexiconFilter
from bella.scikit_features.join_context_vectors import JoinContextVectors
from bella.scikit_features.neural_pooling import NeuralPooling
from bella.scikit_features.tokeniser import ContextTokeniser
from bella.scikit_features.word_vector import ContextWordVectors
from bella.models.base import SKLearnModel


class TargetInd(SKLearnModel):
    '''
    Attributes:

    1. model -- Machine learning model. Expects it to be a
       :py:class:`sklearn.pipeline.Pipeline` instance.
    2. fitted -- If the machine learning model has been fitted (default False)
    3. model_parameters -- The parameters that are set in the machine
       learning model. E.g. Parameter could be the tokeniser used.

    Methods:

    1. fit -- Fit the model according to the given training data.
    2. predict -- Predict class labels for samples in X.
    3. probabilities -- The probability of each class label for all samples
       in X.
    4. __repr__ -- Name of the machine learning model.

    Class Methods:

    1. get_parameters -- Transform the given parameters into a dictonary
       that is accepted as model parameters.
    2. get_cv_parameters -- Transform the given parameters into a list of
       dictonaries that is accepted as `param_grid` parameter in
       :py:class:`sklearn.model_selection.GridSearchCV`
    3. name -- -- Returns the name of the model.

    Functions:

    1. save -- Given a instance of this class will save it to a file.
    2. load -- Loads an instance of this class from a file.
    3. pipeline -- Machine Learning model that is used as the base template
       for the model attribute.
    '''

    @classmethod
    def name(cls) -> str:
        return 'Target Independent'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Whether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        '''
        # The parameters here go into the self.get_parameters method
        super().__init__(word_vectors, tokeniser, lower, C, random_state,
                         scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('contexts', Context('full')),
            ('tokens', ContextTokeniser()),
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
            ('svm', LinearSVC())
        ])

    @classmethod
    def get_parameters(cls,
                       word_vectors: List['bella.word_vectors.WordVectors'],
                       tokeniser: Callable[[str], List[str]] = ark_twokenize,
                       lower: bool = True, C: float = 0.01,
                       random_state: int = 42,
                       scale: Any = MinMaxScaler()) -> Dict[str, Any]:
        '''
        Transform the given parameters into a dictonary that is accepted as
        model parameters

        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Whether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        :return: Model parameters
        '''
        params_dict = {}
        # Add word vectors to Pipeline model
        params_dict = cls._add_to_params_dict(params_dict,
                                              cls._get_word_vector_names(),
                                              word_vectors)
        # Add tokenisers to Pipeline model
        tokenisers_names = [param_name + '__tokeniser'
                            for param_name in cls._get_tokeniser_names()]
        params_dict = cls._add_to_params_dict(params_dict,
                                              tokenisers_names,
                                              tokeniser)
        # Add if the words should be lower cased
        lower_names = [param_name + '__lower'
                       for param_name in cls._get_tokeniser_names()]
        params_dict = cls._add_to_params_dict(params_dict, lower_names, lower)
        # Add how the data should be scaled before going into the SVM
        # If None then it means no scaling happens
        params_dict = cls._add_to_params_dict(params_dict, ['scale'],
                                              scale)
        # Add the C value for the SVM
        params_dict = cls._add_to_params_dict(params_dict, ['svm__C'], C)
        # Add the random state for the SVM
        params_dict = cls._add_to_params_dict(params_dict,
                                              ['svm__random_state'],
                                              random_state)
        return params_dict

    @classmethod
    def get_cv_parameters(cls, word_vectors, tokeniser=[ark_twokenize],
                          lower=[True], C=[0.01], random_state=[42],
                          scale=[MinMaxScaler()]):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]].
        :param tokenisers: A list of tokeniser to be used e.g.
                           :py:meth:`str.split`. Default [ark_twokenize]
        :param lowers: A list of bool values which indicate whether to lower
                       case the input words. Default [True]
        :param C: A list of C values for the :py:class:`sklearn.svm.SVC`
                  estimator that is used in the pipeline. Default [0.01]
        :param random_state: A list of random_state values for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline. Default [42]
        :param scale: List of scale values. The list can include
                      :py:class:`sklearn.preprocessing.MinMaxScaler` type of
                      clases or None if no scaling is to be used. Default
                      [:py:class:`sklearn.preprocessing.MinMaxScaler`]
        :return: Parameters to explore through cross validation
        '''
        params_list = []
        # Word Vectors
        params_list = cls._add_to_params(params_list, word_vectors,
                                         cls._get_word_vector_names())
        # Tokeniser
        tokenisers_names = [param_name + '__tokeniser'
                            for param_name in cls._get_tokeniser_names()]
        params_list = cls._add_to_params(params_list, tokeniser,
                                         tokenisers_names)
        # Lower
        lower_names = [param_name + '__lower'
                       for param_name in cls._get_tokeniser_names()]
        params_list = cls._add_to_params(params_list, lower, lower_names)
        # C
        params_list = cls._add_to_all_params(params_list, 'svm__C', C)
        # Random State
        params_list = cls._add_to_all_params(params_list, 'svm__random_state',
                                             random_state)
        # Scale
        params_list = cls._add_to_all_params(params_list, 'scale', scale)
        return params_list

    @staticmethod
    def _get_word_vector_names() -> List[str]:
        '''
        Method to be overidden by subclasses as each pipeline will be different
        and will have different parameter names for where the word vectors are
        set.

        :returns: A list of of parameter names where the word vectors are set
                  in the pipeline.
        '''

        return ['word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names() -> List[str]:
        '''
        Method to be overidden by subclasses as each pipeline will be different
        and will have different parameter names for where tokenisers are
        used.

        :returns: A list of of parameter names where the tokenisers are used
                  in the pipeline.
        '''

        return ['tokens']


class TargetDepMinus(TargetInd):

    @classmethod
    def name(cls) -> str:
        return 'Target Dependent Minus'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.025,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Wether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is used
                             in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        '''
        # Inherit from SKLearnModel __init__ method
        # The parameters here go into the self.get_parameters method
        super(TargetInd, self).__init__(word_vectors, tokeniser, lower, C,
                                        random_state, scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
            ('svm', LinearSVC())
        ])

    @classmethod
    def get_parameters(cls,
                       word_vectors: List['bella.word_vectors.WordVectors'],
                       tokeniser: Callable[[str], List[str]] = ark_twokenize,
                       lower: bool = True, C: float = 0.025,
                       random_state: int = 42,
                       scale: Any = MinMaxScaler()) -> Dict[str, Any]:
        '''
        Transform the given parameters into a dictonary that is accepted as
        model parameters

        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Wether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is used
                             in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        :return: Model parameters
        '''
        return super().get_parameters(word_vectors, tokeniser, lower, C,
                                      random_state, scale)

    @classmethod
    def get_cv_parameters(cls, word_vectors, tokeniser=[ark_twokenize],
                          lower=[True], C=[0.025], random_state=[42],
                          scale=[MinMaxScaler()]):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]].
        :param tokenisers: A list of tokeniser to be used e.g.
                           :py:meth:`str.split`. Default [ark_twokenize]
        :param lowers: A list of bool values which indicate whether to lower
                       case the input words. Default [True]
        :param C: A list of C values for the :py:class:`sklearn.svm.SVC`
                  estimator that is used in the pipeline. Default [0.025]
        :param random_state: A list of random_state values for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline. Default [42]
        :param scale: List of scale values. The list can include
                      :py:class:`sklearn.preprocessing.MinMaxScaler` type of
                      clases or None if no scaling is to be used. Default
                      [:py:class:`sklearn.preprocessing.MinMaxScaler`]
        :return: Parameters to explore through cross validation
        '''
        return super().get_cv_parameters(word_vectors, tokeniser, lower, C,
                                         random_state, scale)

    @staticmethod
    def _get_word_vector_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the word vectors are set
        in the pipeline.
        '''

        return ['union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the tokenisers are set
        in the pipeline.
        '''

        return ['union__left__tokens',
                'union__right__tokens',
                'union__target__tokens']


class TargetDep(TargetInd):
    '''
    Target-dep model from `Vo and Zhang 2015 paper
    <https://ijcai.org/Proceedings/15/Papers/194.pdf>`_.
    '''

    @classmethod
    def name(cls) -> str:
        return 'Target Dependent'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Wether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        '''
        # Inherit from SKLearnModel __init__ method
        # The parameters here go into the self.get_parameters method
        super(TargetInd, self).__init__(word_vectors, tokeniser, lower, C,
                                        random_state, scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
            ('svm', LinearSVC())
        ])

    @classmethod
    def get_parameters(cls,
                       word_vectors: List['bella.word_vectors.WordVectors'],
                       tokeniser: Callable[[str], List[str]] = ark_twokenize,
                       lower: bool = True, C: float = 0.01,
                       random_state: int = 42,
                       scale: Any = MinMaxScaler()) -> Dict[str, Any]:
        '''
        Transform the given parameters into a dictonary that is accepted as
        model parameters

        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Wether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        :return: Model parameters
        '''
        return super().get_parameters(word_vectors, tokeniser, lower, C,
                                      random_state, scale)

    @classmethod
    def get_cv_parameters(cls, word_vectors, tokeniser=[ark_twokenize],
                          lower=[True], C=[0.01], random_state=[42],
                          scale=[MinMaxScaler()]):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]].
        :param tokenisers: A list of tokeniser to be used e.g.
                           :py:meth:`str.split`. Default [ark_twokenize]
        :param lowers: A list of bool values which indicate whether to lower
                       case the input words. Default [True]
        :param C: A list of C values for the :py:class:`sklearn.svm.SVC`
                  estimator that is used in the pipeline. Default [0.01]
        :param random_state: A list of random_state values for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline. Default [42]
        :param scale: List of scale values. The list can include
                      :py:class:`sklearn.preprocessing.MinMaxScaler` type of
                      clases or None if no scaling is to be used. Default
                      [:py:class:`sklearn.preprocessing.MinMaxScaler`]
        :return: Parameters to explore through cross validation
        '''
        return super().get_cv_parameters(word_vectors, tokeniser, lower, C,
                                         random_state, scale)

    @staticmethod
    def _get_word_vector_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the word vectors are set
        in the pipeline.
        '''

        return ['union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors',
                'union__full__word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the tokenisers are set
        in the pipeline.
        '''

        return ['union__left__tokens',
                'union__right__tokens',
                'union__target__tokens',
                'union__full__tokens']


class TargetDepPlus(TargetInd):

    @classmethod
    def name(cls) -> str:
        return 'Target Dependent Plus'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 senti_lexicon: 'bella.lexicons.Lexicon',
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param senti_lexicon: Sentiment Lexicon to be used for the Left and
                              Right sentiment context (LS and RS).
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Whether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is used
                             in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        '''
        # Inherit from SKLearnModel __init__ method
        # The parameters here go into the self.get_parameters method
        super(TargetInd, self).__init__(word_vectors, senti_lexicon, tokeniser,
                                        lower, C, random_state, scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context('left')),
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
                    ('tokens', ContextTokeniser()),
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
            ('svm', LinearSVC())
        ])

    @classmethod
    def get_parameters(cls,
                       word_vectors: List['bella.word_vectors.WordVectors'],
                       senti_lexicon: 'bella.lexicons.Lexicon',
                       tokeniser: Callable[[str], List[str]] = ark_twokenize,
                       lower: bool = True, C: float = 0.01,
                       random_state: int = 42,
                       scale: Any = MinMaxScaler()) -> Dict[str, Any]:
        '''
        Transform the given parameters into a dictonary that is accepted as
        model parameters

        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param senti_lexicon: Sentiment Lexicon to be used for the Left and
                              Right sentiment context (LS and RS).
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Whether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator
                  that is used in the pipeline.
        :param random_state: The random_state value for the
                             :py:class:`sklearn.svm.SVC` estimator that is used
                             in the pipeline.
        :param scale: How to scale the data before input into the estimator.
                      If no scaling is to be used set this to None.
        :return: Model parameters
        '''
        params_dict = super().get_parameters(word_vectors, tokeniser, lower, C,
                                             random_state, scale)
        params_dict = cls._add_to_params_dict(params_dict,
                                              cls._get_word_senti_names(),
                                              senti_lexicon)
        return params_dict

    @classmethod
    def get_cv_parameters(cls,
                          word_vectors: List[List['bella.word_vectors\
                                                        .WordVectors']],
                          senti_lexicon: List['bella.lexicons.Lexicon'],
                          tokeniser=[ark_twokenize],
                          lower=[True], C=[0.01], random_state=[42],
                          scale=[MinMaxScaler()]):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]].
        :param senti_lexicon: A list of Sentiment Lexicons to be explored for
                              the Left and Right sentiment context (LS and RS).
                              Default None, use the sentiment lexicons already
                              within the model.
        :param tokenisers: A list of tokeniser to be used e.g.
                           :py:meth:`str.split`. Default [ark_twokenize]
        :param lowers: A list of bool values which indicate whether to lower
                       case the input words. Default [True]
        :param C: A list of C values for the :py:class:`sklearn.svm.SVC`
                  estimator that is used in the pipeline. Default [0.01]
        :param random_state: A list of random_state values for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline. Default [42]
        :param scale: List of scale values. The list can include
                      :py:class:`sklearn.preprocessing.MinMaxScaler` type of
                      clases or None if no scaling is to be used. Default
                      [:py:class:`sklearn.preprocessing.MinMaxScaler`]
        :return: Parameters to explore through cross validation
        '''
        params_list = super().get_cv_parameters(word_vectors, tokeniser,
                                                lower, C, random_state, scale)
        params_list = cls._add_to_params(params_list, senti_lexicon,
                                         cls._get_word_senti_names())
        return params_list

    @staticmethod
    def _get_word_vector_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the word vectors are set
        in the pipeline.
        '''

        return ['union__left__word_vectors__vectors',
                'union__left_s__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__right_s__word_vectors__vectors',
                'union__target__word_vectors__vectors',
                'union__full__word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the tokenisers are set
        in the pipeline.
        '''

        return ['union__left__tokens',
                'union__left_s__tokens',
                'union__right__tokens',
                'union__right_s__tokens',
                'union__target__tokens',
                'union__full__tokens']

    @staticmethod
    def _get_word_senti_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the sentiment lexicons
        are set in the pipeline.
        '''

        return ['union__left_s__filter__lexicon',
                'union__right_s__filter__lexicon']
