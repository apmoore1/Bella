'''
Module contains all of the classes that represent Machine Learning models
that are within `Wang et al. paper \
<https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046>`_.

1. :py:class:`bella.models.target.TDParseMinus` -- TDParse Minus model
2. :py:class:`bella.models.target.TDParse` -- TDParse model
3. :py:class:`bella.models.tdparse.TDParsePlus` -- TDParse Plus model
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
from bella.scikit_features import syntactic_context
from bella.scikit_features.lexicon_filter import LexiconFilter
from bella.scikit_features.join_context_vectors import JoinContextVectors
from bella.scikit_features.neural_pooling import NeuralPooling
from bella.scikit_features.tokeniser import ContextTokeniser
from bella.scikit_features.word_vector import ContextWordVectors
from bella.models.target import TargetInd


class TDParseMinus(TargetInd):

    @classmethod
    def name(cls) -> str:
        return 'TDParse Minus'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 parser: Any,
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param parser: The dependency parser to be used.
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
        # Inherit from SKLearnModel __init__ method
        # The parameters here go into the self.get_parameters method
        super(TargetInd, self).__init__(word_vectors, parser, tokeniser, lower,
                                        C, random_state, scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('dependency_context', syntactic_context.SyntacticContext()),
            ('contexts', syntactic_context.Context('full')),
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
                       parser: Any,
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
        :param parser: The dependency parser to be used.
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
        params_list = super().get_parameters(word_vectors, tokeniser,
                                             lower, C, random_state, scale)
        params_list = cls._add_to_params_dict(params_list,
                                              cls._get_dependency_context(),
                                              parser)
        return params_list

    @classmethod
    def get_cv_parameters(cls,
                          word_vectors: List[List['bella.word_vectors\
                                                        .WordVectors']],
                          parser: List[Any],
                          tokeniser=[ark_twokenize],
                          lower=[True], C=[0.01], random_state=[42],
                          scale=[MinMaxScaler()]):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]].
        :param parser: A list of dependency parser to be used.
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
        # dependency parser
        dep_context = cls._get_dependency_context()[0]
        params_list = cls._add_to_all_params(params_list, dep_context,
                                             parser)
        return params_list

    @staticmethod
    def _get_dependency_context() -> List[str]:
        '''
        Method to be overidden by subclasses as each pipeline will be different
        and will have a different parameter name for where the dependency
        parser process.

        :returns: A list of parameters names where the dependency parser is
                  set in the pipeline
        '''

        return ['dependency_context__parser']


class TDParse(TDParseMinus):

    @classmethod
    def name(cls) -> str:
        return 'TDParse'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 parser: Any,
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param parser: The dependency parser to be used.
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
        # Inherit from SKLearnModel __init__ method
        # The parameters here go into the self.get_parameters method
        super(TargetInd, self).__init__(word_vectors, parser, tokeniser, lower,
                                        C, random_state, scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('union', FeatureUnion([
                ('dependency', Pipeline([
                    ('context', syntactic_context.SyntacticContext()),
                    ('contexts', syntactic_context.Context('full')),
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

    @staticmethod
    def _get_word_vector_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the word vectors are set
        in the pipeline.
        '''

        return ['union__dependency__word_vectors__vectors',
                'union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the tokenisers are set
        in the pipeline.
        '''

        return ['union__dependency__tokens',
                'union__left__tokens',
                'union__right__tokens',
                'union__target__tokens']

    @staticmethod
    def _get_dependency_context() -> List[str]:
        '''
        :returns: A list of parameters names where the dependency parser is
                  set in the pipeline
        '''

        return ['union__dependency__context__parser']


class TDParsePlus(TDParseMinus):

    @classmethod
    def name(cls) -> str:
        return 'TDParsePlus'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 parser: Any, senti_lexicon: 'bella.lexicons.Lexicon',
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as
                             feature vector lookups. If more than one is used
                             the word vectors are concatenated together to
                             create a the feature vector for each word.
        :param parser: The dependency parser to be used.
        :param senti_lexicon: Sentiment Lexicon to be used for the Left and
                              Right sentiment context (LS and RS).
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

        # Inherit from SKLearnModel __init__ method
        # The parameters here go into the self.get_parameters method
        super(TargetInd, self).__init__(word_vectors, parser, senti_lexicon,
                                        tokeniser, lower, C, random_state,
                                        scale)

    @staticmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        return Pipeline([
            ('union', FeatureUnion([
                ('dependency', Pipeline([
                    ('context', syntactic_context.SyntacticContext()),
                    ('contexts', syntactic_context.Context('full')),
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
                ]))
            ])),
            ('scale', MinMaxScaler()),
            ('svm', LinearSVC())
        ])

    @classmethod
    def get_parameters(cls,
                       word_vectors: List['bella.word_vectors.WordVectors'],
                       parser: Any,
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
        :param parser: The dependency parser to be used.
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
        params_dict = super().get_parameters(word_vectors, parser, tokeniser,
                                             lower, C, random_state, scale)
        params_dict = cls._add_to_params_dict(params_dict,
                                              cls._get_word_senti_names(),
                                              senti_lexicon)
        return params_dict

    @classmethod
    def get_cv_parameters(cls,
                          word_vectors: List[List['bella.word_vectors\
                                                        .WordVectors']],
                          parser: List[Any],
                          senti_lexicon: List['bella.lexicons\
                                                    .Lexicon'],
                          tokeniser=[ark_twokenize],
                          lower=[True], C=[0.01], random_state=[42],
                          scale=[MinMaxScaler()]):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]].
        :param parser: A list of dependency parser to be used.
        :param senti_lexicon: A list of Sentiment Lexicons to be explored for
                              the Left and Right sentiment context (LS and RS).
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
        params_list = super().get_cv_parameters(word_vectors, parser,
                                                tokeniser, lower, C,
                                                random_state, scale)
        # sentiment lexicon
        params_list = cls._add_to_params(params_list, senti_lexicon,
                                         cls._get_word_senti_names())
        return params_list

    @staticmethod
    def _get_word_vector_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the word vectors are set
        in the pipeline.
        '''

        return ['union__dependency__word_vectors__vectors',
                'union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors',
                'union__right_s__word_vectors__vectors',
                'union__left_s__word_vectors__vectors']

    @staticmethod
    def _get_tokeniser_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the tokenisers are set
        in the pipeline.
        '''

        return ['union__dependency__tokens',
                'union__left__tokens',
                'union__right__tokens',
                'union__target__tokens',
                'union__right_s__tokens',
                'union__left_s__tokens']

    @staticmethod
    def _get_dependency_context() -> List[str]:
        '''
        :returns: A list of parameters names where the dependency parser is
                  set in the pipeline
        '''

        return ['union__dependency__context__parser']

    @staticmethod
    def _get_word_senti_names() -> List[str]:
        '''
        :returns: A list of of parameter names where the sentiment lexicons
        are set in the pipeline.
        '''

        return ['union__left_s__filter__lexicon',
                'union__right_s__filter__lexicon']
