'''
Contains classes of models that can be found in `Wang et al. paper \
<https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046>`_.

Classes:

1. TDParseMinus
2. TDParse
3. TDParsePlus
'''

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

from bella.models.target import TargetInd

from bella.tokenisers import ark_twokenize
from bella.neural_pooling import matrix_max, matrix_min, matrix_avg,\
matrix_median, matrix_prod, matrix_std

from bella.scikit_features.context import Context
from bella.scikit_features import syntactic_context
from bella.scikit_features.tokeniser import ContextTokeniser
from bella.scikit_features.word_vector import ContextWordVectors
from bella.scikit_features.lexicon_filter import LexiconFilter
from bella.scikit_features.neural_pooling import NeuralPooling
from bella.scikit_features.join_context_vectors import JoinContextVectors

class TDParseMinus(TargetInd):
    def __init__(self, child_relations=False):
        super().__init__()
        self.child_relations = child_relations
        self.pipeline = Pipeline([
            ('dependency_context', syntactic_context.SyntacticContext()),
            ('contexts', syntactic_context.Context('full')),
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
        if child_relations:
            self.pipeline = Pipeline([
                ('dependency_context', syntactic_context.DependencyChildContext()),
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

    @staticmethod
    def _get_dependency_context():
        '''
        Method to be overidden by subclasses as each pipeline will be different
        and will have a different parameter name for where the dependency parser
        related process.

        Dependency related process can be either of the following classes:

        1. syntactic_context.DependencyChildContext
        2. syntactic_context.SyntacticContext

        :returns: A String of the parameter name where the related dependency \
        classes are set in the pipeline.
        :rtype: String
        '''

        return 'dependency_context'

    def get_params(self, word_vector, parser, tokeniser=None, token_lower=None,
                   parser_lower=None, C=None, random_state=None, scale=True,
                   rel_depth=None):
        '''
        Method extended to include parser, parser_lower and rel_depth
        parameters.
        '''
        params_dict = super().get_params(word_vector, tokeniser=tokeniser,
                                         lower=token_lower, C=C,
                                         random_state=random_state, scale=scale)
        if self.child_relations:
            if rel_depth is not None:
                if not isinstance(rel_depth, tuple):
                    raise TypeError('rel_depth has to be a tuple not {}'\
                                    .format(type(rel_depth)))
                rel_depth_name = [self._get_dependency_context() +
                                  '__rel_depth']
                params_dict = self._add_to_params_dict(params_dict,
                                                       rel_depth_name,
                                                       rel_depth)
        else:
            if rel_depth is not None:
                error_msg = '''rel_depth has been set but will not be used as
                               you are using the Syntactic Context (full
                               dependency tree) if you wish to use the dependency
                               child context in the constructor of this class
                               set child_relations=True'''
                ValueError(error_msg)
        if parser_lower is not None:
            parser_lower_name = [self._get_dependency_context() + '__lower']
            params_dict = self._add_to_params_dict(params_dict,
                                                   parser_lower_name,
                                                   parser_lower)
        parser_name = [self._get_dependency_context() + '__parser']
        params_dict = self._add_to_params_dict(params_dict, parser_name, parser)
        return params_dict

    def get_cv_params(self, word_vectors, parsers, tokenisers=None,
                      token_lowers=None, C=None, scale=None, random_state=None,
                      parser_lowers=None, rel_depths=None):
        '''
        Method extended to include parser, parser_lower and rel_depth
        parameters.
        '''
        params_list = super().get_cv_params(word_vectors, tokenisers=tokenisers,
                                            lowers=token_lowers, C=C, scale=scale,
                                            random_state=random_state)
        if self.child_relations:
            if rel_depths is not None:
                if not isinstance(rel_depths[0], tuple):
                    raise TypeError('rel_depths has to be a list of tuples not {}'\
                                    .format(type(rel_depths[0])))
                rel_depth_name = self._get_dependency_context() + '__rel_depth'
                params_list = self._add_to_all_params(params_list,
                                                      rel_depth_name, rel_depths)
        else:
            if rel_depths is not None:
                error_msg = '''rel_depths has been set but will not be used as
                               you are using the Syntactic Context (full
                               dependency tree) if you wish to use the dependency
                               child context in the constructor of this class
                               set child_relations=True'''
                ValueError(error_msg)
        if parser_lowers is not None:
            parser_lower_name = self._get_dependency_context() + '__lower'
            params_list = self._add_to_all_params(params_list, parser_lower_name,
                                                  parser_lowers)
        parser_name = self._get_dependency_context() + '__parser'
        params_list = self._add_to_all_params(params_list, parser_name, parsers)
        return params_list

    def __repr__(self):
        return 'TDParse Minus'

class TDParse(TDParseMinus):
    def __init__(self, child_relations=False):
        super().__init__(child_relations)
        self.pipeline = Pipeline([
            ('union', FeatureUnion([
                ('dependency', Pipeline([
                    ('context', syntactic_context.SyntacticContext()),
                    ('contexts', syntactic_context.Context('full')),
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
            ('svm', LinearSVC(C=0.01))
        ])
        if child_relations:
            self.pipeline = Pipeline([
                ('union', FeatureUnion([
                    ('dependency', Pipeline([
                        ('context', syntactic_context.DependencyChildContext()),
                        ('contexts', syntactic_context.Context('full')),
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
                ('svm', LinearSVC(C=0.01))
            ])
    @staticmethod
    def _get_word_vector_names():
        '''
        Overideen

        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__dependency__word_vectors__vectors',
                'union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors']
    @staticmethod
    def _get_tokeniser_names():
        '''
        Overideen

        :returns: A list of of parameter names where the tokenisers are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__dependency__tokens',
                'union__left__tokens',
                'union__right__tokens',
                'union__target__tokens']

    @staticmethod
    def _get_dependency_context():
        '''
        Overideen

        Method to be overidden by subclasses as each pipeline will be different
        and will have a different parameter name for where the dependency parser
        related process.

        Dependency related process can be either of the following classes:

        1. syntactic_context.DependencyChildContext
        2. syntactic_context.SyntacticContext

        :returns: A String of the parameter name where the related dependency \
        classes are set in the pipeline.
        :rtype: String
        '''

        return 'union__dependency__context'

    def __repr__(self):
        return 'TDParse'

class TDParsePlus(TDParse):
    def __init__(self, child_relations=False):
        super().__init__(child_relations)
        self.pipeline = Pipeline([
            ('union', FeatureUnion([
                ('dependency', Pipeline([
                    ('context', syntactic_context.SyntacticContext()),
                    ('contexts', syntactic_context.Context('full')),
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
                ]))
            ])),
            ('scale', MinMaxScaler()),
            ('svm', LinearSVC(C=0.01))
        ])
        if child_relations:
            self.pipeline = Pipeline([
                ('union', FeatureUnion([
                    ('dependency', Pipeline([
                        ('context', syntactic_context.DependencyChildContext()),
                        ('contexts', syntactic_context.Context('full')),
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
                    ]))
                ])),
                ('scale', MinMaxScaler()),
                ('svm', LinearSVC(C=0.01))
            ])
    @staticmethod
    def _get_word_vector_names():
        '''
        Overideen

        :returns: A list of of parameter names where the word vectors are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__dependency__word_vectors__vectors',
                'union__left__word_vectors__vectors',
                'union__right__word_vectors__vectors',
                'union__target__word_vectors__vectors',
                'union__right_s__word_vectors__vectors',
                'union__left_s__word_vectors__vectors']
    @staticmethod
    def _get_tokeniser_names():
        '''
        Overideen

        :returns: A list of of parameter names where the tokenisers are set in \
        the pipeline.
        :rtype: list
        '''

        return ['union__dependency__tokens',
                'union__left__tokens',
                'union__right__tokens',
                'union__target__tokens',
                'union__right_s__tokens',
                'union__left_s__tokens']
    @staticmethod
    def _get_word_senti_names():
        '''
        :returns: A list of of parameter names where the sentiment lexicons are \
        set in the pipeline.
        :rtype: list
        '''

        return ['union__left_s__filter__lexicon',
                'union__right_s__filter__lexicon']
    def get_params(self, word_vector, parser, senti_lexicon, tokeniser=None,
                   token_lower=None, parser_lower=None, C=None,
                   random_state=None, scale=True, rel_depth=None):
        '''
        Method extended to include senti_lexicon
        '''
        params_dict = super().get_params(word_vector, parser=parser,
                                         tokeniser=tokeniser, C=C, scale=scale,
                                         token_lower=token_lower, rel_depth=rel_depth,
                                         parser_lower=parser_lower,
                                         random_state=random_state)

        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_word_senti_names(),
                                               senti_lexicon)
        return params_dict

    def get_cv_params(self, word_vectors, parsers, senti_lexicons, tokenisers=None,
                      token_lowers=None, C=None, scale=None, random_state=None,
                      parser_lowers=None, rel_depths=None):
        '''
        Method extended to include senti_lexicon
        '''
        params_list = super().get_cv_params(word_vectors, parsers=parsers,
                                            tokenisers=tokenisers, C=C, scale=scale,
                                            token_lowers=token_lowers,
                                            rel_depths=rel_depths,
                                            parser_lowers=parser_lowers,
                                            random_state=random_state)
        params_list = self._add_to_params(params_list, senti_lexicons,
                                          self._get_word_senti_names())
        return params_list

    def __repr__(self):
        return 'TDParse Plus'
