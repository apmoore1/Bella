from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVC

from tdparse.helper import read_config
from tdparse.parsers import dong

from tdparse.word_vectors import GensimVectors
from tdparse.tokenisers import ark_twokenize
from tdparse.neural_pooling import matrix_max, matrix_min, matrix_avg,\
matrix_median, matrix_prod, matrix_std

from tdparse.scikit_features.context import Context
from tdparse.scikit_features.tokeniser import ContextTokeniser
from tdparse.scikit_features.word_vector import ContextWordVectors
from tdparse.scikit_features.neural_pooling import NeuralPooling
from tdparse.scikit_features.join_context_vectors import JoinContextVectors

def train(data):
    if data == 'dong':
        vo_zhang_path = read_config('word2vec_files')['vo_zhang']
        vo_zhang = GensimVectors(vo_zhang_path, None, model='word2vec')
        train_data = read_config('dong_twit_train_data')
        train_data = dong(train_data)
        test_data = read_config('dong_twit_test_data')
        test_data = dong(test_data)

        union_parameters = {'svm__C' : [0.01]}

        union_pipeline = Pipeline([
            ('union', FeatureUnion([
                ('left', Pipeline([
                    ('contexts', Context({'l'})),
                    ('tokens', ContextTokeniser(ark_twokenize, False)),
                    ('word_vectors', ContextWordVectors(vo_zhang)),
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
                    ('tokens', ContextTokeniser(ark_twokenize, False)),
                    ('word_vectors', ContextWordVectors(vo_zhang)),
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
                    ('tokens', ContextTokeniser(ark_twokenize, False)),
                    ('word_vectors', ContextWordVectors(vo_zhang)),
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
                    ('contexts', Context({'f'})),
                    ('tokens', ContextTokeniser(ark_twokenize, False)),
                    ('word_vectors', ContextWordVectors(vo_zhang)),
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
            ('svm', LinearSVC(C=0.01))
        ])

        train_y_values = [target_dict['sentiment'] for target_dict in train_data]
        test_y_values = [target_dict['sentiment'] for target_dict in test_data]

        #grid_search = GridSearchCV(union_pipeline, param_grid=union_parameters,
        #                           cv=5, scoring='accuracy', n_jobs=1)
        #grid_clf = grid_search.fit(train_data, train_y_values)

        union_pipeline.fit(train_data, train_y_values)
        preds = union_pipeline.predict(test_data)

        import code
        code.interact(local=locals())


train('dong')
