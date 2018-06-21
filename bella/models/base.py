'''
Module contains all of the classes that are either abstract or Mixin that \
are used in the machine learning models.

Abstract classes:

1. :py:class:`bella.models.base.BaseModel`
'''

from abc import ABC, abstractmethod
from typing import Any, List, Callable
from pathlib import Path

import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from bella.tokenisers import ark_twokenize
from bella.neural_pooling import matrix_max, matrix_min, matrix_avg
from bella.neural_pooling import matrix_median, matrix_prod, matrix_std
from bella.scikit_features.context import Context
from bella.scikit_features.tokeniser import ContextTokeniser
from bella.scikit_features.word_vector import ContextWordVectors
from bella.scikit_features.neural_pooling import NeuralPooling
from bella.scikit_features.join_context_vectors import JoinContextVectors


class BaseModel(ABC):
    '''
    Abstract class for all of the machine learning models.

    Attributes:

    1. model -- Object that contains the machine learning model
    2. fitted -- If the machine learning model has been fitted (default False)

    Methods:

    1. fit -- Fit the model according to the given training data.
    2. predict -- Predict class labels for samples in X.
    3. probabilities -- The probability of each class label for all samples \
    in X.

    Functions:

    1. save -- Saves the given machine learning model instance to a file.
    2. load -- Loads the entire machine learning model from a file.
    '''

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the model according to the given training data.

        :param X: Training samples matrix, shape = [n_samples, n_features]
        :param y: Training targets, shape = [n_samples]
        :return: The `model` attribute will now be trained.
        '''
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict class labels for samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]
        :return: Predicted class label per sample, shape = [n_samples]
        '''
        pass

    @abstractmethod
    def probabilities(self, X: np.ndarray) -> np.ndarray:
        '''
        The probability of each class label for all samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]]
        :return: Probability of each class label for all samples, shape = \
        [n_samples, n_classes]
        '''
        pass

    @staticmethod
    @abstractmethod
    def save(model: 'BaseModel', save_fp: Path) -> None:
        '''
        Saves the entire machine learning model to a file.

        :param model: The machine learning model instance to be saved.
        :param save_fp: File path of the location that the model is to be \
        saved to.
        :return: Nothing.
        '''
        pass

    @staticmethod
    @abstractmethod
    def load(load_fp: Path) -> 'BaseModel':
        '''
        Loads the entire machine learning model from a file.
        :param load_fp: File path of the location that the model was saved to.
        :return: self
        '''
        pass

    @property
    def model(self) -> Any:
        '''
        :return: The machine learning model
        '''
        return self._model

    @model.setter
    def model(self, value) -> None:
        '''
        Sets the model attribute

        :param value: The value to assign to the model attribute
        '''
        self._model = value

    @property
    def fitted(self) -> bool:
        '''
        If the machine learning model has been fitted (default False)

        :return: True or False
        '''
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool) -> None:
        '''
        Sets the fitted attribute

        :param value: The value to assign to the fitted attribute
        '''

        self._fitted = value


class SKLearnModel(BaseModel):

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        return self.model.predict(X)

    def probabilities(self, X):
        return self.model.decision_function(X)

    @staticmethod
    def save(model, save_fp):
        joblib.dump(model.model, save_fp)

    @staticmethod
    def load(load_fp):
        return joblib.load(load_fp)


class TargetInd(SKLearnModel):

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()
                 ):
        super().__init__()
        self._model = Pipeline([
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
        params_dict = {}
        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_word_vector_names(),
                                               word_vectors)
        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_tokeniser_names(),
                                               tokeniser)
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
        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_word_vector_names(),
                                               word_vectors)
        self.model.set_params(**params_dict)

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
