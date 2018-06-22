'''
Module contains all of the classes that are either abstract or Mixin that \
are used in the machine learning models.

Abstract classes:

1. :py:class:`bella.models.base.BaseModel`

Mixin classes:

1. :py:class:`bella.models.base.SKLearnModel`
'''

from abc import ABC, abstractmethod
from typing import Any, List, Callable
from pathlib import Path

import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

import bella
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
    def load(load_fp: Path) -> 'bella.models.base.BaseModel':
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
    '''
    Concrete class that is designed to be used as a Mixin for all machine \
    learning models that are based on the \
    `scikit learn library <http://scikit-learn.org/stable/>`_.

    At the moment expects all of the machine learning models to use a \
    `SVM <http://scikit-learn.org/0.19/modules/cla\
    sses.html#module-sklearn.svm>`_ as their classifier. This is due to \
    assuming the model will have \
    the method :py:meth:`sklearn.svm.SVC.decision_function` to get \
    `probabilities`.

    Attributes:

    1. model -- Machine learning model. Expects it to be a \
    :py:class:`sklearn.pipeline.Pipeline` instance.
    2. fitted -- If the machine learning model has been fitted (default False)

    Methods:

    1. fit -- Fit the model according to the given training data.
    2. predict -- Predict class labels for samples in X.
    3. probabilities -- The probability of each class label for all samples \
    in X.

    Functions:

    1. save -- Given a instance of this class will save it to a file.
    2. load -- Loads an instance of this class from a file.
    '''

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Fit the model according to the given training data.

        :param X: Training samples matrix, shape = [n_samples, n_features]
        :param y: Training targets, shape = [n_samples]
        :return: The `model` attribute will now be trained.
        '''

        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray):
        '''
        Predict class labels for samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]
        :return: Predicted class label per sample, shape = [n_samples]
        :raises ValueError: If the model has not been fitted
        '''

        if self.fitted:
            return self.model.predict(X)
        raise ValueError(f'The model {str(self)} has not been fitted. '
                         'This can be done by using the `fit` method')

    def probabilities(self, X: np.ndarray):
        '''
        The probability of each class label for all samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]]
        :return: Probability of each class label for all samples, shape = \
        [n_samples, n_classes]
        :raises ValueError: If the model has not been fitted
        '''
        if self.fitted:
            return self.model.decision_function(X)
        raise ValueError(f'The model {str(self)} has not been fitted. '
                         'This can be done by using the `fit` method')

    @staticmethod
    def save(model: 'bella.models.base.SKLearnModel',
             save_fp: Path, compress: int = 0) -> None:
        '''
        Given a instance of this class will save it to a file.

        :param model: The machine learning model instance to be saved.
        :param save_fp: File path of the location that the model is to be \
        saved to.
        :param compress: Optional (default 0). Level of compression 0 is no \
        compression and 9 is the most compressed. The more compressed the \
        slower the read/write time.
        :return: Nothing.
        :raises ValueError: If the model has not been fitted or if the model \
        is not of type 'bella.models.base.SKLearn'
        '''

        if not isinstance(model, SKLearnModel):
            raise ValueError('The model parameter has to be of type '
                             f'SKLearnModel not {type(model)}')
        if model.fitted:
            joblib.dump(model, save_fp, compress=compress)
        raise ValueError(f'The model {str(model)} has not been fitted. '
                         'This can be done by using the `fit` method')

    @staticmethod
    def load(load_fp: Path) -> 'bella.models.base.SKLearnModel':
        '''
        Given a instance of this class will save it to a file.

        :param load_fp: File path of the location that the model was saved to.
        :return: self
        '''

        return joblib.load(load_fp)


class TargetInd(SKLearnModel):
    '''
    Class that has changed yes another
    '''

    def __init__(self, word_vectors: List['bella.word_vectors.WordVectors'],
                 tokeniser: Callable[[str], List[str]] = ark_twokenize,
                 lower: bool = True, C: float = 0.01,
                 random_state: int = 42,
                 scale: Any = MinMaxScaler()
                 ) -> None:
        '''
        :param word_vectors: A list of one or more word vectors to be used as \
        feature vector lookups. If more than one is used the word vectors \
        are concatenated together to create a the feature vector for each word.
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`
        :param lower: Wether to lower case the words
        :param C: The C value for the :py:class:`sklearn.svm.SVC` estimator \
        that is used in the pipeline.
        :param random_state: The random_state value for the \
        :py:class:`sklearn.svm.SVC` estimator that is used in the pipeline.
        :param scale: How to scale the data before input into the estimator. \
        If no scaling is to be used set this to None.
        :return: Nothing
        '''

        self._fitted = False
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
        # Add word vectors to Pipeline model
        params_dict = self._add_to_params_dict(params_dict,
                                               self._get_word_vector_names(),
                                               word_vectors)
        # Add tokenisers to Pipeline model
        tokenisers_names = [param_name + '__tokeniser'
                            for param_name in self._get_tokeniser_names()]
        params_dict = self._add_to_params_dict(params_dict,
                                               tokenisers_names,
                                               tokeniser)
        # Add if the words should be lower cased
        lower_names = [param_name + '__lower'
                       for param_name in self._get_tokeniser_names()]
        params_dict = self._add_to_params_dict(params_dict, lower_names, lower)
        # Add how the data should be scaled before going into the SVM
        # If None then it means no scaling happens
        params_dict = self._add_to_params_dict(params_dict, ['scale'],
                                               scale)
        # Add the C value for the SVM
        params_dict = self._add_to_params_dict(params_dict, ['svm__C'], C)
        # Add the random state for the SVM
        params_dict = self._add_to_params_dict(params_dict,
                                               ['svm__random_state'],
                                               random_state)
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
            raise ValueError('The keys parameter has to be of type list and '
                             f'not {type(keys)}')
        for key in keys:
            params_dict[key] = value
        return params_dict
