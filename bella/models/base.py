'''
Module contains all of the classes that are either abstract or Mixin that \
are used in the machine learning models.

Abstract classes:

1. :py:class:`bella.models.base.BaseModel`

Mixin classes:

1. :py:class:`bella.models.base.SKLearnModel`
'''

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
from typing import Any, List, Callable, Dict, Union
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
from bella.scikit_features.lexicon_filter import LexiconFilter
from bella.scikit_features.join_context_vectors import JoinContextVectors
from bella.scikit_features.neural_pooling import NeuralPooling
from bella.scikit_features.tokeniser import ContextTokeniser
from bella.scikit_features.word_vector import ContextWordVectors


class BaseModel(ABC):
    '''
    Abstract class for all of the machine learning models.

    Attributes:

    1. model -- Machine learning model that is associated to this instance.
    2. fitted -- If the machine learning model has been fitted (default False)

    Methods:

    1. fit -- Fit the model according to the given training data.
    2. predict -- Predict class labels for samples in X.
    3. probabilities -- The probability of each class label for all samples
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
        Machine learning model that is associated to this instance.

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
    Concrete class that is designed to be used as a Mixin for all machine
    learning models that are based on the
    `scikit learn library <http://scikit-learn.org/stable/>`_.

    At the moment expects all of the machine learning models to use a
    `SVM <http://scikit-learn.org/0.19/modules/cla\
    sses.html#module-sklearn.svm>`_ as their classifier. This is due to
    assuming the model will have the method
    :py:meth:`sklearn.svm.SVC.decision_function` to get `probabilities`.

    **NOTE** each time the *model_parameters* are set it resets the model
    i.e. the *fitted* attribute is :py:class:`False`

    Attributes:

    1. model -- Machine learning model. Expects it to be a
       :py:class:`sklearn.pipeline.Pipeline` instance.
    2. fitted -- If the machine learning model has been fitted (default False)
    3. model_parameters -- The parameters that are set in the machine
       learning model. E.g. Parameter could be the tokeniser used.

    Abstract Class Methods:

    1. get_parameters -- Transform the given parameters into a dictonary
       that is accepted as model parameters.
    2. get_cv_parameters -- Transform the given parameters into a list of
       dictonaries that is accepted as `param_grid` parameter in
       :py:class:`sklearn.model_selection.GridSearchCV`

    Methods:

    1. fit -- Fit the model according to the given training data.
    2. predict -- Predict class labels for samples in X.
    3. probabilities -- The probability of each class label for all samples
       in X.

    Functions:

    1. save -- Given a instance of this class will save it to a file.
    2. load -- Loads an instance of this class from a file.

    Abstract Functions:

    1. pipeline -- An instance :py:class:`sklearn.pipeline.Pipeline`
       that represents the machine learning model of this class. Used in the
       __init__ to set the model parameter.
    '''

    def __init__(self, *args, **kwargs) -> None:

        self.fitted = False
        self.model = self.pipeline()
        self._model_parameters = self.get_parameters(*args, **kwargs)
        self.model.set_params(**self._model_parameters)
        self.fitted = False

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
        :return: Probability of each class label for all samples, shape =
                 [n_samples, n_classes]
        :raises ValueError: If the model has not been fitted
        '''

        if self.fitted:
            return self.model.decision_function(X)
        raise ValueError(f'The model {str(self)} has not been fitted. '
                         'This can be done by using the `fit` method')

    @property
    def model_parameters(self) -> Dict[str, Any]:
        '''
        The parameters that are set in the machine learning model. E.g.
        Parameter could be the tokeniser used.

        :return: parameters of the machine learning model
        '''
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, value: Dict[str, Any]) -> None:
        '''
        Set the parameters of the machine learning model.

        :param value: The new parameters of the machine learning model
        '''
        self._model_parameters = self.get_parameters(**value)
        self.model.set_params(**self._model_parameters)
        self.fitted = False

    @staticmethod
    @abstractmethod
    def pipeline() -> Pipeline:
        '''
        An instance :py:class:`sklearn.pipeline.Pipeline` that represents the
        machine learning model of this class. Used in the __init__ to set
        the model parameter.

        The pipeline instance may not be used in some cases stright away as
        some of its transformers and estimator may not have parameters set.
        E.g. which tokeniser to use.
        '''
        pass

    @staticmethod
    def save(model: 'bella.models.base.SKLearnModel',
             save_fp: Path, compress: int = 0) -> None:
        '''
        Given a instance of this class will save it to a file.

        :param model: The machine learning model instance to be saved.
        :param save_fp: File path of the location that the model is to be
                        saved to.
        :param compress: Optional (default 0). Level of compression 0 is no
                         compression and 9 is the most compressed. The more
                         compressed the lower the read/write time.
        :return: Nothing.
        :raises ValueError: If the model has not been fitted or if the model
                            is not of type 'bella.models.base.SKLearn'
        '''

        if not isinstance(model, SKLearnModel):
            raise ValueError('The model parameter has to be of type '
                             f'SKLearnModel not {type(model)}')
        if model.fitted:
            joblib.dump(model, save_fp, compress=compress)
        else:
            raise ValueError(f'The model {str(model)} has not been fitted. '
                             'This can be done by using the `fit` method')

    @staticmethod
    def load(load_fp: Path) -> 'bella.models.base.SKLearnModel':
        '''
        Loads an instance of this class from a file.

        :param load_fp: File path of the location that the model was saved to.
        :return: self
        '''

        return joblib.load(load_fp)

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> Dict[str, Any]:
        '''
        Transform the given parameters into a dictonary that is accepted as
        model parameters
        '''

        pass

    @classmethod
    @abstractmethod
    def get_cv_parameters(cls) -> List[Dict[str, List[Any]]]:
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`
        '''

        pass

    @staticmethod
    def _add_to_params_dict(params_dict: Dict[str, Any], keys: List[str],
                            value: Any) -> Dict[str, Any]:
        '''
        Given a dictionary it adds the value to each key in the list of keys
        into the dictionary. Returns the updated dictionary.

        Normally used in subclasses :py:meth:`get_parameters`

        :param params_dict: Dictionary to be updated
        :param keys: list of keys
        :param value: value to be added to each key in the list of keys.
        :returns: The dictionary updated
        '''

        if not isinstance(keys, list):
            raise ValueError('The keys parameter has to be of type list and '
                             f'not {type(keys)}')
        for key in keys:
            params_dict[key] = value
        return params_dict

    @staticmethod
    def _add_to_params(params_list: Union[List[Dict[str, List[Any]]], List],
                       to_add: List[Any],
                       to_add_names: List[str]) -> List[Dict[str, List[Any]]]:
        '''
        Used to add parameters that are stated multiple times in the same
        pipeline that must have the same value.

        Therefore to add them you have to copy the current parameter
        list N amount of times where N is the length of the to_add list.
        Returns the updated parameter list. Method to add parameters that
        are set in multiple parts of the pipeline but should contain the
        same value.

        Normally used in subclasses :py:meth:`get_cv_parameters`

        :params_list: A list of dicts where each dict contains parameters and
                      corresponding values that are to be searched for. Can be
                      an empty List.
        :param to_add: List of values that are to be added to the search space.
        :param to_add_names: List of names that are associated to the values.
        :returns: The updated params_list
        :raises TypeError: If any of the arguments are not of type
                           :py:class:`List`
        '''
        # Check the type of the argument
        if not isinstance(params_list, list):
            raise TypeError(f'params_list: {params_list}\nShould be of type '
                            f'list not {type(params_list)}')
        if not isinstance(to_add_names, list):
            raise TypeError(f'to_add_names: {to_add_names}\nShould be of type '
                            f'list not {type(to_add_names)}')
        param_name = to_add_names[0]
        if len(to_add_names) > 1:
            param_name = ''.join(param_name.split('__')[:-1])
        if not isinstance(to_add, list):
            raise TypeError('If using get_cv_parameters this is due to '
                            f'parameter {param_name} not being of type list.'
                            f'\nto_add: {to_add} should be of '
                            f'type List not {type(to_add)}.')

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
    def _add_to_all_params(params_list: List[Dict[str, List[Any]]],
                           param_name: str, param_value: List[Any]
                           ) -> List[Dict[str, List[Any]]]:
        '''
        Used to add param_name and its associated param_value to each
        dictionary of parameters in the params_list.

        Normally used in subclasses :py:meth:`get_cv_parameters`

        :param params_list: A list of dicts where each dict contains
                            parameters and corresponding values that are to be
                            searched for.
        :param param_name: The name associated to the parameter value to be
                           added to the params_list.
        :param param_value: The list of values associated to the param_name
                            that are added to the params_list.
        :returns: The updated params_list
        :raises TypeError: If the param_value is not of type :py:class:`List`
        '''

        if not isinstance(param_value, list):
            raise TypeError(f'{param_name} should be of type list not '
                            f'{type(param_value)}')
        for param_dict in params_list:
            param_dict[param_name] = param_value
        return params_list


class TargetInd(SKLearnModel):
    '''
    Target-ind model from `Vo and Zhang 2015 paper
    <https://ijcai.org/Proceedings/15/Papers/194.pdf>`_.
    '''

    @classmethod
    def get_cv_parameters(cls, word_vectors, tokeniser=None, lower=None,
                          C=None, scale=None, random_state=None):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]]
        :param tokenisers: A list of tokeniser to be used e.g.
                           :py:meth:`str.split`. Default None, only use default
                           tokeniser.
        :param lowers: A list of bool values which indicate whether to lower
                       case the input words. Default None, only use default
                       setting(True).
        :param C: A list of C values for the :py:class:`sklearn.svm.SVC`
                  estimator that is used in the pipeline. Default None, only
                  use default setting(C=0.01).
        :param random_state: A list of random_state values for the
                             :py:class:`sklearn.svm.SVC` estimator that is
                             used in the pipeline. Default None, only use
                             default setting(random_state=42).
        :param scale: List of scale values. The list can include
                      :py:class:`sklearn.preprocessing.MinMaxScaler` type of
                      clases or None if no scaling is to be used. Default None,
                      only use default setting
                      (:py:class:`sklearn.preprocessing.MinMaxScaler`).
        :return: Parameters to explore through cross validation
        '''
        params_list = []
        params_list = cls._add_to_params(params_list, word_vectors,
                                         cls._get_word_vector_names())
        if tokeniser is not None:
            tokenisers_names = [param_name + '__tokeniser'
                                for param_name in cls._get_tokeniser_names()]
            params_list = cls._add_to_params(params_list, tokeniser,
                                             tokenisers_names)
        if lower is not None:
            lower_names = [param_name + '__lower'
                           for param_name in cls._get_tokeniser_names()]
            params_list = cls._add_to_params(params_list, lower, lower_names)
        if C is not None:
            params_list = cls._add_to_all_params(params_list, 'svm__C', C)
        if random_state is not None:
            params_list = cls._add_to_all_params(params_list,
                                                 'svm__random_state',
                                                 random_state)
        if scale is not None:
            params_list = cls._add_to_all_params(params_list, 'scale', scale)
        return params_list

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

    @staticmethod
    def pipeline() -> Pipeline:
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
    '''
    Target-dep- model from `Vo and Zhang 2015 paper
    <https://ijcai.org/Proceedings/15/Papers/194.pdf>`_.
    '''

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

    @staticmethod
    def pipeline():
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

    @staticmethod
    def pipeline() -> Pipeline:
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
    '''
    Target-dep+ model from `Vo and Zhang 2015 paper
    <https://ijcai.org/Proceedings/15/Papers/194.pdf>`_.
    '''

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
                          tokeniser=None, lower=None,
                          C=None, scale=None, random_state=None):
        '''
        Transform the given parameters into a list of dictonaries that is
        accepted as `param_grid` parameter in
        :py:class:`sklearn.model_selection.GridSearchCV`

        :param word_vectors: A list of a list of word vectors e.g. [[SSWE()],
                             [SSWE(), GloveCommonCrawl()]]
        :param senti_lexicon: A list of Sentiment Lexicons to be explored for
                              the Left and Right sentiment context (LS and RS).
        :param tokenisers: A list of tokeniser to be used
                           e.g. :py:meth:`str.split`. Default None, only use
                           default tokeniser.
        :param lowers: A list of bool values which indicate whether to lower
                       case the input words. Default None, only use default
                       setting(True).
        :param C: A list of C values for the :py:class:`sklearn.svm.SVC`
                  estimator that is used in the pipeline. Default None, only
                  use default setting(C=0.01).
        :param random_state: A list of random_state values for the
                             :py:class:`sklearn.svm.SVC` estimator that is used
                             in the pipeline. Default None, only use default
                             setting(random_state=42).
        :param scale: List of scale values. The list can include
                      :py:class:`sklearn.preprocessing.MinMaxScaler` type of
                      clases or None if no scaling is to be used. Default None,
                      only use default setting
                      (:py:class:`sklearn.preprocessing.MinMaxScaler`).
        :return: Parameters to explore through cross validation
        '''
        params_list = super().get_cv_parameters(word_vectors, tokeniser,
                                                lower, C, scale, random_state)
        params_list = cls._add_to_params(params_list, senti_lexicon,
                                         cls._get_word_senti_names())
        return params_list

    @staticmethod
    def pipeline() -> Pipeline:
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
