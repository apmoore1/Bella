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
import os
from pathlib import Path
import pickle
import random as rn
import tempfile
from typing import Any, List, Dict, Union, Tuple

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.externals import joblib
import tensorflow as tf

import bella


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
    4. __repr__ -- Name of the machine learning model.

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

    @abstractmethod
    def __repr__(self) -> str:
        '''
        Name of the machine learning model.

        :return: Name of the machine learning model.
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


class KerasModel(BaseModel):

    @abstractmethod
    def keras_model(self, num_classes):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Tuple[np.ndarray, np.ndarray],
            verbose: int = 0,
            continue_training: bool = False) -> 'keras.callbacks.History':
        '''
        Fit the model according to the given training data.

        :param X: Training samples matrix, shape = [n_samples, n_features]
        :param y: Training targets, shape = [n_samples]
        :param validation_data: Tuple of `(x_val, y_val)`. Used to evaluate the
                                model at each epoch. Will not be trained on
                                this data.
        :param verbose: 0 = silent, 1 = progress
        :param continue_training: Whether the model that has already been
                                  trained should be trained further.
        :return: A record of training loss values and metrics values at
                 successive epochs, as well as validation loss values and
                 validation metrics values.
        '''
        X_val, y_val = validation_data
        if sum(y_val < 0) or sum(y < 0):
            raise ValueError('The class labels have to be greater than 0')
        X, X_val = self.create_training_text(X, X_val)
        y, y_val = self.create_training_y(y, y_val)
        num_classes = y.shape[1]
        if verbose:
            print(f'Number of classes in the data {num_classes}')

        if not continue_training:
            self.fitted = False
            self._to_be_reproducible(self.reproducible)
            self.model = self.keras_model(num_classes)
        elif self.fitted and not continue_training:
            raise ValueError('The model is already fitted')

        model = self.model
        if not continue_training:
            model.compile(optimizer=self.optimiser(**self.optimiser_params),
                          metrics=['accuracy'],
                          loss='categorical_crossentropy')

        with tempfile.NamedTemporaryFile() as weight_file:
            # Set up the callbacks
            model_checkpoint = ModelCheckpoint(weight_file.name,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='min')
            early_stopping = EarlyStopping(monitor='val_loss', mode='min',
                                           patience=self.patience)
            callbacks = [early_stopping, model_checkpoint]
            history = model.fit(X, y, validation_data=(X_val, y_val),
                                epochs=self.epochs, callbacks=callbacks,
                                verbose=verbose, batch_size=self.batch_size)
            # Load the best model from the saved weight file
            model.load_weights(weight_file.name)
        self.model = model
        self.fitted = True
        return history

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        '''
        The probability of each class label for all samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]]
        :return: Probability of each class label for all samples, shape = \
        [n_samples, n_classes]
        '''

        if self.fitted is False:
            raise ValueError('The model has not been fitted please run the '
                             '`fit` method.')
        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.
        sequence_test_data = self._pre_process(X, training=False)
        predicted_values = self.model.predict(sequence_test_data)
        return predicted_values

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict class labels for samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]
        :return: Predicted class label per sample, shape = [n_samples]
        '''

        return np.argmax(self.probabilities(X), axis=1)

    @staticmethod
    def save(model: 'bella.models.base.KerasModel', save_fp: Path) -> None:
        '''
        Given a Keras Model, mode, path to the folder to save too, and a name
        to save the files it will save the data to restore the model.

        :param model: The machine learning model instance to be saved.
        :param save_fp: File path of the location that the model is to be
                        saved.
        :return: Nothing.
        :raises ValueError: If the model has not been fitted or if the model
                            is not of type
                            :py:class:`bella.models.base.KerasModel`
        '''

        if not isinstance(model, KerasModel):
            raise ValueError('The model parameter has to be of type '
                             f'KearsModel not {type(model)}')
        if model.fitted:
            model_fp = save_fp.with_suffix('.h5')
            model.model.save(model_fp)

            attributes_fp = save_fp.with_suffix('.pkl')
            with attributes_fp.open('wb') as attributes_file:
                # optimiser cannot be pickled
                attributes = model.model_parameters()
                del attributes['class_params']['optimiser']
                pickle.dump(attributes, attributes_file)
        else:
            raise ValueError(f'The model {str(model)} has not been fitted. '
                             'This can be done by using the `fit` method')

    @staticmethod
    def load(load_fp: Path) -> 'bella.models.base.KerasModel':
        '''
        Loads an instance of this class from a file.

        :param load_fp: File path of the location that the model was saved to.
        :return: self
        '''

        model_fp = load_fp.with_suffix('.h5')
        attributes_fp = load_fp.with_suffix('.pkl')
        with attributes_fp.open('rb') as attributes_file:
            attributes = pickle.load(attributes_file)
        # optimiser has to be recovered as it could not be pickled in the
        # model parameters
        keras_model = keras.models.load_model(model_fp)
        attributes['class_params']['optimiser'] = keras_model.optimizer
        model_class = attributes.pop('class')
        model = model_class(**attributes['class_params'])
        for name, class_attr in attributes['class_attrs'].items():
            setattr(model, name, class_attr)
        model.model = keras_model
        model.fitted = True
        return model

    @staticmethod
    def _to_be_reproducible(reproducible: Union[int, None]) -> None:
        '''
        To make the method reproducible or not. If it is not needed then
        we can use all the python threads.

        :param reproducible: If int is provided this int is used as the seed
                             values. Else None should be given if it is not
                             to be reproducible.
        '''
        if reproducible is not None:
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(reproducible)
            rn.seed(reproducible)
            # Forces tensorflow to use only one thread
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1)
            tf.set_random_seed(reproducible)

            sess = tf.Session(graph=tf.get_default_graph(),
                              config=session_conf)
            keras.backend.set_session(sess)
        else:
            np.random.seed(None)
            rn.seed(np.random.randint(0, 1000))
            tf.set_random_seed(np.random.randint(0, 1000))


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
    4. __repr__ -- Name of the machine learning model.

    Functions:

    1. save -- Given a instance of this class will save it to a file.
    2. load -- Loads an instance of this class from a file.

    Abstract Functions:

    1. Pipeline -- Machine Learning model that is used as the base template
       for the model attribute. Expects it to be a
       :py:class:`sklearn.pipeline.Pipeline` instance.
    '''

    def __init__(self, *args, **kwargs) -> None:

        self.model = self.pipeline()
        self.fitted = False
        self._model_parameters = self.get_parameters(*args, **kwargs)
        self.model.set_params(**self._model_parameters)

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
    def save(model: 'bella.models.base.SKLearnModel',
             save_fp: Path, compress: int = 0) -> None:
        '''
        Given an instance of this class will save it to a file.

        :param model: The machine learning model instance to be saved.
        :param save_fp: File path of the location that the model is to be
                        saved to.
        :param compress: Optional (default 0). Level of compression 0 is no
                         compression and 9 is the most compressed. The more
                         compressed the lower the read/write time.
        :return: Nothing.
        :raises ValueError: If the model has not been fitted or if the model
                            is not of type
                            :py:class:`bella.models.base.SKLearn`
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

    @staticmethod
    @abstractmethod
    def pipeline() -> 'sklearn.pipeline.Pipeline':
        '''
        Machine Learning model that is used as the base template for the model
        attribute.

        :returns: The template machine learning model
        '''
        pass
