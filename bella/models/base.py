'''
Module contains all of the main base classes for the machine learning models
these are grouped into 3 categories; 1. Mixin, 2. Abstract, and 3. Concrete.


Mixin classes - This is a function based class that contains functions that
do not rely on the type of model and are useful for all:

1. :py:class:`bella.models.base.ModelMixin`

Abstract classes - This is used to enforce all the functions that all
the machine learning models must have. This is also the class that inherits
the Mixin class:

1. :py:class:`bella.models.base.BaseModel`

Concrete classes - These are more concete classes that still contain some
abstract methods. However they are the classes to inherit from to create a
machine learning model base on a certain framework e.g. SKlearn or Keras:

1. :py:class:`bella.models.base.SKLearnModel`
2. :py:class:`bella.models.base.KerasModel`
'''

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
import os
from pathlib import Path
import pickle
import random as rn
import tempfile
from typing import Any, List, Dict, Union, Tuple, Callable
from multiprocessing.pool import Pool

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import preprocessing
import numpy as np
import pandas as pd
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import tensorflow as tf

import bella
from bella.data_types import TargetCollection, Target


class ModelMixin():
    '''
    Mixin class for all of the machine learning models. Contain functions
    only so they are as generic as possible.

    Functions:

    1. train_val_split -- Splits the training dataset into a train and
       validation set in a stratified split.
    '''

    @staticmethod
    def _convert_to_targets(data: List[Dict[str, Any]]
                            ) -> List['bella.data_types.Target']:
        '''
        Converts a list of dictionaries into a list of
        :py:class:`bella.data_types.Target`.
        '''

        all_targets = []
        for target in data:
            all_targets.append(Target(**target))
        return all_targets

    @staticmethod
    def train_val_split(train: 'TargetCollection',
                        split_size: float = 0.2, seed: Union[None, int] = 42
                        ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                   Tuple[np.ndarray, np.ndarray]]:
        '''
        Splits the training dataset into a train and validation set in a
        stratified split.

        :param train: The training dataset that needs to be split into
        :param split_size: Fraction of the dataset to assign to the
                           validation set.
        :param seed: Seed value to give to the stratified splitter. If
                     None then it uses the radnom state of numpy.
        :return: Two tuples of length two where each tuple is the train
                 and validation splits respectively, and each tuple contains
                 the data (X) and class labels (y) respectively. Returns
                 ((X_train, y_train), (X_val, y_val))
        '''
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=split_size,
                                          random_state=seed)
        data = np.asarray(train.data_dict())
        sentiment = np.asarray(train.sentiment_data())
        for train_indexs, test_indexs in splitter.split(data, sentiment):
            train_data = data[train_indexs]
            test_data = data[test_indexs]

        train = TargetCollection(ModelMixin._convert_to_targets(train_data))
        val = TargetCollection(ModelMixin._convert_to_targets(test_data))

        X_train = np.array(train.data_dict())
        y_train = np.array(train.sentiment_data())
        X_val = np.array(val.data_dict())
        y_val = np.array(val.sentiment_data())
        return (X_train, y_train), (X_val, y_val)


class BaseModel(ModelMixin, ABC):
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

    Class Methods:

    1. name -- -- Returns the name of the model.

    Functions:

    1. save -- Saves the given machine learning model instance to a file.
    2. load -- Loads the entire machine learning model from a file.
    3. evaluate_parameter -- fit and predict given training, validation and
       test data the given model when the given parameter is changed on the
       model.
    4. evaluate_parameters -- same as evaluate_parameter however it
       evaluates over many parameter values for the same parameter.
    '''

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the model according to the given training data.

        :param X: Training samples matrix, shape = [n_samples, n_features]
        :param y: Training targets, shape = [n_samples]
        :return: The `model` attribute will now be trained.
        '''

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict class labels for samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]
        :return: Predicted class label per sample, shape = [n_samples]
        '''

    @abstractmethod
    def probabilities(self, X: np.ndarray) -> np.ndarray:
        '''
        The probability of each class label for all samples in X.

        :param X: Test samples matrix, shape = [n_samples, n_features]]
        :return: Probability of each class label for all samples, shape = \
        [n_samples, n_classes]
        '''

    @abstractmethod
    def __repr__(self) -> str:
        '''
        Name of the machine learning model.

        :return: Name of the machine learning model.
        '''

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

    @staticmethod
    @abstractmethod
    def load(load_fp: Path) -> 'bella.models.base.BaseModel':
        '''
        Loads the entire machine learning model from a file.

        :param load_fp: File path of the location that the model was saved to.
        :return: self
        '''

    @staticmethod
    @abstractmethod
    def evaluate_parameter(model: 'BaseModel',
                           train: Tuple[np.ndarray, np.ndarray],
                           val: Union[None, Tuple[np.ndarray, np.ndarray]],
                           test: np.ndarray, parameter_name: str,
                           parameter: Any) -> Tuple[Any, np.ndarray]:
        '''
        Given a model will set the `parameter_name` to `parameter` fit the
        model and return the a Tuple of parameter changed and predictions of
        the model on the test data, using the train and validation data for
        fitting.

        :param model: :py:class:`bella.models.base.BaseModel` instance
        :param train: Tuple of `(X_train, y_train)`. Used to fit the model.
        :param val: Tuple of `(X_val, y_val)` or None is not required.
                    This is only required if the model requires validation
                    data like the :py:class:`bella.models.base.KerasModel`
                    models do.
        :param test: `X_test` data to predict on.
        :param parameter_name: Name of the parameter to change e.g. optimiser
        :param parameter: value to assign to the parameter e.g.
                          :py:class:`keras.optimizers.RMSprop`
        :return: A tuple of (parameter value, predictions)
        '''

    @staticmethod
    @abstractmethod
    def evaluate_parameters(model: 'bella.models.base.BaseModel',
                            train: Tuple[np.ndarray, np.ndarray],
                            val: Union[None, Tuple[np.ndarray, np.ndarray]],
                            test: np.ndarray, parameter_name: str,
                            parameters: List[Any], n_jobs: int
                            ) -> List[Tuple[Any, np.ndarray]]:
        '''
        Performs :py:func:`bella.models.base.BaseModel.evaluate_parameter` on
        one `parameter_name` but with multiple parameter values.

        This is useful if you would like to know the affect of changing the
        values of a parameter. It can also perform the task in a
        multiprocessing manner if `n_jobs` > 1.

        :param model: :py:class:`bella.models.base.BaseModel` instance
        :param train: Tuple of `(X_train, y_train)`. Used to fit the model.
        :param val: Tuple of `(X_val, y_val)` or None is not required.
                    This is only required if the model requires validation
                    data like the :py:class:`bella.models.base.KerasModel`
                    models do.
        :param test: `X_test` data to predict on.
        :param parameter_name: Name of the parameter to change e.g. optimiser
        :param parameters: A list of values to assign to the parameter e.g.
                           [:py:class:`keras.optimizers.RMSprop`]
        :param n_jobs: Number of cpus to use for multiprocessing if 1 then
                       will not multiprocess.
        :return: A list of tuples of (parameter value, predictions)
        '''

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        '''
        Returns the name of the model.

        :return: Name of the model
        '''

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
    '''
    Concrete class that is designed to be used as the base class for all
    machine learning models that are based on the
    `Keras library <https://keras.io>`_.

    Attributes:

    1. tokeniser -- Tokeniser model uses e.g. :py:meth:`str.split`.
    2. embeddings -- the word embeddings the model uses. e.g.
       :py:class:`bella.word_vectors.SSWE`
    3. lower -- if the model lower cases the words when pre-processing the data
    4. reproducible -- Whether to be reproducible. If None then it is quicker
       to run. Else provide a `int` that will represent the random seed value.
    5. patience -- Number of epochs with no improvement before training
       is stopped.
    6. batch_size -- Number of samples per gradient update.
    7. epcohs -- Number of times to train over the entire training set
       before stopping.
    8. optimiser -- Optimiser the model uses.
       e.g. :py:class:`keras.optimizers.SGD`
    9. optimiser_params -- Parameters for the optimiser. If None uses default
       for the optimiser being used.

    Abstract Methods:

    1. keras_model -- Keras machine Learning model that represents the class
       e.g. single forward LSTM.
    2. create_training_text -- Converts the training and validation data into
       a format that the keras model can take as input.
    3. create_training_y -- Converts the training and validation targets into a
       format that can be used by the keras model.

    Methods:

    1. fit -- Fit the model according to the given training and validation
       data.
    2. probabilities -- The probability of each class label for all samples
       in X.
    3. predict -- Predict class labels for samples in X.

    Functions:

    1. save -- Given a instance of this class will save it to a file.
    2. load -- Loads an instance of this class from a file.
    3. evaluate_parameter -- fit and predict given training, validation and
       test data the given model when the given parameter is changed on the
       model.
    4. evaluate_parameters -- same as evaluate_parameter however it
       evaluates over many parameter values for the same parameter.
    '''

    @abstractmethod
    def keras_model(self, num_classes: int) -> 'keras.models.Model':
        '''
        Keras machine Learning model that represents the class e.g.
        single forward LSTM.

        :returns: Keras machine learning model
        '''
        pass

    @abstractmethod
    def create_training_text(self, train_data: List[Dict[str, Any]],
                             validation_data: List[Dict[str, Any]]
                             ) -> Tuple[Any, Any]:
        '''
        Converts the training and validation data into a format that the keras
        model can take as input.

        :return: A tuple of length two containing the keras model training and
                 validation input respectively.
        '''

    @abstractmethod
    def create_training_y(self, train_y: np.ndarray, validation_y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Converts the training and validation targets into a format that can
        be used by the keras model

        :return: A tuple of length containing two array the first for
                 training and the second for validation.
        '''

    @abstractmethod
    def _pre_process(self, data_dicts: Dict[str, Any], training: bool):
        '''
        Converts the training or validation data into a format that will be
        used by the keras model.

        This function is normally used to process the training and the
        validation to be returned together by
        :py:meth:`bella.models.base.KerasModel.create_training_text`
        '''

    def process_text(self, texts: List[str], max_length: int,
                     padding: str = 'pre', truncate: str = 'pre'
                     ) -> Tuple[int, np.ndarray]:
        '''
        Given a list of Strings, tokenised the text and lower case if set and
        then convert the tokens into a integers representing the tokens in the
        embeddings. Lastly it pads the data based on the max_length param.

        If the max_length is smaller than the sentences size it truncates the
        sentence. If max_length = -1 then the max_length is that of the longest
        sentence in the texts.

        :params texts: List of texts
        :params max_length: How many tokens a sentence can contain. If it is
                            -1 then it uses the sentence with the most tokens
                            as the max_length parameter.
        :params padding: Which side of the sentence to pad: `pre` beginning,
                         `post` end.
        :params truncate: Which side of the sentence to truncate: `pre`
                          beginning `post` end.
        :returns: A tuple of length 2 containg: 1. The max_length parameter,
                  2. A matrix of shape [n_samples, pad_size] where each integer
                  in the matrix represents the word embedding lookup.
        :raises ValueError: If the mex_length argument is equal to or less
                            than 0. Or if the calculated max_length is 0.
        '''

        if max_length == 0:
            raise ValueError('The max length of a sequence cannot be zero')
        elif max_length < -1:
            raise ValueError('The max length has to be either -1 or above '
                             f'zero not {max_length}')

        # Process the text into integers based on the embeddings given
        all_sequence_data = []
        max_sequence = 0
        for text in texts:
            sequence_data = []
            tokens = self.tokeniser(text)
            for token in tokens:
                if self.lower:
                    token = token.lower()
                # If the token does not exist it should lookup the unknown
                # word vector
                sequence_data.append(self.embeddings.word2index[token])
            sequence_length = len(sequence_data)
            if sequence_length > max_sequence:
                max_sequence = sequence_length
            all_sequence_data.append(sequence_data)
        if max_sequence == 0:
            raise ValueError('The max sequence length is 0 suggesting no '
                             'data was provided for training or testing')
        # Pad the sequences
        # If max pad size is set and training the model set the
        # test_pad_size to max sequence length
        if max_length == -1:
            max_length = max_sequence
        return (max_length,
                preprocessing.sequence.pad_sequences(all_sequence_data,
                                                     maxlen=max_length,
                                                     dtype='int32',
                                                     value=0,
                                                     padding=padding,
                                                     truncating=truncate))

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Tuple[np.ndarray, np.ndarray],
            verbose: int = 0,
            continue_training: bool = False) -> 'keras.callbacks.History':
        '''
        Fit the model according to the given training and validation data.

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
        if isinstance(X, tuple):
            X = list(X)
            X_val = list(X_val)

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
    def evaluate_parameter(model: 'bella.models.base.KerasModel',
                           train: Tuple[np.ndarray, np.ndarray],
                           val: Tuple[np.ndarray, np.ndarray],
                           test: np.ndarray, parameter_name: str,
                           parameter: Any) -> Tuple[Any, np.ndarray]:
        '''
        Given a model will set the `parameter_name` to `parameter` fit the
        model and return the a Tuple of parameter changed and predictions of
        the model on the test data, using the train and validation data for
        fitting.

        :param model: KerasModel instance
        :param train: Tuple of `(X_train, y_train)`. Used to fit the model.
        :param val: Tuple of `(X_val, y_val)`. Used to evaluate the
                    model at each epoch. Will not be trained on
                    this data.
        :param test: `X_test` data to predict on.
        :param parameter_name: Name of the parameter to change e.g. optimiser
        :param parameter: value to assign to the parameter e.g.
                          :py:class:`keras.optimizers.RMSprop`
        :return: A tuple of (parameter value, predictions)
        '''

        setattr(model, parameter_name, parameter)
        model.fit(train[0], train[1], val)
        predictions = model.predict(test)
        return (parameter, predictions)

    @staticmethod
    def evaluate_parameters(model: 'bella.models.base.KerasModel',
                            train: Tuple[np.ndarray, np.ndarray],
                            val: Tuple[np.ndarray, np.ndarray],
                            test: np.ndarray, parameter_name: str,
                            parameters: List[Any], n_jobs: int
                            ) -> List[Tuple[Any, np.ndarray]]:
        '''
        Performs :py:func:`bella.models.base.KerasModel.evaluate_parameter` on
        one `parameter_name` but with multiple parameter values.

        This is useful if you would like to know the affect of changing the
        values of a parameter. It can also perform the task in a
        multiprocessing manner if `n_jobs` > 1.

        :param model: :py:class:`bella.models.base.KerasModel` instance
        :param train: Tuple of `(X_train, y_train)`. Used to fit the model.
        :param val: Tuple of `(X_val, y_val)`. Used to evaluate the
                    model at each epoch. Will not be trained on
                    this data.
        :param test: `X_test` data to predict on.
        :param parameter_name: Name of the parameter to change e.g. optimiser
        :param parameters: A list of values to assign to the parameter e.g.
                           [:py:class:`keras.optimizers.RMSprop`]
        :param n_jobs: Number of cpus to use for multiprocessing if 1 then
                       will not multiprocess.
        :return: A list of tuples of (parameter value, predictions)
        '''
        func_args = ((model, train, val, test, parameter_name, parameter)
                     for parameter in parameters)
        if n_jobs == 1:
            return [KerasModel.evaluate_parameter(*args)
                    for args in func_args]
        with Pool(n_jobs) as pool:
            return pool.starmap(KerasModel.evaluate_parameter, func_args)

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

    @property
    def tokeniser(self) -> Callable[[str], List[str]]:
        '''
        tokeniser attribute

        :return: The tokeniser used in the model
        '''

        return self._tokeniser

    @tokeniser.setter
    def tokeniser(self, value: Callable[[str], List[str]]) -> None:
        '''
        Sets the tokeniser attribute

        :param value: The value to assign to the tokeniser attribute
        '''

        self.fitted = False
        self._tokeniser = value

    @property
    def embeddings(self) -> 'bella.word_vectors.WordVectors':
        '''
        embeddings attribute

        :return: The embeddings used in the model
        '''

        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: 'bella.word_vectors.WordVectors') -> None:
        '''
        Sets the embeddings attribute

        :param value: The value to assign to the embeddings attribute
        '''

        self.fitted = False
        self._embeddings = value

    @property
    def lower(self) -> bool:
        '''
        lower attribute

        :return: The lower used in the model
        '''

        return self._lower

    @lower.setter
    def lower(self, value: bool) -> None:
        '''
        Sets the lower attribute

        :param value: The value to assign to the lower attribute
        '''

        self.fitted = False
        self._lower = value

    @property
    def reproducible(self) -> Union[int, None]:
        '''
        reproducible attribute

        :return: The reproducible used in the model
        '''

        return self._reproducible

    @reproducible.setter
    def reproducible(self, value: Union[int, None]) -> None:
        '''
        Sets the reproducible attribute

        :param value: The value to assign to the reproducible attribute
        '''

        self.fitted = False
        self._reproducible = value

    @property
    def patience(self) -> int:
        '''
        patience attribute

        :return: The patience used in the model
        '''

        return self._patience

    @patience.setter
    def patience(self, value: int) -> None:
        '''
        Sets the patience attribute

        :param value: The value to assign to the patience attribute
        '''

        self.fitted = False
        self._patience = value

    @property
    def batch_size(self) -> int:
        '''
        batch_size attribute

        :return: The batch_size used in the model
        '''

        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        '''
        Sets the batch_size attribute

        :param value: The value to assign to the batch_size attribute
        '''

        self.fitted = False
        self._batch_size = value

    @property
    def epochs(self) -> int:
        '''
        epochs attribute

        :return: The epochs used in the model
        '''

        return self._epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        '''
        Sets the epochs attribute

        :param value: The value to assign to the epochs attribute
        '''

        self.fitted = False
        self._epochs = value

    @property
    def optimiser(self) -> 'keras.optimizers.Optimizer':
        '''
        optimiser attribute

        :return: The optimiser used in the model
        '''

        return self._optimiser

    @optimiser.setter
    def optimiser(self, value: 'keras.optimizers.Optimizer') -> None:
        '''
        Sets the optimiser attribute

        :param value: The value to assign to the optimiser attribute
        '''

        self.fitted = False
        self._optimiser = value

    @property
    def optimiser_params(self) -> Union[Dict[str, Any], None]:
        '''
        optimiser_params attribute

        :return: The optimiser_params used in the model
        '''

        return self._optimiser_params

    @optimiser_params.setter
    def optimiser_params(self, value: Union[Dict[str, Any], None]) -> None:
        '''
        Sets the optimiser_params attribute

        :param value: The value to assign to the optimiser_params attribute
        '''

        self.fitted = False
        self._optimiser_params = value


class SKLearnModel(BaseModel):
    '''
    Concrete class that is designed to be used as the base class for all
    machine learning models that are based on the
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
    3. evaluate_parameter -- fit and predict given training, validation and
       test data the given model when the given parameter is changed on the
       model.
    4. evaluate_parameters -- same as evaluate_parameter however it
       evaluates over many parameter values for the same parameter.
    5. grid_search_model -- Given a model class it will perform a Grid Search
       over the parameters you give to the models
       :py:func:`bella.models.base.SKLearnModel.get_cv_parameters` function
       via the keyword arguments. Returns a pandas dataframe representation of
       the grid search results.
    6. get_grid_score -- Given the return of the :py:func:`grid_search_model`
       will return the grid scores as a List of the mean test accuracy result.
    7. models_best_parameter -- Given a list of models and their base model
       arguments, it will find the best parameter value out of the values
       given for that parameter while keeping the base model arguments
       constant for each model.

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

    @staticmethod
    def evaluate_parameter(model: 'bella.models.base.SKLearnModel',
                           train: Tuple[np.ndarray, np.ndarray],
                           val: None,
                           test: np.ndarray, parameter_name: str,
                           parameter: Any) -> Tuple[Any, np.ndarray]:
        '''
        Given a model will set the `parameter_name` to `parameter` fit the
        model and return the a Tuple of parameter changed and predictions of
        the model on the test data, using the train and validation data for
        fitting.

        :param model: :py:class:`bella.models.base.SKLearn` instance
        :param train: Tuple of `(X_train, y_train)`. Used to fit the model.
        :param val: Use None. This is only kept to keep the API clean.
        :param test: `X_test` data to predict on.
        :param parameter_name: Name of the parameter to change
                               e.g. word_vectors
        :param parameter: value to assign to the parameter e.g.
                          :py:class:`bella.word_vectors.SSWE`
        :return: A tuple of (parameter value, predictions)
        '''

        model.model_parameters = {parameter_name: parameter}
        model.fit(train[0], train[1])
        predictions = model.predict(test)
        return (parameter, predictions)

    @staticmethod
    def evaluate_parameters(model: 'bella.models.base.SKLearnModel',
                            train: Tuple[np.ndarray, np.ndarray],
                            val: None,
                            test: np.ndarray, parameter_name: str,
                            parameters: List[Any], n_jobs: int
                            ) -> List[Tuple[Any, np.ndarray]]:
        '''
        Performs :py:func:`bella.models.base.KerasModel.evaluate_parameter` on
        one `parameter_name` but with multiple parameter values.

        This is useful if you would like to know the affect of changing the
        values of a parameter. It can also perform the task in a
        multiprocessing manner if `n_jobs` > 1.

        :param model: :py:class:`bella.models.base.SKLearn` instance
        :param train: Tuple of `(X_train, y_train)`. Used to fit the model.
        :param val: Use None. This is only kept to keep the API clean.
        :param test: `X_test` data to predict on.
        :param parameter_name: Name of the parameter to change e.g.
                               word_vectors
        :param parameters: A list of values to assign to the parameter e.g.
                           [:py:class:`bella.word_vectors.SSWE`]
        :param n_jobs: Number of cpus to use for multiprocessing if 1 then
                       will not multiprocess.
        :return: A list of tuples of (parameter value, predictions)
        '''
        func_args = ((model, train, val, test, parameter_name, parameter)
                     for parameter in parameters)
        if n_jobs == 1:
            return [SKLearnModel.evaluate_parameter(*args)
                    for args in func_args]
        with Pool(n_jobs) as pool:
            return pool.starmap(SKLearnModel.evaluate_parameter, func_args)

    @staticmethod
    def grid_search_model(model: 'bella.models.base.SKLearnModel',
                          X: np.ndarray, y: np.ndarray, n_cpus: int = 1,
                          num_folds: int = 5, **kwargs) -> pd.DataFrame:
        '''
        Given a model class it will perform a Grid Search over the parameters
        you give to the models :py:func:`bella.models.base.SKLearnModel\
        .get_cv_parameters` function via the keyword arguments. Returns a
        pandas dataframe representation of the grid search results.

        :param model: The class of the model to use not an instance of the
                      model.
        :param X: Training samples matrix, shape = [n_samples, n_features]
        :param y: Training targets, shape = [n_samples]
        :param n_cpus: Number of estimators to fit in parallel. Default 1.
        :param num_folds: Number of Stratified cross validation folds.
                          Default 5.
        :param kwargs: Keyword arguments to give to the models
                       :py:func:`bella.models.base.SKLearnModel\
                       .get_cv_parameters` function.
        :return: Pandas dataframe representation of the grid search results.
        '''
        stratified_folds = StratifiedKFold(num_folds)
        grid_params = model.get_cv_parameters(**kwargs)
        grid_model = GridSearchCV(model.pipeline(), grid_params,
                                  cv=stratified_folds, n_jobs=n_cpus,
                                  return_train_score=False)
        grid_model.fit(X, y)
        return pd.DataFrame(grid_model.cv_results_)

    @staticmethod
    def get_grid_score(grid_scores: pd.DataFrame,
                       associated_param: Union[None, str] = None
                       ) -> Union[List[float], List[Tuple[float, str]]]:
        '''
        Given the return of the :py:func:`grid_search_model` will return
        the grid scores as a List of the mean test accuracy result.

        :param grid_scores: Return of the :py:func:`grid_search_model`
        :param associated_param: Optional. The name of the parameter you want
                                 to associate to the score. E.g. lexicon as you
                                 have grid searched over different lexicons and
                                 you want the return to be associated with the
                                 lexicon name e.g. [(0.68, 'MPQA),
                                 (0.70, 'NRC')]
        :return: A list of test scores from the grid search and if
                 associated_param is not None a list of scores and parameter
                 names.
        '''
        extracted_scores = grid_scores['mean_test_score'].astype(float)
        extracted_scores = extracted_scores.round(4) * 100
        extracted_scores = extracted_scores.tolist()
        if associated_param is not None:
            if associated_param not in grid_scores:
                for column_name in grid_scores.columns:
                    if associated_param in column_name:
                        associated_param = column_name
            associated_param = grid_scores[associated_param]
            associated_param = associated_param.apply(str).tolist()
            extracted_scores = list(zip(extracted_scores, associated_param))
        return extracted_scores

    @staticmethod
    def models_best_parameter(models_kwargs: List[Tuple['bella.models.base.SKLearnModel',
                                                        Dict[str, Any]]],
                              param_name: str, param_values: List[Any],
                              X: List[Any], y: np.ndarray, n_cpus: int = 1,
                              num_folds: int = 5
                              ) -> Dict['bella.models.base.SKLearnModel', str]:
        '''
        Given a list of models and their base model arguments, it will
        find the best parameter value out of the values given for that
        parameter while keeping the base model arguments constant for
        each model.

        This essentially performs 5 fold cross validation grid search
        for the one parameter given, across all models given.

        :param models_kwargs: A list of tuples where each tuple contains
                              a model and the models keyword arguments to
                              give to its `get_cv_parameters` method. These
                              arguments are the models standard arguments
                              that are not to be changed.
        :param param_name: Name of the parameter to be changed. This name
                           has to be the name of the keyword argument in
                           the models `get_cv_parameters` method.
        :param param_values: The different values to assign to the param_name
                             argument.
        :param X: The training samples.
        :param y: The training target samples.
        :return: A dictionary of model and the name of the best parameter.
        '''
        model_best_param = {}
        for model, model_kwargs in models_kwargs:
            temp_model_kwargs = {**model_kwargs, param_name: param_values}
            grid_results = model.grid_search_model(model, X, y, n_cpus=n_cpus,
                                                   num_folds=num_folds,
                                                   **temp_model_kwargs)
            param_scores = model.get_grid_score(grid_results, param_name)
            param_scores = sorted(param_scores, key=lambda x: x[1],
                                  reverse=True)
            best_param = sorted(param_scores, key=lambda x: x[0])[-1][1]
            model_best_param[model] = best_param
        return model_best_param

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
