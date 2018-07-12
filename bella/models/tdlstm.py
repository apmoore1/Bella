import random as rn
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, Callable, Any, List, Union, Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import preprocessing, models, optimizers, initializers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import model_from_yaml

# Displaying the Neural Network models
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

import bella
from bella.models.base import KerasModel
from bella.contexts import context
from bella.neural_pooling import matrix_median
from bella.notebook_helper import get_json_data, write_json_data


class LSTM(KerasModel):

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.

        :return: Name of the machine learning model.
        '''

        return 'LSTM'

    def __init__(self, tokeniser: Callable[[str], List[str]],
                 embeddings: 'bella.word_vectors.WordVectors',
                 reproducible: Union[int, None] = None, pad_size: int = -1,
                 lower: bool = True, patience: int = 10,
                 batch_size: int = 32, epochs: int = 300,
                 embedding_layer_kwargs: Dict[str, Any] = None,
                 lstm_layer_kwargs: Dict[str, Any] = None,
                 dense_layer_kwargs: Dict[str, Any] = None,
                 optimiser: 'keras.optimizers.Optimizer' = optimizers.SGD,
                 optimiser_params: Union[Dict[str, Any], None] = None
                 ) -> None:
        '''
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`.
        :param embeddings: Embedding (Word vectors) to be used e.g.
                           :py:class:`bella.word_vectors.SSWE`
        :param reproducible: Whether to be reproducible. If None then it is
                             but quicker to run. Else provide a `int` that
                             will represent the random seed value.
        :param pad_size: The max number of tokens to use per sequence. If -1
                         use the text sequence in the training data that has
                         the most tokens as the pad size.
        :param lower: Whether to lower case the words being processed.
        :param patience: Number of epochs with no improvement before training
                         is stopped.
        :param batch_size: Number of samples per gradient update.
        :param epochs: Number of times to train over the entire training set
                       before stopping. If patience is set, then it may
                       stop before reaching the number of epochs specified
                       here.
        :param embedding_layer_kwargs: Keyword arguments to pass to the
                                       embedding layer which is a
                                       :py:class:`keras.layers.Embedding`
                                       object. If no parameters to pass leave
                                       as None.
        :param lstm_layer_kwargs: Keyword arguments to pass to the lstm
                                  layer(s) which is a
                                  :py:class:`keras.layers.LSTM` object. If no
                                  parameters to pass leave as None.
        :param dense_layer_kwargs: Keyword arguments to pass to the dense
                                   (final layer) which is a
                                   :py:class:`keras.layers.Dense` object. If no
                                   parameters to pass leave as None.
        :param optimiser: Optimiser to be used accepts any
                          `keras optimiser <https://keras.io/optimizers/>`_.
                          Default is :py:class:`keras.optimizers.SGD`
        :param optimiser_params: Parameters for the optimiser. If None uses
                                 default optimiser parameters.
        '''

        self.tokeniser = tokeniser
        self.embeddings = embeddings
        self.reproducible = reproducible
        self.pad_size = pad_size
        self.test_pad_size = 0
        self.lower = lower
        self.patience = patience
        self.batch_size = batch_size
        self.epochs = epochs

        self.optimiser_params = optimiser_params
        if optimiser_params is None:
            self.optimiser_params = {}

        self.embedding_layer_kwargs = embedding_layer_kwargs
        if embedding_layer_kwargs is None:
            self.embedding_layer_kwargs = {}
        self.lstm_layer_kwargs = lstm_layer_kwargs
        if lstm_layer_kwargs is None:
            self.lstm_layer_kwargs = {}
        self.dense_layer_kwargs = dense_layer_kwargs
        if dense_layer_kwargs is None:
            self.dense_layer_kwargs = {}
        self.optimiser = optimiser
        self.model = None
        self.fitted = False

    def model_parameters(self) -> Dict[str, Any]:
        '''
        Returns a dictionary of all of the attributes that affect the model as
        well as the class the model belongs to.
        '''

        class_params = {'tokeniser': self.tokeniser,
                        'embeddings': self.embeddings,
                        'reproducible': self.reproducible,
                        'pad_size': self.pad_size,
                        'lower': self.lower,
                        'patience': self.patience,
                        'batch_size': self.batch_size, 'epochs': self.epochs,
                        'embedding_layer_kwargs': self.embedding_layer_kwargs,
                        'lstm_layer_kwargs': self.lstm_layer_kwargs,
                        'dense_layer_kwargs': self.dense_layer_kwargs,
                        'optimiser': self.optimiser,
                        'optimiser_params': self.optimiser_params}
        class_attrs = {'test_pad_size': self.test_pad_size}
        return {'class': self.__class__, 'class_attrs': class_attrs,
                'class_params': class_params}

    def process_text(self, texts, max_length, padding='pre', truncate='pre'):
        '''
        Given a list of Strings, tokenised the text and lower case if set and then
        convert the tokens into a integers representing the tokens in the
        embeddings. Lastly it pads the data based on the max_length param.

        If the max_length is smaller than the sentences size it truncates the
        sentence. If max_length = -1 then the max_length is that of the longest
        sentence in the texts.

        :params texts: list of Strings
        :params max_length: How many tokens a sentence can contain. If it is \
        -1 then it uses the sentence with the most tokens as the max_length param.
        :params padding: which side of the sentence to pad: `pre` beginning, \
        `post` end.
        :params truncate: which side of the sentence to truncate: `pre` beginning \
        `post` end.
        :type texts: list
        :type max_length: int
        :type padding: String. Either `pre` or `post` default `pre`
        :type truncate: String. Either `pre` or `post` default `pre`
        :returns: A tuple of length 2 containg: 1. The max_length parameter, 2. \
        A list of a list of integers that have been padded.
        :rtype: tuple
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
                sequence_data.append(self.embeddings.word2index[token])
            sequence_length = len(sequence_data)
            if sequence_length > max_sequence:
                max_sequence = sequence_length
            all_sequence_data.append(sequence_data)
        if max_sequence == 0:
            raise ValueError('The max sequence length is 0 suggesting no '\
                             'data was provided for training or testing')
        # Pad the sequences
        # If max pad size is set and training the model set the test_pad_size
        # to max sequence length
        if max_length == -1:
            max_length = max_sequence
        return (max_length,
                preprocessing.sequence.pad_sequences(all_sequence_data,
                                                     maxlen=max_length,
                                                     dtype='int32',
                                                     padding=padding,
                                                     truncating=truncate))

    def _pre_process(self, data_dicts, training=False):
        text_data = [data['text'] for data in data_dicts]
        if training:
            pad_data = self.process_text(text_data, self.pad_size)
            self.test_pad_size, sequence_data = pad_data
            return sequence_data
        else:
            _, sequence_data = self.process_text(text_data, self.test_pad_size)
            return sequence_data

    def create_training_y(self, train_y, validation_y):
        train_y = to_categorical(train_y).astype(np.float32)
        validation_y = to_categorical(validation_y).astype(np.float32)
        return train_y, validation_y

    def create_training_text(self, train_data, validation_data):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values
        :validation_size: The fraction of the training data to be set aside \
        for validation data
        :type train_data: list
        :type train_y: list
        :type validation_size: float Default 0.2
        :returns: A tuple of length 2 where the first value is a list of \
        Integers that reprsent the words in the text features where each Integer \
        corresponds to a Word Vector in the embedding vector. Second value are \
        the target values. Both lists in the tuples contain training data in the \
        first part of the list and the second part of the list based on the \
        validation split contains the validation data.
        :rtype: tuple
        '''

        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.
        sequence_train_data = self._pre_process(train_data, training=True)
        sequence_val_data = self._pre_process(validation_data, training=False)
        # Stack the validation data with the training data to complie with Keras.
        # all_text = np.vstack((sequence_train_data, sequence_val_data))
        return sequence_train_data, sequence_val_data

    def repeated_results(self, train, test, n_results, score_func, dataset_name,
                         score_args=None, score_kwargs=None,
                         results_file=None, re_write=False, **fit_kwargs):
        if results_file is not None:
            all_scores = get_json_data(results_file, dataset_name)
            if len(all_scores) != 0  and not re_write:
                return all_scores
        train_data = train.data_dict()
        train_y = train.sentiment_data()
        test_data = test.data_dict()
        test_y = test.sentiment_data()
        scores = []
        for i in range(n_results):
            score, _ = self.fit_predict(train_data, train_y, test_data,
                                        test_y, fit_kwargs, score_func,
                                        score_args, score_kwargs)
            print(score)
            scores.append(score)
            if results_file is not None:
                write_json_data(results_file, dataset_name, scores)
        return scores

    def keras_model(self, num_classes):
        # Embeddings
        embedding_matrix = self.embeddings.embedding_matrix
        vocab_size, vector_size = embedding_matrix.shape

        embedding_layer_kwargs = self.embedding_layer_kwargs
        embedding_layer_trainable = True
        if 'trainable' in embedding_layer_kwargs:
            embedding_layer_trainable = embedding_layer_kwargs.pop('trainable')

        lstm_layer_kwargs = self.lstm_layer_kwargs
        lstm_dimension = vector_size
        if 'cell' in self.lstm_layer_kwargs:
            lstm_dimension = lstm_layer_kwargs.pop('cell')

        dense_layer_kwargs = self.dense_layer_kwargs
        # output_activation = 'softmax' if num_classes > 2 else ''
        # Model layers
        input_layer = layers.Input(shape=(self.test_pad_size,),
                                   name='text_input')
        embedding_layer = layers\
                          .Embedding(input_dim=vocab_size,
                                     output_dim=vector_size,
                                     input_length=self.test_pad_size,
                                     trainable=embedding_layer_trainable,
                                     weights=[embedding_matrix],
                                     name='embedding_layer',
                                     **embedding_layer_kwargs
                                     )(input_layer)
        lstm_layer = layers.LSTM(lstm_dimension,
                                 name='lstm_layer',
                                 **lstm_layer_kwargs)(embedding_layer)
        prediction_layer = layers.Dense(num_classes, activation='softmax',
                                        name='output',
                                        **dense_layer_kwargs)(lstm_layer)

        return models.Model(inputs=input_layer, outputs=prediction_layer)

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
    def pad_size(self) -> int:
        '''
        pad_size attribute

        :return: The pad_size used in the model
        '''

        return self._pad_size

    @pad_size.setter
    def pad_size(self, value: int) -> None:
        '''
        Sets the pad_size attribute

        :param value: The value to assign to the pad_size attribute
        '''

        self.fitted = False
        self._pad_size = value

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
    def embedding_layer_kwargs(self) -> Dict[str, Any]:
        '''
        embedding_layer_kwargs attribute

        :return: The embedding_layer_kwargs used in the model
        '''

        return self._embedding_layer_kwargs

    @embedding_layer_kwargs.setter
    def embedding_layer_kwargs(self, value: Dict[str, Any]) -> None:
        '''
        Sets the embedding_layer_kwargs attribute

        :param value: The value to assign to the embedding_layer_kwargs
                      attribute
        '''

        self.fitted = False
        self._embedding_layer_kwargs = value

    @property
    def lstm_layer_kwargs(self) -> Dict[str, Any]:
        '''
        lstm_layer_kwargs attribute

        :return: The lstm_layer_kwargs used in the model
        '''

        return self._lstm_layer_kwargs

    @lstm_layer_kwargs.setter
    def lstm_layer_kwargs(self, value: Dict[str, Any]) -> None:
        '''
        Sets the lstm_layer_kwargs attribute

        :param value: The value to assign to the lstm_layer_kwargs
                      attribute
        '''

        self.fitted = False
        self._lstm_layer_kwargs = value

    @property
    def dense_layer_kwargs(self) -> Dict[str, Any]:
        '''
        dense_layer_kwargs attribute

        :return: The dense_layer_kwargs used in the model
        '''

        return self._dense_layer_kwargs

    @dense_layer_kwargs.setter
    def dense_layer_kwargs(self, value: Dict[str, Any]) -> None:
        '''
        Sets the dense_layer_kwargs attribute

        :param value: The value to assign to the dense_layer_kwargs
                      attribute
        '''

        self.fitted = False
        self._dense_layer_kwargs = value

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



    def visulaise(self, plot_format='vert'):
        '''
        :param plot_format: Whether the plot is shown vertical or horizontal. \
        Vertical is default and denoted as `vert` else horizontal is `hoz`
        :type plot_format: String
        :returns: A plot showing the structure of the Neural Network when using \
        a Jupyter or IPython notebook
        :rtype: IPython.core.display.SVG
        '''

        if self.model is None:
            raise ValueError('The model has to be fitted before being able '\
                             'to visulaise it.')
        rankdir = 'TB'
        if plot_format == 'hoz':
            rankdir = 'LR'
        dot_model = model_to_dot(self.model, show_shapes=True,
                                 show_layer_names=True, rankdir=rankdir)
        return SVG(dot_model.create(prog='dot', format='svg'))

    def visulaise_to_file(self, file_path, plot_format='vert'):
        '''
        :param file_path: File path to save the plot of the Neural Network.
        :param plot_format: Whether the plot is shown vertical or horizontal. \
        Vertical is default and denoted as `vert` else horizontal is `hoz`
        :type file_path: String
        :type plot_format: String. Default 'vert'
        :returns: Nothing. Saves the visual to the file path given.
        :rtype: None
        '''

        if self.model is None:
            raise ValueError('The model has to be fitted before being able '\
                             'to visulaise it.')
        rankdir = 'TB'
        if plot_format == 'hoz':
            rankdir = 'LR'
        plot_model(self.model, to_file=file_path, show_shapes=True,
                   show_layer_names=True, rankdir=rankdir)


class TDLSTM(LSTM):

    def __repr__(self):
        '''
        Name of the machine learning model.

        :return: Name of the machine learning model.
        '''

        return 'TDLSTM'

    def __init__(self, tokeniser: Callable[[str], List[str]],
                 embeddings: 'bella.word_vectors.WordVectors',
                 reproducible: Union[int, None] = None, pad_size: int = -1,
                 lower: bool = True, patience: int = 10,
                 batch_size: int = 32, epochs: int = 300,
                 embedding_layer_kwargs: Dict[str, Any] = None,
                 lstm_layer_kwargs: Dict[str, Any] = None,
                 dense_layer_kwargs: Dict[str, Any] = None,
                 optimiser: 'keras.optimizers.Optimizer' = optimizers.SGD,
                 optimiser_params: Union[Dict[str, Any], None] = None,
                 include_target: bool = True) -> None:
        '''
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`.
        :param embeddings: Embedding (Word vectors) to be used e.g.
                           :py:class:`bella.word_vectors.SSWE`
        :param reproducible: Whether to be reproducible. If None then it is
                             but quicker to run. Else provide a `int` that
                             will represent the random seed value.
        :param pad_size: The max number of tokens to use per sequence. If -1
                         use the text sequence in the training data that has
                         the most tokens as the pad size.
        :param lower: Whether to lower case the words being processed.
        :param patience: Number of epochs with no improvement before training
                         is stopped.
        :param batch_size: Number of samples per gradient update.
        :param epochs: Number of times to train over the entire training set
                       before stopping. If patience is set, then it may
                       stop before reaching the number of epochs specified
                       here.
        :param embedding_layer_kwargs: Keyword arguments to pass to the
                                       embedding layer which is a
                                       :py:class:`keras.layers.Embedding`
                                       object. If no parameters to pass leave
                                       as None.
        :param lstm_layer_kwargs: Keyword arguments to pass to the lstm
                                  layer(s) which is a
                                  :py:class:`keras.layers.LSTM` object. If no
                                  parameters to pass leave as None.
        :param dense_layer_kwargs: Keyword arguments to pass to the dense
                                   (final layer) which is a
                                   :py:class:`keras.layers.Dense` object. If no
                                   parameters to pass leave as None.
        :param optimiser: Optimiser to be used accepts any
                          `keras optimiser <https://keras.io/optimizers/>`_.
                          Default is :py:class:`keras.optimizers.SGD`
        :param optimiser_params: Parameters for the optimiser. If None uses
                                 default optimiser parameters.
        :param include_target: Wheather to include the target in the LSTM
                               representations.
        '''

        super().__init__(tokeniser, embeddings, reproducible, pad_size, lower,
                         patience, batch_size, epochs, embedding_layer_kwargs,
                         lstm_layer_kwargs, dense_layer_kwargs, optimiser,
                         optimiser_params)

        self.left_pad_size = pad_size
        self.left_test_pad_size = 0
        self.right_pad_size = pad_size
        self.right_test_pad_size = 0
        self.include_target = include_target

    def model_parameters(self) -> Dict[str, Any]:
        '''
        Returns a dictionary of all of the attributes that affect the model as
        well as the class the model belongs to.
        '''

        attributes = super().model_parameters()
        class_attrs = {'left_test_pad_size': self.left_test_pad_size,
                       'right_test_pad_size': self.right_test_pad_size}
        attributes['class_attrs'] = class_attrs

        class_params = attributes['class_params']
        class_params['include_target'] = self.include_target
        attributes['class_params'] = class_params
        return attributes

    def _pre_process(self, data_dicts, training=False):

        def context_texts(context_data_dicts):
            # Context returns all of the left and right context occurrences
            # therefore if a target is mentioned Twice and are associated then
            # for a single text two left and right occurrences are returned.
            # Thus these are a list of lists we therefore chose only the
            # first mentioned target as the paper linked to this method does
            # not specify which they used.
            left_texts = [context(data, 'left', inc_target=self.include_target)
                          for data in context_data_dicts]
            right_texts = [context(data, 'right', inc_target=self.include_target)
                           for data in context_data_dicts]
            left_texts = [texts[0] for texts in left_texts]
            right_texts = [texts[0] for texts in right_texts]
            return left_texts, right_texts

        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.

        # Get left and right contexts
        left_text, right_text = context_texts(data_dicts)
        if training:
            left_pad_sequence = self.process_text(left_text,
                                                  self.left_pad_size)
            self.left_test_pad_size, left_sequence = left_pad_sequence

            right_pad_sequence = self.process_text(right_text,
                                                   self.right_pad_size,
                                                   padding='post',
                                                   truncate='post')
            self.right_test_pad_size, right_sequence = right_pad_sequence
            return left_sequence, right_sequence
        else:
            left_pad_sequence = self.process_text(left_text,
                                                  self.left_test_pad_size)
            _, left_sequence = left_pad_sequence

            right_pad_sequence = self.process_text(right_text,
                                                   self.right_test_pad_size,
                                                   padding='post',
                                                   truncate='post')
            _, right_sequence = right_pad_sequence
            return [left_sequence, right_sequence]

    def create_training_text(self, train_data, validation_data):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values
        :validation_size: The fraction of the training data to be set aside \
        for validation data
        :type train_data: list
        :type train_y: list
        :type validation_size: float Default 0.2
        :returns: A tuple of length 2 where the first value is a list of \
        Integers that reprsent the words in the text features where each Integer \
        corresponds to a Word Vector in the embedding vector. Second value are \
        the target values. Both lists in the tuples contain training data in the \
        first part of the list and the second part of the list based on the \
        validation split contains the validation data.
        :rtype: tuple
        '''

        train_sequences = self._pre_process(train_data, training=True)
        validation_sequences = self._pre_process(validation_data,
                                                 training=False)
        return train_sequences, validation_sequences

    def keras_model(self, num_classes):
        # Embeddings
        embedding_matrix = self.embeddings.embedding_matrix
        vocab_size, vector_size = embedding_matrix.shape

        embedding_layer_kwargs = self.embedding_layer_kwargs
        embedding_layer_trainable = True
        if 'trainable' in embedding_layer_kwargs:
            embedding_layer_trainable = embedding_layer_kwargs.pop('trainable')

        lstm_layer_kwargs = self.lstm_layer_kwargs
        lstm_dimension = vector_size
        if 'cell' in self.lstm_layer_kwargs:
            lstm_dimension = lstm_layer_kwargs.pop('cell')

        dense_layer_kwargs = self.dense_layer_kwargs
        # Model layers
        # Left LSTM
        left_input = layers.Input(shape=(self.left_test_pad_size,),
                                  name='left_text_input')
        left_embedding_layer = layers\
                               .Embedding(input_dim=vocab_size,
                                          output_dim=vector_size,
                                          input_length=self.left_test_pad_size,
                                          trainable=embedding_layer_trainable,
                                          weights=[embedding_matrix],
                                          name='left_embedding_layer',
                                          **embedding_layer_kwargs
                                          )(left_input)
        left_lstm_layer = layers.LSTM(lstm_dimension,
                                      name='left_lstm_layer',
                                      **lstm_layer_kwargs
                                      )(left_embedding_layer)
        # Right LSTM
        right_input = layers.Input(shape=(self.right_test_pad_size,),
                                   name='right_text_input')
        right_embedding_layer = layers\
                                .Embedding(input_dim=vocab_size,
                                           output_dim=vector_size,
                                           input_length=self.right_test_pad_size,
                                           trainable=embedding_layer_trainable,
                                           weights=[embedding_matrix],
                                           name='right_embedding_layer',
                                           **embedding_layer_kwargs
                                           )(right_input)
        right_lstm_layer = layers.LSTM(lstm_dimension,
                                       name='right_lstm_layer',
                                       **lstm_layer_kwargs
                                       )(right_embedding_layer)
        # Merge the outputs of the left and right LSTMs
        merge_layer = layers.concatenate([left_lstm_layer, right_lstm_layer],
                                         name='left_right_lstm_merge')
        predictions = layers.Dense(num_classes, activation='softmax',
                                   name='output',
                                   **dense_layer_kwargs)(merge_layer)

        return models.Model(inputs=[left_input, right_input],
                            outputs=predictions)

    @property
    def include_target(self) -> bool:
        '''
        include_target attribute

        :return: The include_target used in the model
        '''

        return self._include_target

    @include_target.setter
    def include_target(self, value: bool) -> None:
        '''
        Sets the include_target attribute

        :param value: The value to assign to the include_target attribute
        '''

        self.fitted = False
        self._include_target = value


class TCLSTM(TDLSTM):

    def __repr__(self):
        '''
        Name of the machine learning model.

        :return: Name of the machine learning model.
        '''

        return 'TCLSTM'

    def __init__(self, tokeniser: Callable[[str], List[str]],
                 embeddings: 'bella.word_vectors.WordVectors',
                 reproducible: Union[int, None] = None, pad_size: int = -1,
                 lower: bool = True, patience: int = 10,
                 batch_size: int = 32, epochs: int = 300,
                 embedding_layer_kwargs: Dict[str, Any] = None,
                 lstm_layer_kwargs: Dict[str, Any] = None,
                 dense_layer_kwargs: Dict[str, Any] = None,
                 optimiser: 'keras.optimizers.Optimizer' = optimizers.SGD,
                 optimiser_params: Union[Dict[str, Any], None] = None,
                 include_target: bool = True) -> None:
        '''
        :param tokeniser: Tokeniser to be used e.g. :py:meth:`str.split`.
        :param embeddings: Embedding (Word vectors) to be used e.g.
                           :py:class:`bella.word_vectors.SSWE`
        :param reproducible: Whether to be reproducible. If None then it is
                             but quicker to run. Else provide a `int` that
                             will represent the random seed value.
        :param pad_size: The max number of tokens to use per sequence. If -1
                         use the text sequence in the training data that has
                         the most tokens as the pad size.
        :param lower: Whether to lower case the words being processed.
        :param patience: Number of epochs with no improvement before training
                         is stopped.
        :param batch_size: Number of samples per gradient update.
        :param epochs: Number of times to train over the entire training set
                       before stopping. If patience is set, then it may
                       stop before reaching the number of epochs specified
                       here.
        :param embedding_layer_kwargs: Keyword arguments to pass to the
                                       embedding layer which is a
                                       :py:class:`keras.layers.Embedding`
                                       object. If no parameters to pass leave
                                       as None.
        :param lstm_layer_kwargs: Keyword arguments to pass to the lstm
                                  layer(s) which is a
                                  :py:class:`keras.layers.LSTM` object. If no
                                  parameters to pass leave as None.
        :param dense_layer_kwargs: Keyword arguments to pass to the dense
                                   (final layer) which is a
                                   :py:class:`keras.layers.Dense` object. If no
                                   parameters to pass leave as None.
        :param optimiser: Optimiser to be used accepts any
                          `keras optimiser <https://keras.io/optimizers/>`_.
                          Default is :py:class:`keras.optimizers.SGD`
        :param optimiser_params: Parameters for the optimiser. If None uses
                                 default optimiser parameters.
        :param include_target: Wheather to include the target in the LSTM
                               representations.
        '''

        super().__init__(tokeniser, embeddings, reproducible, pad_size, lower,
                         patience, batch_size, epochs, embedding_layer_kwargs,
                         lstm_layer_kwargs, dense_layer_kwargs, optimiser,
                         optimiser_params, include_target)

    def _pre_process(self, data_dicts, training=False):
        def context_median_targets(pad_size):
            vector_size = self.embeddings.vector_size
            target_matrix = np.zeros((len(data_dicts),
                                      pad_size, vector_size))
            for index, data in enumerate(data_dicts):
                target_vectors = []
                target_words = data['target'].split()
                for target_word in target_words:
                    if self.lower:
                        target_word = target_word.lower()
                    target_embedding = self.embeddings\
                                           .lookup_vector(target_word)
                    target_vectors.append(target_embedding)
                target_vectors = np.vstack(target_vectors)
                median_target_vector = matrix_median(target_vectors)
                median_vectors = np.repeat(median_target_vector, pad_size,
                                           axis=0)
                target_matrix[index] = median_vectors
            return target_matrix

        sequences = super()._pre_process(data_dicts, training=training)
        left_sequence, right_sequence = sequences
        left_target_vectors = context_median_targets(self.left_test_pad_size)
        right_target_vectors = context_median_targets(self.right_test_pad_size)
        return [left_sequence, left_target_vectors,
                right_sequence, right_target_vectors]

    def create_training_text(self, train_data, validation_data):
        '''
        :param train_data: :param train_data: Training features. Specifically \
        a list of dict like structures that contain `target` key.
        '''

        train_seq_targ = self._pre_process(train_data, training=True)
        validation_seq_targ = self._pre_process(validation_data,
                                                training=False)
        return train_seq_targ, validation_seq_targ

    def keras_model(self, num_classes):
        # Embeddings
        embedding_matrix = self.embeddings.embedding_matrix
        vocab_size, vector_size = embedding_matrix.shape

        embedding_layer_kwargs = self.embedding_layer_kwargs
        embedding_layer_trainable = True
        if 'trainable' in embedding_layer_kwargs:
            embedding_layer_trainable = embedding_layer_kwargs.pop('trainable')

        lstm_layer_kwargs = self.lstm_layer_kwargs
        # Double the vector size as we have to take into consideration the
        # concatenated target vector
        lstm_dimension = vector_size * 2
        if 'cell' in self.lstm_layer_kwargs:
            lstm_dimension = lstm_layer_kwargs.pop('cell')

        dense_layer_kwargs = self.dense_layer_kwargs
        # Model layers
        # Left LSTM
        left_input = layers.Input(shape=(self.left_test_pad_size,),
                                  name='left_text_input')
        left_embedding_layer = layers\
                               .Embedding(input_dim=vocab_size,
                                          output_dim=vector_size,
                                          input_length=self.left_test_pad_size,
                                          trainable=embedding_layer_trainable,
                                          weights=[embedding_matrix],
                                          name='left_embedding_layer',
                                          **embedding_layer_kwargs
                                          )(left_input)
        left_target_input = layers.Input(shape=(self.left_test_pad_size,
                                                vector_size),
                                         name='left_target')
        left_text_target = layers.concatenate([left_embedding_layer,
                                               left_target_input],
                                              name='left_text_target')
        left_lstm_layer = layers.LSTM(lstm_dimension,
                                      name='left_lstm_layer',
                                      **lstm_layer_kwargs
                                      )(left_text_target)
        # Right LSTM
        right_input = layers.Input(shape=(self.right_test_pad_size,),
                                   name='right_text_input')
        right_embedding_layer = layers\
                                .Embedding(input_dim=vocab_size,
                                           output_dim=vector_size,
                                           input_length=self.right_test_pad_size,
                                           trainable=embedding_layer_trainable,
                                           weights=[embedding_matrix],
                                           name='right_embedding_layer',
                                           **embedding_layer_kwargs
                                           )(right_input)
        right_target_input = layers.Input(shape=(self.right_test_pad_size,
                                                 vector_size),
                                          name='right_target')
        right_text_target = layers.concatenate([right_embedding_layer,
                                                right_target_input],
                                               name='right_text_target')
        right_lstm_layer = layers.LSTM(lstm_dimension,
                                       name='right_lstm_layer',
                                       **lstm_layer_kwargs
                                       )(right_text_target)
        # Merge the outputs of the left and right LSTMs
        merge_layer = layers.concatenate([left_lstm_layer, right_lstm_layer],
                                         name='left_right_lstm_merge')
        predictions = layers.Dense(num_classes, activation='softmax',
                                   name='output',
                                   **dense_layer_kwargs)(merge_layer)

        input_layers = [left_input, left_target_input,
                        right_input, right_target_input]
        return models.Model(inputs=input_layers, outputs=predictions)
