from pathlib import Path
from typing import Dict, Callable, Any, List, Union, Tuple

import numpy as np
from keras import models, optimizers, layers
from keras.utils import to_categorical

# Displaying the Neural Network models
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG

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
        Returns a dictionary containing the attributes of the class instance,
        the parameters to give to the class constructior to re-create this
        instance, and the class itself.

        This is used by the :py:meth:`save` method so that the instance can
        be re-created when loaded by the :py:meth:`load` method.
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

    def _pre_process(self, data_dicts: Dict[str, str],
                     training: bool = False) -> List[List[int]]:
        '''
        Converts the text in the data_dicts into a list of a list of integers
        representing the text as their embedding lookups, so that it can
        be used as input to the keras model.

        The text from the data_dicts are converted by the
        :py:meth:`process_text` method.

        :param data_dicts: A list of dictonaries that contain a `text` key.
        :param training: Whether the text should be processed for training or
                         for prediction. prediction = False, training = True
        :return: The output of :py:meth:`process_text` method.
        '''
        text_data = [data['text'] for data in data_dicts]
        if training:
            pad_data = self.process_text(text_data, self.pad_size)
            self.test_pad_size, sequence_data = pad_data
            return sequence_data
        _, sequence_data = self.process_text(text_data, self.test_pad_size)
        return sequence_data

    def create_training_y(self, train_y: np.ndarray, validation_y: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Converts the training and validation target values from a vector of
        class lables into a matrix of binary values.

        To convert the vector of classes to a matrix we the
        :py:func:`keras.utils.to_categorical` function.

        :param train_y: Vector of class labels, shape = [n_samples]
        :param validation_y: Vector of class labels, shape = [n_samples]
        :return: A tuple of length two containing the train and validation
                 matrices respectively. The shape of each matrix is:
                 [n_samples, n_classes]
        '''
        train_y = to_categorical(train_y).astype(np.float32)
        validation_y = to_categorical(validation_y).astype(np.float32)
        return train_y, validation_y

    def create_training_text(self, train_data: Dict[str, str],
                             validation_data: Dict[str, str]
                             ) -> Tuple[List[List[int]], List[List[int]]]:
        '''
        Converts the training and validation data into a format that the keras
        model can take as input.

        :return: A tuple of length two containing the train and validation
                 input that are both the output of :py:meth:`_pre_process`
        '''
        train_sequence = self._pre_process(train_data, training=True)
        val_sequence = self._pre_process(validation_data, training=False)
        return train_sequence, val_sequence

    def keras_model(self, num_classes: int) -> 'keras.models.Model':
        '''
        The model that represents this class. This is a Single forward LSTM.

        :param num_classes: Number of classes to predict.
        :return: Forward LSTM keras model.
        '''
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
        Returns a dictionary containing the attributes of the class instance,
        the parameters to give to the class constructior to re-create this
        instance, and the class itself.

        This is used by the :py:meth:`save` method so that the instance can
        be re-created when loaded by the :py:meth:`load` method.
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

        return super().create_training_text(train_data, validation_data)

    def keras_model(self, num_classes: int) -> 'keras.models.Model':
        '''
        The model that represents this class. This is a custom combination
        of two LSTMs.

        :param num_classes: Number of classes to predict.
        :return: Two LSTMs, one forward from the left context and the other
                 backward from the right context. The output of the two are
                 concatenated and are input to the output layer.
        '''
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
                                       go_backwards=True,
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

        return super().create_training_text(train_data, validation_data)

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
                                       go_backwards=True,
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
