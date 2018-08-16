'''
Module contains all of the classes that represent Machine Learning models
that are within `Tang et al. 2016 paper \
<https://aclanthology.info/papers/C16-1311/c16-1311>`_:

1. :py:class:`bella.models.tdlstm.LSTM` -- LSTM model.
2. :py:class:`bella.models.tdlstm.TDLSTM` -- TDLSTM model.
3. :py:class:`bella.models.tdlstm.TCLSTM` -- TCLSTM model.
'''

from typing import Dict, Callable, Any, List, Union, Tuple

import numpy as np
import keras
from keras import models, optimizers, layers
from keras.utils import to_categorical

import bella
from bella.models.base import KerasModel
from bella.contexts import context
from bella.neural_pooling import matrix_median


class LSTM(KerasModel):
    '''
    Attributes:

    1. pad_size -- The max number of tokens to use per sequence. If -1
       use the text sequence in the training data that has the most tokens as
       the pad size.
    2. embedding_layer_kwargs -- Keyword arguments to pass to the embedding
       layer which is a :py:class:`keras.layers.Embedding` object. Can be
       None if no parameters to pass.
    3. lstm_layer_kwargs -- Keyword arguments to pass to the lstm layer(s)
       which is a :py:class:`keras.layers.LSTM` object. Can be
       None if no parameters to pass.
    4. dense_layer_kwargs -- Keyword arguments to pass to the dense (final
       layer) which is a :py:class:`keras.layers.Dense` object. Can be
       None if no parameters to pass.

    Methods:

    1. model_parameters -- Returns a dictionary containing the attributes of
       the class instance, the parameters to give to the class constructior to
       re-create this instance, and the class itself.
    2. create_training_text -- Converts the training and validation data into a
       format that the keras model can take as input.
    3. create_training_y -- Converts the training and validation target values
       from a vector of class lables into a matrix of binary values. of shape
       [n_samples, n_classes].
    4. keras_model -- The model that represents this class. This is a single
       forward LSTM.
    '''

    @classmethod
    def name(cls) -> str:
        return 'LSTM'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

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
                             quicker to run. Else provide a `int` that
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

    def _pre_process(self, data_dicts: List[Dict[str, str]],
                     training: bool = False) -> np.ndarray:
        '''
        Converts the text in the data_dicts into a matrix of shape
        [n_samples, pad_size] where each integer in the matrix represents
        the word embedding lookup. This is then used as input into the
        keras model.

        The text from the data_dicts are converted by the
        :py:meth:`process_text` method.

        :param data_dicts: A list of dictonaries that contains a `text` field.
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
        class lables into a matrix of binary values of shape [n_samples,
        n_classes].

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

    def create_training_text(self, train_data: List[Dict[str, str]],
                             validation_data: List[Dict[str, str]]
                             ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Converts the training and validation data into a format that the keras
        model can take as input.

        :param train_data: Data to be trained on. Which is a list of
                           dictionaries where each dictionary has a `text`
                           field containing text.
        :param validation_data: Data to evaluate the model at training time.
                                Which is a list of dictionaries where each
                                dictionary has a `text` field containing text.
        :return: A tuple of length two containing the train and validation
                 input that are both the output of :py:meth:`_pre_process`
        '''
        train_sequence = self._pre_process(train_data, training=True)
        val_sequence = self._pre_process(validation_data, training=False)
        return train_sequence, val_sequence

    def keras_model(self, num_classes: int) -> 'keras.models.Model':
        '''
        The model that represents this class. This is a single forward LSTM.

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


class TDLSTM(LSTM):
    '''
    Attributes:

    1. include_target -- Wheather to include the target in the LSTM
       representations.
    '''

    @classmethod
    def name(cls) -> str:
        return 'TDLSTM'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

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

    def _pre_process(self, data_dicts: List[Dict[str, Any]],
                     training: bool = False) -> List[np.ndarray]:
        '''
        Converts the text in the data_dicts into a List of size two
        representing the left and right context of the target word
        respectively. Each List is made up of a matrix of of integers
        representing the text as their embedding lookups. These two Lists
        are the inputs into the keras model.

        Two find the left and right contexts it uses the `spans` field of
        the dictionaries in the `data_dicts`. The `spans` field is a list of
        Tuples where each Tuple represents a occurence of the Target, each
        Tuple contains the index of the starting and ending character index
        (Expects the List to be of size 1 as there should be only one target
        per target sample. This case is not True for the
        `Dong et al. <https://aclanthology.info/papers/P14-2009/p14-2009>`_
        dataset therefore it only takes the first target instance in the
        sentence as the target).

        The texts are converted into integers using the
        :py:meth:`process_text` method.

        :param data_dicts: A list of dictonaries that contains a `text` and
                           `spans` field.
        :param training: Whether the text should be processed for training or
                         for prediction. prediction = False, training = True
        :return: A list of two contaning the left and right context of
                 the target both represented by the output of
                 :py:meth:`process_text` method.
        '''

        def context_texts(context_data_dicts: List[Dict[str, Any]]
                          ) -> Tuple[List[str], List[str]]:
            '''
            :param context_data_dicts: A list of dictonaries that contains a
                                       `text` and `spans` field.
            :return: A list of the left and right text contexts for all the
                     dictionaries.
            '''
            # Context returns all of the left and right context occurrences
            # therefore if a target is mentioned Twice and are associated then
            # for a single text two left and right occurrences are returned.
            # Thus these are a list of lists we therefore chose only the
            # first mentioned target as the paper linked to this method does
            # not specify which they used.
            left_texts = [context(data, 'left', inc_target=self.include_target)
                          for data in context_data_dicts]
            right_texts = [context(data, 'right',
                                   inc_target=self.include_target)
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

    def create_training_text(self, train_data: List[Dict[str, Any]],
                             validation_data: List[Dict[str, Any]]
                             ) -> Tuple[List[np.ndarray],
                                        List[np.ndarray]]:
        '''
        Converts the training and validation data into a format that the keras
        model can take as input.

        :param train_data: Data to be trained on. Which is a list of
                           dictionaries where each dictionary has a `text`
                           field containing text and a field `spans` containing
                           a list of Tuples where each Tuple represents a
                           occurence of the Target, each Tuple contains the
                           index of the starting and ending character index
                           (Expects the List to be of size 1 as there should
                           be only one target per target sample. This case is
                           not True for the
                           `Dong et al. <https://aclanthology.info/papers/P14-\
                           2009/p14-2009>`_ dataset therefore it only takes
                           the first target instance in the sentence as the
                           target).
        :param validation_data: Data to evaluate the model at training time.
                                Expects the same data as the `train_data`
                                parameter.
        :return: A tuple of length two containing the train and validation
                 input that are both the output of :py:meth:`_pre_process`
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

    @classmethod
    def name(cls) -> str:
        return 'TCLSTM'

    def __repr__(self) -> str:
        '''
        Name of the machine learning model.
        '''
        return self.name()

    def _pre_process(self, data_dicts: List[Dict[str, Any]],
                     training: bool = False) -> List[np.ndarray]:
        '''
        Converts the text in the data_dicts into a list of size four
        representing the left context, left targets, right context and
        right targets. Where the contexts come are the same as those from
        TDLSTM :py:meth:`bella.models.tdlstm.TDLSTM._pre_process` method.

        The targets are a matrix of size [word_embedding_dimension, pad_size]
        and each vector in the matrix is the word embedding representation
        of the target word. If the target word is made up of multiple words
        it is then the average of the words vector representation (we use the
        median as the average). Both the contexts and the target matrix are
        used as input into the keras model.

        The texts are converted into integers using the
        :py:meth:`process_text` method.

        :param data_dicts: A list of dictonaries that contains a `text` and
                           `spans` field.
        :param training: Whether the text should be processed for training or
                         for prediction. prediction = False, training = True
        :return: A list of four contaning the left context, left vectors,
                 right context, and right vectors.
        '''
        def context_median_targets(pad_size: int):
            '''
            :param pad_size: The number of timesteps within the LSTM
            :return: Matrix of size [word_embedding_dimension, pad_size] where
                     each word embedding represents the target word or if
                     multiple words make up the target the word embedding is
                     the median of the words embeddings.
            '''
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

    def create_training_text(self, train_data: List[Dict[str, Any]],
                             validation_data: List[Dict[str, Any]]
                             ) -> Tuple[List[np.ndarray],
                                        List[np.ndarray]]:
        '''
        Converts the training and validation data into a format that the keras
        model can take as input.

        :param train_data: See :py:meth:`bella.models.tdlstm.\
                           TDLSTM.create_training_text` `train_data`
                           parameter.
        :param validation_data: See :py:meth:`bella.models.tdlstm.\
                                TDLSTM.create_training_text` `validation_data`
                                parameter.
        :return: A tuple of length two containing the train and validation
                 input that are both the output of :py:meth:`_pre_process`
        '''

        return super().create_training_text(train_data, validation_data)

    def keras_model(self, num_classes: int) -> 'keras.models.Model':
        '''
        The model that represents this class. This is the same as the
        :py:meth:`bella.models.tdlstm.TDLSTM.keras_model` model, however
        the words in before inputting into the LSTM are concatenated with
        the word embedding of the target. If the target is more than one word
        then the word embedding of the target is the average (median in our
        case) embeddings of the target words.

        :param num_classes: Number of classes to predict.
        :return: Two LSTMs one forward from the left context and the other
                 backward from the right context taking into account the
                 target vector embedding.
        '''
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
