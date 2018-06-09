import random as rn
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import preprocessing, models, optimizers, initializers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import model_from_yaml

# Displaying the Neural Network models
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from bella.contexts import context
from bella.neural_pooling import matrix_median
from bella.notebook_helper import get_json_data, write_json_data

class LSTM():
    def __init__(self, tokeniser, embeddings, pad_size=-1, lower=False):
        '''
        :param tokeniser: The tokeniser function to tokenise the text.
        :param embeddings: The embeddings to use
        :param pad_size: The max number of tokens to use per sequence. If -1 \
        use the text sequence in the training data that has the most tokens as \
        the pad size.
        :param lower: Whether to lower case the words being processed.
        :param lstm_dimension: Output of the LSTM layer. If None it is the \
        which is the default then the dimension will be the same as the \
        embedding vector.
        :param optimiser: Optimiser to for the LSTM default is SGD. Accepts any \
        `keras optimiser <https://keras.io/optimizers/>`_.
        :param patience: Wether or not to use EarlyStopping default is not \
        stated by the None value. If so this is the patience value e.g. 5.
        :param batch_size: Number of samples per gradient update
        :param epochs: Number of epochs to train the model.
        :type tokeniser: function
        :type embeddings: :py:class:`bella.word_vectors.WordVectors` instance
        :type pad_size: int. Default -1
        :type lower: bool. Default False
        :returns: The instance of TLSTM
        :rtype: :py:class:`bella.models.tdlstm.TLSTM`
        '''

        self.tokeniser = tokeniser
        self.embeddings = embeddings
        self.pad_size = pad_size
        self.test_pad_size = 0
        self.lower = lower
        self.model = None

    def save_model(self, model_arch_fp, model_weights_fp, verbose=0):
        if self.model is None:
            raise ValueError('Model is not fitted please fit the model '\
                             'using the fit function')
        time_taken = time.time()
        with open(model_arch_fp, 'w') as model_arch_file:
            model_arch_file.write(self.model.to_yaml())
        self.model.save_weights(model_weights_fp)
        if verbose == 1:
            time_taken = round(time.time() - time_taken, 2)
            print('Model architecture saved to: {}\nModel weights saved to {}\n'\
                  'Save time {}'\
                  .format(model_arch_fp, model_weights_fp, time_taken))

    def load_model_dir(self, model_zoo_path: Path, dataset_name: str,
                       verbose: int = 0) -> None:
        '''
        :param model_zoo_path: File path to the model zoo directory
        :param dataset_name: Name of the dataset the pre-trained model is \
        trained on.
        :param verbose: Output of where the weights and architecture was \
        loaded from and how long loading took.
        :return: Nothing. Loads the pre-trained model to self.model.
        '''

        file_name = f'{str(self)} {dataset_name}'
        model_arch_fp = model_zoo_path / (file_name + ' architecture.yaml')
        model_weights_fp = model_zoo_path / (file_name + ' weights.h5')
        self.load_model(model_arch_fp, model_weights_fp, verbose=verbose)


    def load_model(self, model_arch_fp, model_weights_fp, verbose=0):
        time_taken = time.time()
        loaded_model = None
        with open(model_arch_fp, 'r') as model_arch_file:
            loaded_model = model_from_yaml(model_arch_file)
        if loaded_model is None:
            raise ValueError('The model architecture was not loaded')
        # load weights into new model
        loaded_model.load_weights(model_weights_fp)
        self.model = loaded_model
        if verbose == 1:
            time_taken = round(time.time() - time_taken, 2)
            print('Model architecture loaded {}\nModel weights loaded {}\n'\
                  'Load time {}'\
                  .format(model_arch_fp, model_weights_fp, time_taken))
        self.test_pad_size = loaded_model.inputs[0].shape.dims[1].value
        # self.test_pad_size = self.pad_size

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
            raise ValueError('The max length has to be either -1 or above '\
                             'zero not {}'.format(max_length))

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
                                                     dtype='int32', padding=padding,
                                                     truncating=truncate))
    @staticmethod
    def validation_split(train_data, train_y, validation_size=0.2,
                         reproducible=False):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values
        :validation_size: The fraction of the training data to be set aside \
        for validation data
        :param reproducible: Whether the validation split should be determinstic
        :type train_data: list
        :type train_y: list
        :type validation_size: float Default 0.2
        :type reproducible: bool. Default False
        :returns: A tuple of length 4 which contains: 1. Training features, \
        2. Training Target Values, 3. Validation features, 4. Validation Targets
        :rtype: tuple
        '''

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size)
        if reproducible:
            splitter = StratifiedShuffleSplit(n_splits=1,
                                              test_size=validation_size,
                                              random_state=42)
        train_data = np.asarray(train_data)
        train_y = np.asarray(train_y)
        for train_indexs, validation_index in splitter.split(train_data, train_y):
            return (train_data[train_indexs], train_y[train_indexs],
                    train_data[validation_index], train_y[validation_index])

    def create_training_y(self, train_y, validation_y):
        all_y = np.hstack((train_y, validation_y))
        # Convert the training true values into categorical data format
        num_classes = np.unique(all_y).shape[0]
        all_y = to_categorical(all_y, num_classes=num_classes)\
                .astype(np.float32)
        return all_y

    def _pre_process(self, data_dicts, training=False):
        text_data = [data['text'] for data in data_dicts]
        if training:
            if self.model is not None:
                raise ValueError('When pre-process the data for training the '\
                                 'the model should be None not {}'\
                                 .format(self.model))
            self.test_pad_size, sequence_data = self.process_text(text_data,
                                                                  self.pad_size)
            return sequence_data
        else:
            _, sequence_data = self.process_text(text_data, self.test_pad_size)
            return sequence_data

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
        all_text = np.vstack((sequence_train_data, sequence_val_data))
        return all_text


    def fit_predict(self, train_data, train_y, test_data, test_y,
                    fit_params, score_func, score_args=None, score_kwargs=None):
        '''
        Function to train, test, and return the scores and predictions.
        '''
        self.fit(train_data, train_y, **fit_params)
        predictions = self.predict(test_data)
        if score_args is None:
            score_args = []
        if score_kwargs is None:
            score_kwargs = {}
        score = self.score(test_y, predictions, score_func, *score_args,
                           **score_kwargs)
        return score, predictions

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




    def cross_val(self, train_data, train_y, scorer, dataset_name, search_param,
                  cv=5, kfold_reproducible=True, results_file=None,
                  re_write=False, **fit_kwargs):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values of the training data
        :param dataset_name: Name of the dataset being analyzed
        :param search_param: Name of the parameter being analyzed
        :param cv: Number of folds the cross validation performs
        :param scorer: The scoring function to perform each fold. \
        The function must take the true targets as the first parameter and \
        predicted targets as the second parameter. e.g sklearn.metrics.f1_score
        :param kfold_reproducible: Whether the train and validation splits are random \
        or not.
        :param results_file: Path to the file that the results will be saved to \
        if results do not need saving keep default value None.
        :param re_write: If saving data determines if to re-write over previous \
        results
        :param fit_kwargs: key word arguments to pass to the fit function.
        :type train_data: list
        :type train_y: list
        :type dataset_name: String
        :type search_param: String
        :type cv: int. Default 5
        :type scorer: function
        :type kfold_reproducible: bool. Default True
        :type results_file: String. Default None
        :type re_write: bool. Default False
        :type fit_kwargs: dict
        :returns: A tuple of length 2 where the 1. list of raw prediction values \
        2. list of scores produced from the scorer.
        :rtype: tuple
        '''

        results = {}
        if results_file is not None:
            results = get_json_data(results_file, dataset_name)
            if search_param in results and not re_write:
                return results[search_param]

        splitter = StratifiedKFold(n_splits=cv)
        if not kfold_reproducible:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True)
        train_data = np.asarray(train_data)
        train_y = np.asarray(train_y)
        all_predictions = []
        scores = []
        for train_index, test_index in splitter.split(train_data, train_y):
            sub_train_data = train_data[train_index]
            sub_train_y = train_y[train_index]
            sub_test_data = train_data[test_index]
            sub_test_y = train_y[test_index]
            train_test_param = (sub_train_data, sub_train_y, sub_test_data,
                                sub_test_y, fit_kwargs, scorer)
            score, predictions = self.fit_predict(*train_test_param)
            scores.append(score)
            all_predictions.append(predictions.tolist())
        self.model = None
        result = (scores, all_predictions)
        results[search_param] = result
        if results_file is not None:
            write_json_data(results_file, dataset_name, results)
        return result

    @staticmethod
    def prediction_to_cats(true_values, pred_values, mapper=None):
        num_classes = pred_values.shape[1]
        if num_classes != len(set(true_values)):
            raise ValueError('The number of classes in the test data {} is '\
                             'is different to the number of classes in the '\
                             'train data {}'\
                             .format(len(set(true_values)), num_classes))
        # Convert the true values to the same format as the predicted values
        norm_true_values = to_categorical(true_values, num_classes=num_classes)
        # Converted both the true and predicted values from one hot encoded
        # matrix to single value vector
        norm_true_values = np.argmax(norm_true_values, axis=1)
        pred_values = np.argmax(pred_values, axis=1)
        if mapper is not None:
            pred_values = [mapper[pred_value] for pred_value in pred_values]
        return pred_values

    @staticmethod
    def score(true_values, pred_values, scorer, *args, **kwargs):
        '''
        Allows true values which are in the format as a list of Target values
        to be scored against the predicted values which are one hot encoded
        due to the output of the predict function. The score is defined by the
        scorer function.

        :param true_values: Correct Target values
        :param pred_values: Predicted Target values
        :param scorer: Scoring function. The function must take the true \
        targets as the first parameter and predicted targets as the second \
        parameter. e.g sklearn.metrics.f1_score
        :param args: Additional arguments to the scorer function
        :param kwargs: Additional key word arguments to the scorer function
        :type true_values: array
        :type pred_values: array
        :type scorer: function
        :returns: The output from the scorer based on the true and predicted \
        values normally a float.
        :rtype: scorer output
        '''
        num_classes = pred_values.shape[1]
        if num_classes != len(set(true_values)):
            raise ValueError('The number of classes in the test data {} is '\
                             'is different to the number of classes in the '\
                             'train data {}'\
                             .format(len(set(true_values)), num_classes))
        # Convert the true values to the same format as the predicted values
        norm_true_values = to_categorical(true_values, num_classes=num_classes)
        # Converted both the true and predicted values from one hot encoded
        # matrix to single value vector
        norm_true_values = np.argmax(norm_true_values, axis=1)
        pred_values = np.argmax(pred_values, axis=1)
        return scorer(norm_true_values, pred_values, *args, **kwargs)

    @staticmethod
    def _to_be_reproducible(reproducible):
        if reproducible:
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(42)
            rn.seed(42)
            # Forces tensorflow to use only one thread
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1)
            tf.set_random_seed(1234)

            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
        else:
            np.random.seed(None)
            rn.seed(np.random.randint(0, 400))
            tf.set_random_seed(np.random.randint(0, 400))


    def fit(self, train_data, train_y, validation_size=0.2, verbose=0,
            reproducible=True, embedding_layer_trainable=False,
            lstm_dimension=None, optimiser=None, patience=None,
            batch_size=32, epochs=100, org_initialisers=True):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values of the training data
        :param validation_size: The fraction of the training data to be set aside \
        for validation data
        :param verbose: Verbosity of the traning the model. 0=silent, \
        1=progress bar, and 2=one line per epoch
        :param reproducible: Wether or not to make the model to be reproducible. \
        This will slow done the training.
        :param embedding_layer_trainable: Whether the word embeddings weights \
        are updated during training.
        :param lstm_dimension: Output of the LSTM layer. If None it is the \
        which is the default then the dimension will be the same as the \
        embedding vector.
        :param optimiser: Optimiser to for the LSTM default is SGD. Accepts any \
        `keras optimiser <https://keras.io/optimizers/>`_.
        :param patience: Wether or not to use EarlyStopping default is not \
        stated by the None value. If so this is the patience value e.g. 5.
        :param batch_size: Number of samples per gradient update
        :param epochs: Number of epochs to train the model.
        :param org_initialisers: Whether to use the original weight initializers \
        that were stated in the paper. If False then use Keras default initializers.
        :type train_data: list
        :type train_y: list
        :type validation_size: float. Default 0.2
        :type verbose: int. Default 1
        :type reproducible: bool. Default True.
        :type embedding_layer_trainable: bool. Default False
        :type lstm_dimension: int. Default None
        :type optimiser: Keras optimiser. Default None which uses SDG.
        :type patience: int. Default None.
        :type batch_size: int. Default 32
        :type epochs: int. Default 100.
        :type org_initialisers: bool. Default True
        :returns: Nothing. The self.model will be fitted.
        :rtype: None
        '''

        self.model = None
        self._to_be_reproducible(reproducible)

        # Data pre-processing
        data = self.validation_split(train_data, train_y,
                                     validation_size=validation_size,
                                     reproducible=reproducible)
        temp_train, temp_train_y, validation_data, validation_y = data
        all_data = self.create_training_text(temp_train, validation_data)
        all_y = self.create_training_y(temp_train_y, validation_y)
        num_classes = all_y.shape[1]

        # LSTM model
        embedding_matrix = self.embeddings.embedding_matrix
        vocab_size, vector_size = embedding_matrix.shape
        if lstm_dimension is None:
            lstm_dimension = vector_size
        # Model layers
        input_layer = layers.Input(shape=(self.test_pad_size,),
                                   name='text_input')
        embedding_layer = layers\
                          .Embedding(input_dim=vocab_size,
                                     output_dim=vector_size,
                                     input_length=self.test_pad_size,
                                     trainable=embedding_layer_trainable,
                                     weights=[embedding_matrix],
                                     name='embedding_layer')(input_layer)
        lstm_layer = layers.LSTM(lstm_dimension,
                                 name='lstm_layer')(embedding_layer)
        prediction_layer = layers.Dense(num_classes, activation='softmax',
                                        name='output')(lstm_layer)
        if org_initialisers:
            uniform_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
            lstm_init = {'kernel_initializer' : uniform_init,
                         'recurrent_initializer' : uniform_init,
                         'bias_initializer' : uniform_init}
            dense_init = {'kernel_initializer' : uniform_init,
                          'bias_initializer' : uniform_init}
            embedding_init = {'embeddings_initializer' : uniform_init}
            # Model layers
            embedding_layer = layers\
                              .Embedding(input_dim=vocab_size,
                                         output_dim=vector_size,
                                         input_length=self.test_pad_size,
                                         trainable=embedding_layer_trainable,
                                         weights=[embedding_matrix],
                                         name='embedding_layer',
                                         **embedding_init)(input_layer)
            lstm_layer = layers.LSTM(lstm_dimension, name='lstm_layer',
                                     **lstm_init)(embedding_layer)
            prediction_layer = layers.Dense(num_classes, activation='softmax',
                                            name='output', **dense_init)\
                                           (lstm_layer)

        model = models.Model(inputs=input_layer, outputs=prediction_layer)

        if optimiser is None:
            optimiser = optimizers.SGD(lr=0.01)
        model.compile(optimizer=optimiser, metrics=['accuracy'],
                      loss='categorical_crossentropy')
        with tempfile.NamedTemporaryFile() as weight_file:
            # Set up the callbacks
            callbacks = None
            if patience is not None:
                model_checkpoint = ModelCheckpoint(weight_file.name,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='min')
                early_stopping = EarlyStopping(monitor='val_loss', mode='min',
                                               patience=patience)
                callbacks = [early_stopping, model_checkpoint]
            history = model.fit(all_data, all_y, validation_split=validation_size,
                                epochs=epochs, callbacks=callbacks,
                                verbose=verbose, batch_size=batch_size)
            # Load the best model from the saved weight file
            if patience is not None:
                model.load_weights(weight_file.name)
        self.model = model
        return history

    def predict(self, test_data, sentiment_mapper=None):
        '''
        :param test_y: Test features. Specifically a list of dict like \
        structures that contain `text` key.
        :type test_y: list
        :returns: A list of predicted samples for the test data.
        :rtype: numpy.ndarray
        '''

        if self.model is None:
            raise ValueError('The model has not been fitted please run the '
                             '`fit` method.')
        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.
        sequence_test_data = self._pre_process(test_data, training=False)
        pred_values = self.model.predict(sequence_test_data)
        if sentiment_mapper is None:
            return pred_values
        pred_labels = np.argmax(pred_values, axis=1)
        pred_labels = [sentiment_mapper[pred_label] for
                       pred_label in pred_labels]
        return np.array(pred_labels)



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

    def __repr__(self):
        return 'LSTM'

class TDLSTM(LSTM):

    def __init__(self, tokeniser, embeddings, pad_size=-1, lower=False,
                 inc_target=True):
        '''
        :param pad_size: Applies to both the right and left hand side. However \
        if -1 is set then the left and right maximum pad size is found \
        independently.
        :type pad_size: int
        '''
        super().__init__(tokeniser, embeddings, pad_size=pad_size, lower=lower)
        self.left_pad_size = pad_size
        self.left_test_pad_size = 0
        self.right_pad_size = pad_size
        self.right_test_pad_size = 0
        self.inc_target = inc_target

    def load_model(self, model_arch_fp, model_weights_fp, verbose=0):
        super().load_model(model_arch_fp, model_weights_fp, verbose=verbose)
        self.left_test_pad_size = self.model.inputs[0].shape.dims[1].value
        self.right_test_pad_size = self.model.inputs[1].shape.dims[1].value


    def predict(self, test_data, sentiment_mapper=None):
        '''
        :param test_y: Test features. Specifically a list of dict like \
        structures that contain `text` key.
        :type test_y: list
        :returns: A list of predicted samples for the test data.
        :rtype: numpy.ndarray
        '''

        if self.model is None:
            raise ValueError('The model has not been fitted please run the '\
                             '`fit` method.')
        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.
        left_sequence, right_sequence = self._pre_process(test_data,
                                                          training=False)

        pred_values = self.model.predict({'left_text_input': left_sequence,
                                          'right_text_input': right_sequence})
        if sentiment_mapper is None:
            return pred_values
        pred_labels = np.argmax(pred_values, axis=1)
        pred_labels = [sentiment_mapper[pred_label] for
                       pred_label in pred_labels]
        return np.array(pred_labels)

    def _pre_process(self, data_dicts, training=False):

        def context_texts(context_data_dicts):
            # Context returns all of the left and right context occurrences
            # therefore if a target is mentioned Twice and are associated then
            # for a single text two left and right occurrences are returned.
            # Thus these are a list of lists we therefore chose only the
            # first mentioned target as the paper linked to this method does
            # not specify which they used.
            left_texts = [context(data, 'left', inc_target=self.inc_target) \
                         for data in context_data_dicts]
            right_texts = [context(data, 'right', inc_target=self.inc_target) \
                           for data in context_data_dicts]
            left_texts = [texts[0] for texts in left_texts]
            right_texts = [texts[0] for texts in right_texts]
            return left_texts, right_texts

        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.

        # Get left and right contexts
        left_text, right_text = context_texts(data_dicts)
        if training:
            if self.model is not None:
                raise ValueError('When pre-process the data for training the '\
                                 'the model should be None not {}'\
                                 .format(self.model))
            left_pad_sequence = self.process_text(left_text, self.left_pad_size)
            self.left_test_pad_size, left_sequence = left_pad_sequence

            right_pad_sequence = self.process_text(right_text, self.right_pad_size,
                                                   padding='post', truncate='post')
            self.right_test_pad_size, right_sequence = right_pad_sequence
            return left_sequence, right_sequence
        else:
            left_pad_sequence = self.process_text(left_text,
                                                  self.left_test_pad_size)
            _, left_sequence = left_pad_sequence

            right_pad_sequence = self.process_text(right_text,
                                                   self.right_test_pad_size,
                                                   padding='post', truncate='post')
            _, right_sequence = right_pad_sequence
            return left_sequence, right_sequence


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

        train_sequences  = self._pre_process(train_data, training=True)
        left_sequence_train, right_sequence_train = train_sequences
        validation_sequences  = self._pre_process(validation_data, training=False)
        left_sequence_val, right_sequence_val = validation_sequences

        # Stack the validation data with the training data to complie with Keras.
        left_data = np.vstack((left_sequence_train, left_sequence_val))
        right_data = np.vstack((right_sequence_train, right_sequence_val))
        return left_data, right_data

    def fit(self, train_data, train_y, validation_size=0.2, verbose=0,
            reproducible=True, embedding_layer_trainable=False,
            lstm_dimension=None, optimiser=None, patience=None,
            batch_size=32, epochs=100, org_initialisers=True):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values of the training data
        :param validation_size: The fraction of the training data to be set aside \
        for validation data
        :param verbose: Verbosity of the traning the model. 0=silent, \
        1=progress bar, and 2=one line per epoch
        :param reproducible: Wether or not to make the model to be reproducible. \
        This will slow done the training.
        :param embedding_layer_trainable: Whether the word embeddings weights \
        are updated during training.
        :param lstm_dimension: Output of the LSTM layer. If None it is the \
        which is the default then the dimension will be the same as the \
        embedding vector.
        :param optimiser: Optimiser to for the LSTM default is SGD. Accepts any \
        `keras optimiser <https://keras.io/optimizers/>`_.
        :param patience: Wether or not to use EarlyStopping default is not \
        stated by the None value. If so this is the patience value e.g. 5.
        :param batch_size: Number of samples per gradient update
        :param epochs: Number of epochs to train the model.
        :param org_initialisers: Whether to use the original weight initializers \
        that were stated in the paper. If False then use Keras default initializers.
        :type train_data: list
        :type train_y: list
        :type validation_size: float. Default 0.2
        :type verbose: int. Default 1
        :type reproducible: bool. Default True.
        :type embedding_layer_trainable: bool. Default False
        :type lstm_dimension: int. Default None
        :type optimiser: Keras optimiser. Default None which uses SDG.
        :type patience: int. Default None.
        :type batch_size: int. Default 32
        :type epochs: int. Default 100.
        :type org_initialisers: bool. Default True
        :returns: Nothing. The self.model will be fitted.
        :rtype: None
        '''

        self.model = None
        self._to_be_reproducible(reproducible)

        # Data pre-processing
        data = self.validation_split(train_data, train_y,
                                     validation_size=validation_size,
                                     reproducible=reproducible)
        temp_train, temp_train_y, validation_data, validation_y = data
        left_data, right_data = self.create_training_text(temp_train, validation_data)
        all_y = self.create_training_y(temp_train_y, validation_y)
        num_classes = all_y.shape[1]

        # LSTM model
        embedding_matrix = self.embeddings.embedding_matrix
        vocab_size, vector_size = embedding_matrix.shape
        if lstm_dimension is None:
            lstm_dimension = vector_size
        if optimiser is None:
            optimiser = optimizers.SGD(lr=0.01)
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
                                          name='left_embedding_layer')(left_input)
        left_lstm_layer = layers.LSTM(lstm_dimension,
                                      name='left_lstm_layer')(left_embedding_layer)
        # Right LSTM
        right_input = layers.Input(shape=(self.right_test_pad_size,),
                                   name='right_text_input')
        right_embedding_layer = layers\
                                .Embedding(input_dim=vocab_size,
                                           output_dim=vector_size,
                                           input_length=self.right_test_pad_size,
                                           trainable=embedding_layer_trainable,
                                           weights=[embedding_matrix],
                                           name='right_embedding_layer')(right_input)
        right_lstm_layer = layers.LSTM(lstm_dimension,
                                       name='right_lstm_layer')(right_embedding_layer)
        # Merge the outputs of the left and right LSTMs
        merge_layer = layers.concatenate([left_lstm_layer, right_lstm_layer],
                                         name='left_right_lstm_merge')
        predictions = layers.Dense(num_classes, activation='softmax',
                                   name='output')(merge_layer)

        if org_initialisers:
            uniform_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
            lstm_init = {'kernel_initializer' : uniform_init,
                         'recurrent_initializer' : uniform_init,
                         'bias_initializer' : uniform_init}
            dense_init = {'kernel_initializer' : uniform_init,
                          'bias_initializer' : uniform_init}
            embedding_init = {'embeddings_initializer' : uniform_init}
            # Model layers
            left_embedding_layer = layers\
                                   .Embedding(input_dim=vocab_size,
                                              output_dim=vector_size,
                                              input_length=self.left_test_pad_size,
                                              trainable=embedding_layer_trainable,
                                              weights=[embedding_matrix],
                                              name='left_embedding_layer',
                                              **embedding_init)(left_input)
            left_lstm_layer = layers.LSTM(lstm_dimension, name='left_lstm_layer',
                                          **lstm_init)(left_embedding_layer)
            right_embedding_layer = layers\
                                    .Embedding(input_dim=vocab_size,
                                               output_dim=vector_size,
                                               input_length=self.right_test_pad_size,
                                               trainable=embedding_layer_trainable,
                                               weights=[embedding_matrix],
                                               name='right_embedding_layer',
                                               **embedding_init)(right_input)
            right_lstm_layer = layers.LSTM(lstm_dimension, name='right_lstm_layer',
                                           **lstm_init)(right_embedding_layer)
            predictions = layers.Dense(num_classes, activation='softmax',
                                       name='output', **dense_init)(merge_layer)

        model = models.Model(inputs=[left_input, right_input],
                             outputs=predictions)
        model.compile(optimizer=optimiser, metrics=['accuracy'],
                      loss='categorical_crossentropy')
        with tempfile.NamedTemporaryFile() as weight_file:
            # Set up the callbacks
            callbacks = None
            if patience is not None:
                model_checkpoint = ModelCheckpoint(weight_file.name,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='min')
                early_stopping = EarlyStopping(monitor='val_loss', mode='min',
                                               patience=patience)
                callbacks = [early_stopping, model_checkpoint]
            history = model.fit([left_data, right_data], all_y,
                                validation_split=validation_size,
                                epochs=epochs, callbacks=callbacks,
                                verbose=verbose, batch_size=batch_size)
            # Load the best model from the saved weight file
            if patience is not None:
                model.load_weights(weight_file.name)
        self.model = model
        return history

    def __repr__(self):
        return 'TDLSTM'

class TCLSTM(TDLSTM):

    def __init__(self, tokeniser, embeddings, pad_size=-1, lower=False,
                 inc_target=True):
        '''
        :param pad_size: Applies to both the right and left hand side. However \
        if -1 is set then the left and right maximum pad size is found \
        independently.
        :type pad_size: int
        '''
        super().__init__(tokeniser, embeddings, pad_size=pad_size, lower=lower)
        self.left_pad_size = pad_size
        self.left_test_pad_size = 0
        self.right_pad_size = pad_size
        self.right_test_pad_size = 0
        self.inc_target = inc_target

    def predict(self, test_data, sentiment_mapper=None):
        '''
        :param test_y: Test features. Specifically a list of dict like \
        structures that contain `text` key.
        :type test_y: list
        :returns: A list of predicted samples for the test data.
        :rtype: numpy.ndarray
        '''

        if self.model is None:
            raise ValueError('The model has not been fitted please run the '
                             '`fit` method.')
        # Convert from a sequence of dictionaries into texts and then integers
        # that represent the tokens in the text within the embedding space.
        sequence_targets = self._pre_process(test_data, training=False)
        left_sequence, left_targets = sequence_targets[0], sequence_targets[1]
        right_sequence = sequence_targets[2]
        right_targets = sequence_targets[3]
        pred_values = self.model.predict({'left_text_input': left_sequence,
                                          'left_target': left_targets,
                                          'right_text_input': right_sequence,
                                          'right_target': right_targets})
        if sentiment_mapper is None:
            return pred_values
        pred_labels = np.argmax(pred_values, axis=1)
        pred_labels = [sentiment_mapper[pred_label] for
                       pred_label in pred_labels]
        return np.array(pred_labels)

    def load_model(self, model_arch_fp, model_weights_fp, verbose=0):
        super().load_model(model_arch_fp, model_weights_fp, verbose=verbose)
        self.left_test_pad_size = self.model.inputs[0].shape.dims[1].value
        self.right_test_pad_size = self.model.inputs[2].shape.dims[1].value

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
        return (left_sequence, left_target_vectors,
                right_sequence, right_target_vectors)

    def create_training_data(self, train_data, validation_data):
        '''
        :param train_data: :param train_data: Training features. Specifically \
        a list of dict like structures that contain `target` key.
        '''

        train_seq_targ = self._pre_process(train_data, training=True)
        validation_seq_targ = self._pre_process(validation_data, training=False)

        train_l_seq, train_l_targ, train_r_seq, train_r_targ = train_seq_targ
        val_l_seq, val_l_targ, val_r_seq, val_r_targ = validation_seq_targ

        left_sequences = np.vstack((train_l_seq, val_l_seq))
        right_sequences = np.vstack((train_r_seq, val_r_seq))
        left_targets = np.vstack((train_l_targ, val_l_targ))
        right_targets = np.vstack((train_r_targ, val_r_targ))

        return left_sequences, left_targets, right_sequences, right_targets


    def fit(self, train_data, train_y, validation_size=0.2, verbose=1,
            reproducible=True, embedding_layer_trainable=False,
            lstm_dimension=None, optimiser=None, patience=None,
            batch_size=32, epochs=100, org_initialisers=True):
        '''
        :param train_data: Training features. Specifically a list of dict like \
        structures that contain `text` key.
        :param train_y: Target values of the training data
        :param validation_size: The fraction of the training data to be set aside \
        for validation data
        :param verbose: Verbosity of the traning the model. 0=silent, \
        1=progress bar, and 2=one line per epoch
        :param reproducible: Wether or not to make the model to be reproducible. \
        This will slow done the training.
        :param embedding_layer_trainable: Whether the word embeddings weights \
        are updated during training.
        :param lstm_dimension: Output of the LSTM layer. If None it is the \
        which is the default then the dimension will be the same as the \
        embedding vector.
        :param optimiser: Optimiser to for the LSTM default is SGD. Accepts any \
        `keras optimiser <https://keras.io/optimizers/>`_.
        :param patience: Wether or not to use EarlyStopping default is not \
        stated by the None value. If so this is the patience value e.g. 5.
        :param batch_size: Number of samples per gradient update
        :param epochs: Number of epochs to train the model.
        :param org_initialisers: Whether to use the original weight initializers \
        that were stated in the paper. If False then use Keras default initializers.
        :type train_data: list
        :type train_y: list
        :type validation_size: float. Default 0.2
        :type verbose: int. Default 1
        :type reproducible: bool. Default True.
        :type embedding_layer_trainable: bool. Default False
        :type lstm_dimension: int. Default None
        :type optimiser: Keras optimiser. Default None which uses SDG.
        :type patience: int. Default None.
        :type batch_size: int. Default 32
        :type epochs: int. Default 100.
        :type org_initialisers: bool. Default True
        :returns: Nothing. The self.model will be fitted.
        :rtype: None
        '''

        self.model = None
        self._to_be_reproducible(reproducible)

        # Data pre-processing
        data = self.validation_split(train_data, train_y,
                                     validation_size=validation_size,
                                     reproducible=reproducible)
        temp_train, temp_train_y, validation_data, validation_y = data
        sequence_targets = self.create_training_data(temp_train, validation_data)
        left_data, left_targets, right_data, right_targets = sequence_targets
        all_y = self.create_training_y(temp_train_y, validation_y)
        num_classes = all_y.shape[1]

        # LSTM model
        embedding_matrix = self.embeddings.embedding_matrix
        vocab_size, vector_size = embedding_matrix.shape
        if lstm_dimension is None:
            # Double the vector size as we have to take into consideration the
            # concatenated target vector
            lstm_dimension = 2 * vector_size
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
                                          name='left_embedding_layer')(left_input)
        left_target_input = layers.Input(shape=(self.left_test_pad_size, vector_size),
                                         name='left_target')
        left_text_target = layers.concatenate([left_embedding_layer,
                                               left_target_input],
                                              name='left_text_target')
        left_lstm_layer = layers.LSTM(lstm_dimension,
                                      name='left_lstm')(left_text_target)
        # Right LSTM
        right_input = layers.Input(shape=(self.right_test_pad_size,),
                                   name='right_text_input')
        right_embedding_layer = layers\
                                .Embedding(input_dim=vocab_size,
                                           output_dim=vector_size,
                                           input_length=self.right_test_pad_size,
                                           trainable=embedding_layer_trainable,
                                           weights=[embedding_matrix],
                                           name='right_embedding_layer')(right_input)
        right_target_input = layers.Input(shape=(self.right_test_pad_size, vector_size),
                                          name='right_target')
        right_text_target = layers.concatenate([right_embedding_layer,
                                                right_target_input],
                                               name='right_text_target')
        right_lstm_layer = layers.LSTM(lstm_dimension,
                                       name='right_lstm')(right_text_target)
        # Merge the outputs of the left and right LSTMs
        merge_layer = layers.concatenate([left_lstm_layer, right_lstm_layer],
                                         name='left_right_lstm_merge')
        predictions = layers.Dense(num_classes, activation='softmax',
                                   name='output')(merge_layer)
        if org_initialisers:
            uniform_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
            lstm_init = {'kernel_initializer' : uniform_init,
                         'recurrent_initializer' : uniform_init,
                         'bias_initializer' : uniform_init}
            dense_init = {'kernel_initializer' : uniform_init,
                          'bias_initializer' : uniform_init}
            embedding_init = {'embeddings_initializer' : uniform_init}
            # Model layers
            left_embedding_layer = layers\
                                   .Embedding(input_dim=vocab_size,
                                              output_dim=vector_size,
                                              input_length=self.left_test_pad_size,
                                              trainable=embedding_layer_trainable,
                                              weights=[embedding_matrix],
                                              name='left_embedding_layer',
                                              **embedding_init)(left_input)
            left_lstm_layer = layers.LSTM(lstm_dimension, name='left_lstm',
                                          **lstm_init)(left_text_target)

            right_embedding_layer = layers\
                                    .Embedding(input_dim=vocab_size,
                                               output_dim=vector_size,
                                               input_length=self.right_test_pad_size,
                                               trainable=embedding_layer_trainable,
                                               weights=[embedding_matrix],
                                               name='right_embedding_layer',
                                               **embedding_init)(right_input)
            right_lstm_layer = layers.LSTM(lstm_dimension, name='right_lstm',
                                           **lstm_init)(right_text_target)
            predictions = layers.Dense(num_classes, activation='softmax',
                                       name='output', **dense_init)(merge_layer)


        input_layers = [left_input, left_target_input,
                        right_input, right_target_input]
        model = models.Model(inputs=input_layers, outputs=predictions)
        if optimiser is None:
            optimiser = optimizers.SGD(lr=0.01)
        model.compile(optimizer=optimiser, metrics=['accuracy'],
                      loss='categorical_crossentropy')
        with tempfile.NamedTemporaryFile() as weight_file:
            # Set up the callbacks
            callbacks = None
            if patience is not None:
                model_checkpoint = ModelCheckpoint(weight_file.name,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='min')
                early_stopping = EarlyStopping(monitor='val_loss', mode='min',
                                               patience=patience)
                callbacks = [early_stopping, model_checkpoint]
            input_data = {'left_text_input' : left_data,
                          'left_target' : left_targets,
                          'right_text_input' : right_data,
                          'right_target' : right_targets}
            history = model.fit(input_data, all_y, validation_split=validation_size,
                                epochs=epochs, callbacks=callbacks,
                                verbose=verbose, batch_size=batch_size)
            # Load the best model from the saved weight file
            if patience is not None:
                model.load_weights(weight_file.name)
        self.model = model
        return history

    def __repr__(self):
        return 'TCLSTM'
