from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, Input, Dense, LSTM, concatenate,\
                                    LSTM, multiply, Lambda
from tensorflow.keras.models import Model
#from tensorflow.train import Optimizer, AdamOptimizer
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.utils import Sequence

from bella.custom_layers import Average, ConcatMask, Expand, Ones

def test_avergaer(initial_embedding):
    def multi_mask_lambda(inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None
    def context_target(context_embeddings, average_target_embedding):
        context_ones = Ones()(context_embeddings)
        average_expanded = Expand()(average_target_embedding)
        average_expanded = multiply([context_ones, average_expanded])
        return ConcatMask()([context_embeddings, average_expanded])
        
        #context_shape_layer = Lambda(lambda x: tf.ones(tf.shape(x)), 
        #                             mask=True)
        #expand_dims_layer = Lambda(lambda x: tf.expand_dims(x, axis=1), 
        #                           mask=True)
        #context_ones = context_shape_layer(context_embeddings)
        #average_expanded = expand_dims_layer(average_target_embedding)
        #average_expanded = multiply([context_ones, average_expanded])
        
        #return Lambda(lambda x: tf.concat(x, 2), mask=multi_mask_lambda)([context_embeddings, average_expanded])

    def embedding_and_target(input_layer, embedded_target_average, embedder):
        embedded_input = embedder(input_layer)
        context = context_target(embedded_input, embedded_target_average)
        return Average()(context) 

    num_words, embedding_dim = initial_embedding.shape
    embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                mask_zero=True, weights=[initial_embedding], 
                                trainable=False, 
                                name='shared_embedding')
    
    target_input = Input(batch_shape=(None, None), dtype='int32',
                         name='target_text')
    
    target_embedding = embedding_layer(target_input)
    target_average = Average()(target_embedding)

    left_input = Input(batch_shape=(None, None), dtype='int32', 
                       name='left_text')
    left_embedding = embedding_and_target(left_input, target_average, 
                                          embedding_layer)

    right_input = Input(batch_shape=(None, None), dtype='int32', 
                        name='right_text')
    right_embedding = embedding_and_target(right_input, target_average,
                                           embedding_layer)
    out = concatenate([left_embedding, right_embedding])
    
    return Model(inputs=[left_input, right_input, target_input], 
                 outputs=out)

def tdlstm(initial_embedding, num_classes=3, lstm_size=300,
           train_embeddings=True, dropout=0.0):
    
    def text_to_lstm(input_layer, embedder, go_backwards):
        embedded_input = embedder(input_layer)
        encoding = LSTM(lstm_size, go_backwards=go_backwards, 
                        dropout=dropout, recurrent_dropout=dropout)(embedded_input)
        return encoding
    
    num_words, embedding_dim = initial_embedding.shape
    embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                mask_zero=True, weights=[initial_embedding], 
                                trainable=train_embeddings, 
                                name='shared_embedding')

    # Left text to (text, target) embeddings
    left_input = Input(batch_shape=(None, None), dtype='int32', 
                       name='left_text')
    left_encoding = text_to_lstm(left_input, embedding_layer, False)

    # Right text to (text, target) embeddings
    right_input = Input(batch_shape=(None, None), dtype='int32', 
                        name='right_text')
    right_encdoing = text_to_lstm(right_input, embedding_layer, True)

    combined_encoding = concatenate([left_encoding, right_encdoing])

    out = Dense(num_classes, activation='softmax', name='out')(combined_encoding)
    return Model(inputs=[left_input, right_input], outputs=out)


def tclstm(initial_embedding, num_classes=3, lstm_size=300, 
           train_embeddings=True, target_embedding=None, dropout=0.0,
           train_target_embeddings=True):

    def context_target(context_embeddings, average_target_embedding):
        context_ones = Ones()(context_embeddings)
        average_expanded = Expand()(average_target_embedding)
        average_expanded = multiply([context_ones, average_expanded])
        return ConcatMask()([context_embeddings, average_expanded])

    def text_to_lstm(input_layer, target_average, 
                     embedder, go_backwards):
        embedded_input = embedder(input_layer)
        context = context_target(embedded_input, target_average)
        encoding = LSTM(lstm_size, go_backwards=go_backwards,
                        dropout=dropout, recurrent_dropout=dropout)(context)
        return encoding

    num_words, embedding_dim = initial_embedding.shape
    embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                mask_zero=True, weights=[initial_embedding], 
                                trainable=train_embeddings, 
                                name='shared_embedding')

    # Target encoding
    target_input = Input(batch_shape=(None, None), dtype='int32',
                         name='target_text')
    if target_embedding is None:
        target_embedding = embedding_layer(target_input)
    else:
        num_target_words, target_embedding_dim = target_embedding.shape
        target_embedding_layer = Embedding(input_dim=num_target_words, 
                                           output_dim=target_embedding_dim,
                                           mask_zero=True, 
                                           weights=[target_embedding], 
                                           trainable=train_target_embeddings, 
                                           name='target_embedding')
        target_embedding = target_embedding_layer(target_input)
    target_average = Average()(target_embedding)
    
    # Left text to (text, target) embeddings
    left_input = Input(batch_shape=(None, None), dtype='int32', 
                       name='left_text')
    left_encoding = text_to_lstm(left_input, target_average, embedding_layer, 
                                 False)

    # Right text to (text, target) embeddings
    right_input = Input(batch_shape=(None, None), dtype='int32', 
                        name='right_text')
    right_encdoing = text_to_lstm(right_input, target_average, embedding_layer,
                                  True)

    combined_encoding = concatenate([left_encoding, right_encdoing])
    out = Dense(num_classes, activation='softmax', name='out')(combined_encoding)
    return Model(inputs=[left_input, right_input, target_input], outputs=out)
    

def fit(model: Model, train_generator: Sequence, 
        validation_generator: Sequence, save_path: Path, patience: int = 5, 
        verbose: int = 0, optimizer: Optimizer = Adam(),
        **fit_kwargs):
    '''
    Wrapper around the fit_generator method that will also comile the model 
    and create useful callbacks.
    '''

    model_checkpoint = ModelCheckpoint(filepath=save_path, save_best_only=True, 
                                       monitor='val_loss', 
                                       save_weights_only=True, verbose=verbose)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                   verbose=verbose)
    callbacks = [model_checkpoint, early_stopping]

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(train_generator, verbose=verbose, 
                                  validation_data=validation_generator, 
                                  callbacks=callbacks, shuffle=True,
                                  **fit_kwargs)    
    model.load_weights(save_path)
    model.save_weights(save_path, save_format='h5')
    return history