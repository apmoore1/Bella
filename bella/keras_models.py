#from keras.layers import Embedding, Input, Dense
#from keras.models import Model

#from bella.custom_layers import Average

#def example(initial_embedding):
#    num_words, embedding_dim = initial_embedding.shape
#    target_text_input = Input(batch_shape=(None, None), dtype='int32',
#                              name='target_text')
#    target_embedding = Embedding(input_dim=num_words, output_dim=embedding_dim,
#                                 mask_zero=True, weights=[initial_embedding], 
#                                 trainable=False, 
#                                 name='target_embedding')(target_text_input)
#    average = Average()(target_embedding)
#    out = Dense(3, activation='softmax')(average)
#    return Model(inputs=target_text_input, outputs=out)

from tensorflow.keras.layers import Embedding, Input, Dense, TimeDistributed,\
                                    concatenate, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

from bella.custom_layers import Average, ExpandedAverage

#def print_information(x):


def example(initial_embedding):
    num_words, embedding_dim = initial_embedding.shape
    embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                mask_zero=True, weights=[initial_embedding], 
                                trainable=False, 
                                name='shared_embedding')
    
    
    target_text_input = Input(batch_shape=(None, None), dtype='int32',
                              name='target_text')
    target_embedding = embedding_layer(target_text_input)

    context_text_input = Input(batch_shape=(None, None), dtype='int32',
                               name='context_text')
    context_embedding = embedding_layer(context_text_input)


    target_average = ExpandedAverage(mask_zero=False, name='Target_Average')\
                                    (target_embedding, expansion_shape=context_embedding)

    
    context_target_embeddings = concatenate([context_embedding, target_average])
    context_average = Average(name='Context_Average')(context_target_embeddings)
    out = Dense(3, activation='softmax', name='Out')(context_average)
    return Model(inputs=[target_text_input, context_text_input], outputs=out)