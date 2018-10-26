import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from pathlib import Path
import sys
a_path = str(Path('..').resolve())
sys.path.insert(0, a_path)

from bella.keras_models import tclstm, test_avergaer

test_example = np.array([[0, 0, 1, 2, 0], [1,2,0,1,0]])


test_example = np.array([[0, 0, 2, 0, 1, 1], 
                          [0,0,1, 3, 1, 1]], dtype=np.int32)

test_example1 = np.array([[1, 3, 1], 
                         [1, 1, 2]], dtype=np.int32)

test_example2 = np.array([[0, 0, 3, 2, 0, 1], 
                          [2,3,2, 3, 0, 3]], dtype=np.int32)

#test_example = np.array([[0, 0, 2, 0, 1, 1]], dtype=np.int32)
#
#test_example1 = np.array([[1, 3, 1]], dtype=np.int32)

#test_example2 = np.array([[0, 0, 3, 2, 0, 1]], dtype=np.int32)


test_embedding =np.array([[0.5, 0.5, 0.5, 0.5],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2],
                          [3, 3, 3, 3]], dtype=np.float32)
#test_target_embedding = np.array([[0.2, 0.2, 0.2, 0.2],
#                          [3, 3, 3, 3],
#                          [8, 8, 8, 8],
#                          [4, 4, 4, 4]], dtype=np.float32)

example_model = test_avergaer(test_embedding)
print([layer.name for layer in example_model.layers])
print('-------------')
print(f'Input: {test_example}')
print(f'Input1: {test_example1}')
print(f'Input2: {test_example2}')
print('-------------')
func = K.function([example_model.layers[0].input, example_model.layers[3].input,
                   example_model.layers[1].input],
                  [example_model.layers[-1].output])
print(func([test_example, test_example1, test_example2]))
print(example_model.summary())
raise Exception('END')                          



#example_model = tclstm(test_embedding, train_embeddings=False, target_embedding=test_target_embedding)
#func = K.function([example_model.layers[1].input, example_model.layers[3].input,
#                   example_model.layers[0].input],
#                  [example_model.layers[8].input])
#print('-------------')
#print(f'Input: {test_example}')
#print(f'Input1: {test_example1}')
#print(f'Input: {test_example2}')
#print('-----------------')
#print([layer.name for layer in example_model.layers])
#print(f'Input: {test_example}')
#print(func([test_example, test_example1, test_example2]))
#example_model.summary()



#test_example = tf.convert_to_tensor(test_example, np.int32)
#test_example1 = tf.convert_to_tensor(test_example1, np.int32)
#test_example2 = tf.convert_to_tensor(test_example2, np.int32)
#example_model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#example_model.fit([test_example, test_example1, test_example2], np.array([1, 1]))


#test_example = np.array([[0, 0, 2, 0, 1, 1]], dtype=np.int32)
#test_example1 = np.array([[1, 3, 0, 0, 1, 0]], dtype=np.int32)
#test_example2 = np.array([[0, 0, 3, 2, 0, 1]], dtype=np.int32)