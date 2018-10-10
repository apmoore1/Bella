from pathlib import Path
import sys
a_path = str(Path('..').resolve())
sys.path.insert(0, a_path)

import tensorflow.keras.backend as K
import numpy as np

from bella.keras_models import example

test_example = np.array([[0, 0, 1, 2, 0], [1,2,0,1,0]])
test_example = np.array([[1, 0, 2, 0, 1, 1]])
test_example1 = np.array([[0, 0, 3, 2, 0]])
print(np.size(test_example))
test_embedding =np.array([[0.5, 0.5, 0.5, 0.5],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2],
                          [3, 3, 3, 3]], dtype=np.float32)

example_model = example(test_embedding)
func = K.function([example_model.layers[0].input, example_model.layers[1].input],
                  [example_model.layers[-1].input])
print([layer.name for layer in example_model.layers])
print(f'Input: {test_example}')
func([test_example, test_example1])
example_model.summary()