import tensorflow as tf
from tensorflow import keras
class Average(keras.layers.Layer):
    '''
    Reference
    ---------
    https://github.com/ruidan/Aspect-level-sentiment/blob/master/code/my_layers.py
    '''
    def __init__(self, mask_zero=True, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.supports_masking = True

    def mask_average(self, x, mask):
        if mask is not None: #self.mask_zero:
            mask = tf.cast(mask, 'float32')
            mask = tf.expand_dims(mask, axis=-1)
            x_mask = tf.multiply(x, mask, name='X_Masked')
            # Reducing each word vector into one dimension from n dimensions
            mask_sum = tf.reduce_sum(mask, axis=1, name='Mask_Summed')
            x_sum = tf.reduce_sum(x_mask, axis=1, name='X_Summed')
            return x_sum / (mask_sum + 1e-07)
        else:
            return tf.reduce_mean(x, axis=1)

    def call(self, x, mask=None):
        avg = self.mask_average(x, mask)
        return avg
    
    def compute_mask(self, x, mask):
        if self.mask_zero is not None:
            return mask
        return None

class ConcatMask(keras.layers.Layer):

    def __init__(self, mask_zero=True, **kwargs):
            super().__init__(**kwargs)
            self.mask_zero = mask_zero
            self.supports_masking = True

    def call(self, x, mask=None):
        context_embeddings = x[0]
        target_embedding = x[1]
        return tf.concat([context_embeddings, target_embedding], 2)
    
    def compute_mask(self, x, mask):
        if self.mask_zero is not None:
            return mask[0]
        return None

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Expand(keras.layers.Layer):

    def __init__(self, mask_zero=True, **kwargs):
            super().__init__(**kwargs)
            self.mask_zero = mask_zero
            self.supports_masking = True

    def call(self, x, mask=None):
        return tf.expand_dims(x, axis=1)
    
    def compute_mask(self, x, mask):
        if self.mask_zero is not None:
            mask
        return None

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Ones(keras.layers.Layer):

    def __init__(self, mask_zero=True, **kwargs):
            super().__init__(**kwargs)
            self.mask_zero = mask_zero
            self.supports_masking = True

    def call(self, x, mask=None):
        return tf.ones(tf.shape(x))
    
    def compute_mask(self, x, mask):
        if self.mask_zero is not None:
            mask
        return None

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Not needed anymore
class ExpandedAverage(Average):
    '''
    Calculates the Average and then expands the average vector so that it 
    has an average vector at each time step according to the length of the 
    input sequence.

    Reference
    ---------
    https://github.com/ruidan/Aspect-level-sentiment/blob/master/code/my_layers.py
    '''
    def __init__(self, mask_zero=True, **kwargs):
        super().__init__(mask_zero=mask_zero, **kwargs)

    def call(self, y, mask=None):#, expansion_shape=None):
        #if expansion_shape is None:
        #    average_expansion = tf.ones(tf.shape(x), name='Intial_Expansion')
        #else:
        x, expansion_shape = y
        mask = tf.Print(mask, [mask], message='Mask: ', summarize=6*4)
        mask = tf.Print(mask, [x], message='x: ', summarize=6*4)
        mask = tf.Print(mask, [expansion_shape], message='Sequence: ', summarize=6*4)
        average_expansion = tf.ones(tf.shape(expansion_shape), 
                                    name='Intial_Expansion')
        avg = super().call(x, mask)
        avg = tf.expand_dims(avg, axis=1)
        average_expansion = tf.multiply(average_expansion, avg, 
                                        name='Average_Expansion')
        return average_expansion