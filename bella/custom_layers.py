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
        x = tf.Print(x, [mask], message='Next step ', summarize=4*8)
        avg = self.mask_average(x, mask)
        avg = tf.Print(avg, [avg], message='Average ', summarize=6*4)
        return avg
        

    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0], input_shape[-1])
    
    def compute_mask(self, x, mask):
        if not self.mask_zero:
            return None
        return mask

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

    def call(self, x, mask=None, expansion_shape=None):
        if expansion_shape is None:
            average_expansion = tf.ones(tf.shape(x), name='Intial_Expansion')
        else:
            average_expansion = tf.ones(tf.shape(expansion_shape), 
                                        name='Intial_Expansion')
        average_expansion_shape = tf.shape(average_expansion)
        avg = super().call(x, mask)
        avg = tf.Print(avg, [average_expansion_shape], message='Yes ')
        avg = tf.Print(avg, [avg], message='Average1 ', summarize=6*4)
        average_expansion = tf.multiply(average_expansion, avg, 
                                        name='Average_Expansion')
        return average_expansion

    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0], input_shape[-1])