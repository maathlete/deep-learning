#source articles: https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2
#source articles: https://towardsdatascience.com/autoencoder-on-dimension-reduction-100f2c98608c

#not fully finished, will update soon

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape 

class InverseLayer(keras.layers.Layer):
    
    def __init__(self, dense, activation = None, **kwargs):
        
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        
        self.biases = self.add_weight(name = 'bias',
                                    shape = [self.dense.input_shape[-1]])
        
        super().build(input_shape)
        
        
    def call(self, inputs):
        
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b = True)
        
        return(self.activation(z + self.biases))
    
class TiedAutoEncoder():
    
    def __init__(self, input_shape, n_hidden_layers, layer_sizes):
        
        self.input_shape = input_shape
        self.n_hidden_layers = n_hidden_layers
        self.layer_sizes = layer_sizes
    
    def compile_autoencoder(self):
        
        inputs = keras.Input(shape = self.input_shape)

        h1 = Dense(8, activation = 'relu')
        h2 = Dense(4, activation = 'relu')
        h3 = Dense(2, activation = 'relu')

        x = Flatten()(inputs)

        x = h1(x)
        x = h2(x)
        encoded = h3(x)
        x = InverseLayer(h3, activation = 'relu')(encoded)
        x = InverseLayer(h2, activation = 'relu')(x)
        decoded = InverseLayer(h1, activation = 'sigmoid')(x)

        outputs = Reshape(self.input_shape)(decoded)

        tied_autoencoder = keras.Model(inputs = inputs, outputs = outputs)

        encoder = keras.Model(inputs = inputs, outputs = encoded)

        tied_autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        self.tied_autoencoder = tied_autoencoder
        self.encoder = encoder
        
    def fit(self, train, n_epochs):
        
        (self.tied_autoencoder).fit(train, train, epochs = n_epochs)
        
        self.fit = True
        
    def encode(self, data):
        
        if self.fit:
        
            return((self.encoder).predict(data))
        
        else:
            
            print('error: fit model first')
            
 tae = TiedAutoEncoder(input_shape = (features.iloc[0, 3:]).shape,
                      n_hidden_layers = 3,
                      layer_sizes = [8, 4, 2])
tae.compile_autoencoder()
tae.fit(train, 10)
