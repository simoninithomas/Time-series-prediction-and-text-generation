import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
import keras
import string

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # Let's calculate how many series we can make
    nb_of_pairs = len(series) - window_size
    print(nb_of_pairs)
    for i in range(nb_of_pairs):
        
        # init_val = p at which we began
        init_val = i

        end_val = window_size + i

        y_val = end_val

        X.append(series[init_val:end_val])
        y.append(series[y_val])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    print(X)
    print(y)
    
   
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    model = Sequential()
    #layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(5,
                   input_shape = (window_size, 1)))
    #layer 2 uses a fully connected module with one unit
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    text_char = ''.join(punctuation)
    text_char = text_char + string.ascii_lowercase
    
    # Lowercase
    text = text.lower()
        
    for t in text:
        if not t in text_char:
            text = text.replace(t, ' ')
        
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # Let's calculate how many series we can make
    nb_of_pairs = len(text) - window_size
    
    index = 0
    i = 0
    for i in range(0, nb_of_pairs, step_size):
        
        # init_val = p at which we began
        init_val = i
        end_val = init_val + window_size 
        y_val = end_val
        
        inputs.append(text[init_val:end_val])
        outputs.append(text[y_val])

    return inputs,outputs
    

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    # layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(200,
                   input_shape = (window_size, num_chars)))
    
    # layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text 
    model.add(Dense(num_chars))
    
    #layer 3 should be a softmax activation ( since we are solving a multiclass classification) Use the categorical_crossentropy loss       
    model.add(Activation("softmax"))

    return model

    




